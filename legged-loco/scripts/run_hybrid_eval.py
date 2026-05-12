"""Hybrid trajectory tracking + NaVILA obstacle avoidance evaluation.

State machine:
    TRACKING ──[trigger fires]──► AVOIDING ──[handoff]──► RETURNING ──[on path]──► TRACKING

Usage (from legged-loco/ in isaaclab conda env):

    CUDA_VISIBLE_DEVICES=1 \
    /srv/local/ganeshr3/conda/envs/isaaclab/bin/python scripts/run_hybrid_eval.py \
        --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
        --trajectory traj.npz \
        --model_path ../NaVILA/checkpoints/navila-llama3-8b-8f \
        --history_length 9 \
        --headless \
        --enable_cameras

    # Regression: no trigger (should match track_trajectory.py)
    ... --no_trigger
"""

"""Launch Isaac Sim first."""

import argparse
import os
import subprocess
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hybrid trajectory+NaVILA+trigger eval.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--trajectory", required=True)
parser.add_argument("--model_path",
                    default="../NaVILA/checkpoints/navila-llama3-8b-8f")
parser.add_argument("--trigger_checkpoint",
                    default="checkpoints/trigger_real_visual.pt")
parser.add_argument("--avoidance_instruction", default="Turn left 90 degrees now.")

parser.add_argument("--trigger_threshold", type=float, default=0.990)
parser.add_argument("--trigger_every",     type=int,   default=5)
parser.add_argument("--handoff_clearance", type=float, default=1.5)
parser.add_argument("--trigger_cooldown",  type=int,   default=150)
parser.add_argument("--max_avoiding_steps",type=int,   default=1000)
parser.add_argument("--return_radius",     type=float, default=0.3)
parser.add_argument("--return_heading_tol",type=float, default=0.2)
parser.add_argument("--lookahead",         type=float, default=0.5)
parser.add_argument("--obstacle_pos", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"))
parser.add_argument("--navila_server_host", default="localhost")
parser.add_argument("--navila_server_port", type=int, default=15432)
parser.add_argument("--history_length", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--seed",       type=int, default=0)
parser.add_argument("--video",      action="store_true")
parser.add_argument("--video_length", type=int, default=10000)
parser.add_argument("--no_trigger", action="store_true",
                    help="Disable trigger — runs as pure trajectory tracker")
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

import datetime
import json
import math

import imageio
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper,
)

from omni.isaac.leggedloco.config.go2.go2_low_base_cfg import Go2RoughPPORunnerCfg
from omni.isaac.leggedloco.config.go2.go2_hybrid_eval_cfg import Go2HybridEvalCfg
from omni.isaac.leggedloco.config.go2.go2_trajectory_play_cfg import Go2TrajectoryPlayCfg
from omni.isaac.leggedloco.config.go2.go2_trigger_eval_cfg import OBSTACLE_SIZE
from omni.isaac.leggedloco.leggedloco.mdp.commands import (
    HybridCommandGeneratorCfg, TrajectoryCommandGeneratorCfg,
)
from omni.isaac.leggedloco.leggedloco.mdp.commands.return_to_path_command_generator import ReturnToPathCfg
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

sys.path.insert(0, os.path.dirname(__file__))
from optimal_trajectory_solver import load_trajectory

_OBSTACLE_H = OBSTACLE_SIZE[2]


def _resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt"))
        if pts:
            chosen = os.path.join(path, pts[-1])
            print(f"[INFO] Auto-selected: {chosen}")
            return chosen
    raise FileNotFoundError(path)


def _teleport_obstacle(env_raw, xy, z):
    obs = env_raw.scene["obstacle"]
    state = obs.data.default_root_state.clone()
    state[:, 0] = float(xy[0]); state[:, 1] = float(xy[1]); state[:, 2] = z
    state[:, 3:7] = torch.tensor([1., 0., 0., 0.]); state[:, 7:] = 0.
    obs.write_root_state_to_sim(state)


def _start_navila_server(model_path, host, port):
    navila_python = "/srv/local/ganeshr3/conda/envs/navila/bin/python"
    script = os.path.join(os.path.dirname(__file__), "navila_server.py")
    navila_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "NaVILA"))
    env = os.environ.copy()
    env["PYTHONPATH"] = navila_repo + ":" + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [navila_python, script, "--model_path", model_path, "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    print("[INFO] Waiting for NaVILA server...")
    for line in proc.stdout:
        print(f"  [navila_server] {line}", end="")
        if "READY" in line:
            return proc
        if proc.poll() is not None:
            raise RuntimeError("navila_server.py exited")
    raise RuntimeError("navila_server stdout closed before READY")


def _yaw(quat_wxyz):
    w, x, y, z = quat_wxyz.tolist()
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def main():
    traj = load_trajectory(args_cli.trajectory)
    M    = traj.M
    print(f"[INFO] Trajectory: {M+1} waypoints, goal={traj.positions[-1].round(2)}")

    # Obstacle position
    if args_cli.obstacle_pos is not None:
        obs_xy = np.array(args_cli.obstacle_pos[:2], dtype=np.float32)
        obs_z  = float(args_cli.obstacle_pos[2])
    else:
        mid_wp = int(0.5 * M)
        obs_xy = traj.positions[mid_wp].astype(np.float32)
        obs_z  = _OBSTACLE_H / 2.0
        print(f"[INFO] Auto-placed obstacle at wp {mid_wp}: {obs_xy.round(2)}")

    # Start NaVILA server (unless --no_trigger skips avoidance entirely)
    navila_proc = None
    if not args_cli.no_trigger:
        navila_proc = _start_navila_server(
            os.path.abspath(args_cli.model_path),
            args_cli.navila_server_host,
            args_cli.navila_server_port,
        )

    try:
        _run(traj, M, obs_xy, obs_z, navila_proc)
    finally:
        if navila_proc is not None:
            navila_proc.terminate(); navila_proc.wait()
            print("[INFO] NaVILA server terminated.")


def _run(traj, M, obs_xy, obs_z, navila_proc):
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)

    if args_cli.no_trigger:
        # Pure trajectory tracking — use the simpler play config
        env_cfg = Go2TrajectoryPlayCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(
            robot_attr="robot", trajectory_file=args_cli.trajectory,
        )
    else:
        env_cfg = Go2HybridEvalCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        hcfg = env_cfg.commands.base_velocity   # HybridCommandGeneratorCfg
        hcfg.trajectory_file        = args_cli.trajectory
        hcfg.trigger_checkpoint     = args_cli.trigger_checkpoint
        hcfg.trigger_threshold      = args_cli.trigger_threshold
        hcfg.trigger_every          = args_cli.trigger_every
        hcfg.trigger_cooldown       = args_cli.trigger_cooldown
        hcfg.navila_server_host     = args_cli.navila_server_host
        hcfg.navila_server_port     = args_cli.navila_server_port
        hcfg.avoidance_instruction  = args_cli.avoidance_instruction
        hcfg.handoff_clearance      = args_cli.handoff_clearance
        hcfg.max_avoiding_steps     = args_cli.max_avoiding_steps
        hcfg.return_cfg             = ReturnToPathCfg(
            return_radius=args_cli.return_radius,
            heading_tol=args_cli.return_heading_tol,
            lookahead=args_cli.lookahead,
        )

    # Extend episode length to cover max_steps
    env_cfg.episode_length_s = args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation + 60.0

    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    env_raw = env.unwrapped
    sim_dt  = env_raw.cfg.sim.dt * env_raw.cfg.decimation
    device  = env_raw.device

    # Load πloco policy
    agent_cfg = Go2RoughPPORunnerCfg()
    agent_yaml = os.path.join(os.path.dirname(checkpoint_path), "..", "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        update_class_from_dict(agent_cfg, load_yaml(agent_yaml))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Loaded πloco from {checkpoint_path}")

    cmd_term = env_raw.command_manager._terms["base_velocity"]
    obs_xy_t = torch.tensor(obs_xy, dtype=torch.float32, device=device)

    # Teleport obstacle and reset
    env.reset()
    if not args_cli.no_trigger:
        _teleport_obstacle(env_raw, obs_xy, obs_z)

    robot_start = env_raw.scene["robot"].data.root_pos_w[0].cpu().numpy()
    goal = traj.positions[-1]
    mid  = (robot_start[:2] + goal) / 2.0
    env_raw.sim.set_camera_view(eye=(mid[0], mid[1]-8., 6.), target=(mid[0], mid[1], 0.))

    obs, _ = env.get_observations()

    # Trace
    run_id    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = os.path.join(os.path.dirname(__file__), "..", "logs", "hybrid_eval", run_id)
    os.makedirs(log_dir, exist_ok=True)
    trace_f = open(os.path.join(log_dir, "trace.jsonl"), "w")

    import csv as _csv
    _CSV_FIELDS = ["step","time_s","state","d_obs_m","d_goal_m","e_track_m","wp_idx",
                   "trigger_score","trigger_fired","avoiding_steps","cooldown_remaining",
                   "navila_action","navila_queries","handoff","resume","navila_ms","trigger_ms"]
    csv_f   = open(os.path.join(log_dir, "trace.csv"), "w", newline="")
    csv_w   = _csv.DictWriter(csv_f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    csv_w.writeheader()

    frames, step = [], 0
    terminated_by = "max_steps"
    events = []   # list of (step, event_name, frame_path)

    _STATE_NAMES = {0: "TRACKING", 1: "AVOIDING", 2: "RETURNING"}

    # Goal position (last trajectory waypoint)
    goal_xy = torch.tensor(traj.positions[-1][:2], dtype=torch.float32, device=device)

    print(f"\n{'='*70}")
    print(f"  Obstacle at ({obs_xy[0]:.2f}, {obs_xy[1]:.2f}, {obs_z:.2f})")
    print(f"  Goal at     ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
    print(f"  Mode: {'NO-TRIGGER (tracker only)' if args_cli.no_trigger else 'HYBRID'}")
    print(f"{'='*70}\n")
    print(f"{'step':>6}  {'state':>10}  {'d_obs':>7}  {'d_goal':>7}  {'e_t':>7}  {'wp':>5}/{M}  (AVOIDING: act/nq, TRACKING: ts/tf)")
    print("-" * 80)

    prev_state = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        step += 1

        robot_pos  = env_raw.scene["robot"].data.root_pos_w[0]
        robot_quat = env_raw.scene["robot"].data.root_quat_w[0]
        d_obs  = float(torch.norm(robot_pos[:2] - obs_xy_t))
        d_goal = float(torch.norm(robot_pos[:2] - goal_xy))

        state_int  = int(env_raw.extras.get("hybrid_state", 0))
        state_name = env_raw.extras.get("hybrid_state_name", "TRACKING")

        # Tracking error and wp from whichever term is active
        if args_cli.no_trigger:
            e_t    = float(cmd_term.tracking_error()[0])
            wp_idx = int(cmd_term.active_waypoint_idx[0])
        else:
            e_t    = float(cmd_term.tracking_error()[0])
            wp_idx = int(cmd_term._tracker.active_waypoint_idx[0])

        # Trace entry
        entry = {
            "step":               step,
            "time_s":             round(step * sim_dt, 3),
            "state":              state_name,
            "d_obs_m":            round(d_obs, 4),
            "d_goal_m":           round(d_goal, 4),
            "e_track_m":          round(e_t, 4),
            "wp_idx":             wp_idx,
            "robot_pos":          robot_pos[:2].cpu().tolist(),
            "trigger_score":      round(float(env_raw.extras.get("trigger_score", 0.)), 4),
            "trigger_fired":      bool(env_raw.extras.get("trigger_fired_this_step", False)),
            "avoiding_steps":     int(env_raw.extras.get("avoiding_steps", 0)),
            "cooldown_remaining": int(env_raw.extras.get("cooldown_remaining", 0)),
            "navila_action":      str(env_raw.extras.get("navila_action", "")),
            "navila_queries":     int(env_raw.extras.get("navila_queries", 0)),
            "handoff":            bool(env_raw.extras.get("handoff_this_step", False)),
            "resume":             bool(env_raw.extras.get("resume_this_step", False)),
            "trigger_ms":         round(float(env_raw.extras.get("trigger_inference_ms", 0.)), 2),
            "navila_ms":          round(float(env_raw.extras.get("navila_query_ms", 0.)), 2),
        }
        trace_f.write(json.dumps(entry) + "\n")
        csv_w.writerow(entry)

        # Save frame on transition
        if state_int != prev_state:
            ev_name = f"transition_{_STATE_NAMES.get(prev_state,'?')}_to_{state_name}_step{step}"
            try:
                rgb = env_raw.scene["front_camera"].data.output["rgb"][0].cpu().numpy()
                _save_frame = True
            except Exception:
                _save_frame = False
            if _save_frame:
                fp  = os.path.join(log_dir, f"{ev_name}.png")
                from PIL import Image
                Image.fromarray(rgb[..., :3].astype(np.uint8)).save(fp)
                events.append((step, ev_name, fp))
            prev_state = state_int

        if step % 100 == 0:
            nav_act = str(env_raw.extras.get("navila_action", ""))
            nq      = int(env_raw.extras.get("navila_queries", 0))
            ts      = float(env_raw.extras.get("trigger_score", 0.))
            tf      = bool(env_raw.extras.get("trigger_fired_this_step", False))
            suffix  = f"  act={nav_act!r:14s} nq={nq:3d}" if state_name == "AVOIDING" else f"  ts={ts:.3f} tf={tf}"
            print(f"{step:>6}  {state_name:>10}  {d_obs:>7.3f}  {d_goal:>7.3f}  {e_t:>7.3f}  {wp_idx:>5}{suffix}")

        if args_cli.video and len(frames) < args_cli.video_length:
            frames.append(env_raw.render())

        # Termination
        if bool(cmd_term.goal_reached.all()) or d_goal < 0.3:
            terminated_by = "goal_reached"; break
        if dones[0]:
            terminated_by = "fall"; break
        if step >= args_cli.max_steps:
            terminated_by = "max_steps"; break
        if args_cli.video and len(frames) >= args_cli.video_length:
            break

    trace_f.close()
    csv_f.close()

    print(f"\n{'='*70}")
    print(f"  HYBRID RUN: terminated_by={terminated_by}, steps={step}, "
          f"d_goal={d_goal:.3f}m, d_obs_final={d_obs:.3f}m, e_track={e_t:.3f}m")
    if not args_cli.no_trigger and events:
        for ev_step, ev_name, _ in events:
            print(f"    step {ev_step}: {ev_name}")
    print(f"{'='*70}\n")

    if args_cli.video and frames:
        vp = os.path.join(log_dir, "video.mp4")
        w  = imageio.get_writer(vp, fps=50)
        for f in frames: w.append_data(f)
        w.close()
        print(f"[INFO] Video → {vp}")
    print(f"[INFO] Trace → {os.path.join(log_dir, 'trace.jsonl')}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
