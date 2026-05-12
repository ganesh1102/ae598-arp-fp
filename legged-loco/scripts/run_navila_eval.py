"""Run Go2 locomotion driven by NaVILA with a static obstacle on the path.

Architecture
------------
    navila_server.py  (navila conda env, TCP socket)
          ↕  JSON frames + instruction / action command
    NavilaCommandGenerator  (isaaclab env, this process)
          ↓  (v_x, v_y, ω_z) command
    πloco policy  →  Go2 sim

Usage
-----
    # From legged-loco/ in isaaclab conda env:
    CUDA_VISIBLE_DEVICES=1 \\
    /srv/local/ganeshr3/conda/envs/isaaclab/bin/python scripts/run_navila_eval.py \
        --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
        --model_path ../NaVILA/checkpoints/navila-llama3-8b-8f \
        --history_length 9 \
        --headless \
        --enable_cameras

    # With custom instruction:
    ... --instruction "Move forward, avoid the box on the right, and stop."

    # Single-query smoke-test (NaVILA queried once, command held until done):
    ... --query_every 99999
"""

"""Launch Isaac Sim first."""

import argparse
import os
import subprocess
import sys
import time

from omni.isaac.lab.app import AppLauncher

_DEFAULT_INSTRUCTION = (
    "You are a navigation module on a quadruped robot. "
    "When you encounter an obstacle, you must avoid it by going around it. "
    "The obstacle is a box located somewhere in front of you. "
    "When you are moving around the obstacle, keep a safe distance. "
    "When you are moving around the obstacle, maintain your original heading as much as possible. "
    "Once you have passed the obstacle, stop. "
)

parser = argparse.ArgumentParser(description="NaVILA evaluation on Go2.")
parser.add_argument("--checkpoint", required=True,
                    help="Path to πloco checkpoint (.pt) or run directory")
parser.add_argument("--model_path",
                    default="../NaVILA/checkpoints/navila-llama3-8b-8f",
                    help="Path to NaVILA checkpoint directory")
parser.add_argument("--instruction", default=_DEFAULT_INSTRUCTION,
                    help="Navigation instruction passed to NaVILA")
parser.add_argument("--obstacle_pos", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Obstacle centre [m]. Default: 3m ahead on X-axis at half-height.")
parser.add_argument("--server_host", default="localhost")
parser.add_argument("--server_port", type=int, default=15432)
parser.add_argument("--history_length", type=int, default=0,
                    help="Must match πloco training value")
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--collision_dist", type=float, default=0.4,
                    help="Distance [m] at which obstacle contact is declared")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=3000)
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
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.leggedloco.config.go2.go2_low_base_cfg import Go2RoughPPORunnerCfg
from omni.isaac.leggedloco.config.go2.go2_navila_eval_cfg import (
    Go2NavilaEvalCfg, OBSTACLE_SIZE,
)
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

_OBSTACLE_H = OBSTACLE_SIZE[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt"))
        if not pts:
            raise FileNotFoundError(f"No model_*.pt in {path}")
        chosen = os.path.join(path, pts[-1])
        print(f"[INFO] Auto-selected checkpoint: {chosen}")
        return chosen
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def _teleport_obstacle(env_raw, xy: np.ndarray, z_center: float):
    obs = env_raw.scene["obstacle"]
    state = obs.data.default_root_state.clone()
    state[:, 0] = float(xy[0])
    state[:, 1] = float(xy[1])
    state[:, 2] = z_center
    state[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    state[:, 7:] = 0.0
    obs.write_root_state_to_sim(state)


def _start_navila_server(model_path: str, host: str, port: int) -> subprocess.Popen:
    """Launch navila_server.py in the navila conda env, return the process."""
    navila_python = "/srv/local/ganeshr3/conda/envs/navila/bin/python"
    server_script = os.path.join(os.path.dirname(__file__), "navila_server.py")
    navila_repo    = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "NaVILA")
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = navila_repo + ":" + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [navila_python, server_script,
         "--model_path", model_path,
         "--host", host,
         "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    print("[INFO] Waiting for NaVILA server to load model...")
    for line in proc.stdout:
        print(f"  [navila_server] {line}", end="")
        if "READY" in line:
            print("[INFO] NaVILA server ready.")
            return proc
        if proc.poll() is not None:
            raise RuntimeError("navila_server.py exited unexpectedly during startup")

    raise RuntimeError("navila_server.py closed stdout before printing READY")


def _heading_error(robot_quat_w: torch.Tensor) -> float:
    """Yaw angle of robot in world frame [rad]."""
    w, x, y, z = robot_quat_w[0].tolist()
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)

    # ── Obstacle position ──────────────────────────────────────────────────
    '''if args_cli.obstacle_pos is not None:
        obs_xy = np.array(args_cli.obstacle_pos[:2], dtype=np.float32)
        obs_z  = float(args_cli.obstacle_pos[2])
    else:'''
    obs_xy = np.array([3.0, 0.0], dtype=np.float32)   # 3m ahead on X-axis
    obs_z  = _OBSTACLE_H / 2.0
    print(f"[INFO] Default obstacle at ({obs_xy[0]:.2f}, {obs_xy[1]:.2f}, {obs_z:.2f})")

    # ── Start NaVILA server ────────────────────────────────────────────────
    navila_proc = _start_navila_server(
        model_path=os.path.abspath(args_cli.model_path),
        host=args_cli.server_host,
        port=args_cli.server_port,
    )

    try:
        _run(checkpoint_path, obs_xy, obs_z, navila_proc)
    finally:
        navila_proc.terminate()
        navila_proc.wait()
        print("[INFO] NaVILA server terminated.")


def _run(checkpoint_path: str, obs_xy: np.ndarray, obs_z: float,
         navila_proc: subprocess.Popen):
    # ── Build env ──────────────────────────────────────────────────────────
    env_cfg = Go2NavilaEvalCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Override NavilaCommandGenerator settings before env construction
    nav_cfg = env_cfg.commands.base_velocity
    nav_cfg.server_host  = args_cli.server_host
    nav_cfg.server_port  = args_cli.server_port
    nav_cfg.instruction  = args_cli.instruction

    # Extend episode length so Isaac Lab's internal timer never fires before
    # our max_steps limit.  Default (~20 s) causes premature "fall" resets.
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

    # ── Load πloco policy ──────────────────────────────────────────────────
    agent_cfg: RslRlOnPolicyRunnerCfg = Go2RoughPPORunnerCfg()
    agent_yaml = os.path.join(os.path.dirname(checkpoint_path), "..", "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        update_class_from_dict(agent_cfg, load_yaml(agent_yaml))
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=device)
    print(f"[INFO] Loaded πloco from {checkpoint_path}")

    cmd_term = env_raw.command_manager._terms["base_velocity"]
    obs_xy_t = torch.tensor(obs_xy, dtype=torch.float32, device=device)

    # ── Place obstacle ─────────────────────────────────────────────────────
    env.reset()
    _teleport_obstacle(env_raw, obs_xy, obs_z)

    # ── Camera view ────────────────────────────────────────────────────────
    robot_start = env_raw.scene["robot"].data.root_pos_w[0].cpu().numpy()
    mid = (robot_start[:2] + obs_xy) / 2.0
    env_raw.sim.set_camera_view(
        eye=(mid[0], mid[1] - 8.0, 6.0),
        target=(mid[0], mid[1], 0.0),
    )

    obs, _ = env.get_observations()

    # ── Trace file setup ───────────────────────────────────────────────────
    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "navila_eval", run_id)
    os.makedirs(log_dir, exist_ok=True)
    trace_path = os.path.join(log_dir, "trace.jsonl")
    trace_f    = open(trace_path, "w")

    frames        = []
    step          = 0
    terminated_by = "max_steps"
    _final_queries = 0   # snapshot before env auto-reset can zero the counter

    print(f"\n{'='*70}")
    print(f"  Obstacle at ({obs_xy[0]:.2f}, {obs_xy[1]:.2f}, {obs_z:.2f})")
    print(f"  Instruction: {args_cli.instruction[:60]}...")
    print(f"  Trace → {trace_path}")
    print(f"{'='*70}\n")
    print(f"{'step':>6}  {'d_obs [m]':>10}  {'yaw [rad]':>10}  "
          f"{'action':>14}  {'navila_q':>8}")
    print("-" * 55)

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        step += 1

        robot_pos = env_raw.scene["robot"].data.root_pos_w[0]
        robot_quat = env_raw.scene["robot"].data.root_quat_w[0]
        d_obs = float(torch.norm(robot_pos[:2] - obs_xy_t))
        yaw   = _heading_error(robot_quat.unsqueeze(0))

        # ── Log to trace ───────────────────────────────────────────────────
        if cmd_term.last_cmd:
            entry = {
                "step":          step,
                "time_s":        step * sim_dt,
                "robot_pos":     robot_pos[:2].cpu().tolist(),
                "yaw_rad":       yaw,
                "d_obs_m":       d_obs,
                "navila_action": cmd_term.last_cmd.get("action", ""),
                "navila_raw":    cmd_term.last_raw,
                "navila_cmd":    cmd_term.last_cmd,
                "navila_queries": cmd_term.navila_queries,
                "twist":         cmd_term._twist[0].cpu().tolist(),
            }
            trace_f.write(json.dumps(entry) + "\n")

        # ── Periodic console log ───────────────────────────────────────────
        if step % 50 == 0:
            act = cmd_term.last_cmd.get("action", "—") if cmd_term.last_cmd else "—"
            print(f"{step:>6}  {d_obs:>10.3f}  {yaw:>10.3f}  "
                  f"{act:>14}  {cmd_term.navila_queries:>8}")

        # ── Video ──────────────────────────────────────────────────────────
        if args_cli.video and len(frames) < args_cli.video_length:
            frames.append(env_raw.render())

        # ── Snapshot query count before any env reset can zero it ─────────
        _final_queries = cmd_term.navila_queries

        # ── Termination ────────────────────────────────────────────────────
        if cmd_term.done:
            terminated_by = "stop"
            break
        if dones[0]:
            # Isaac Lab sets dones for both falls and its own episode timeout.
            # Our episode_length_s extension above ensures this is a real fall.
            terminated_by = "fall"
            break
        if d_obs < args_cli.collision_dist:
            terminated_by = "collision"
            break
        if step >= args_cli.max_steps:
            terminated_by = "max_steps"
            break
        if args_cli.video and len(frames) >= args_cli.video_length:
            break

    trace_f.close()

    # ── Summary ────────────────────────────────────────────────────────────
    robot_pos   = env_raw.scene["robot"].data.root_pos_w[0]
    d_obs_final = float(torch.norm(robot_pos[:2] - obs_xy_t))
    yaw_final   = _heading_error(
        env_raw.scene["robot"].data.root_quat_w[0].unsqueeze(0)
    )

    print(f"\n{'='*70}")
    print(
        f"  NAVILA RUN: terminated_by={terminated_by}, "
        f"distance_to_obstacle={d_obs_final:.3f}m, "
        f"final_heading={yaw_final:.3f}rad, "
        f"steps={step}, "
        f"navila_queries={_final_queries}"
    )
    print(f"{'='*70}\n")

    # ── Save video ─────────────────────────────────────────────────────────
    if args_cli.video and frames:
        video_path = os.path.join(log_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=50)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[INFO] Video saved to {video_path}")

    print(f"[INFO] Trace saved to {trace_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
