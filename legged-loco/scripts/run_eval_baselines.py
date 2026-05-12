"""Baseline evaluation — Section V metrics.

Compares three methods over N episodes each:
  hybrid     — Learned visual trigger + NaVILA avoidance  (proposed)
  threshold  — Geometric d_obs / e_track trigger + NaVILA (Baseline 2)
  no_vla     — Pure trajectory tracking, no VLA           (Baseline 4)

Metrics (Section V):
  rho_J   — Cost overhead ratio  J_hat / J*
  SPL     — Success-weighted path length
  N_vla   — VLA calls per episode
  T_wall  — Total wall-clock mission time (sim time + VLA latency)

Usage (from legged-loco/ in isaaclab conda env):

    # Learned trigger:
    CUDA_VISIBLE_DEVICES=1 \\
    /srv/local/ganeshr3/conda/envs/isaaclab/bin/python scripts/run_eval_baselines.py \
        --method threshold --scalar_trigger_checkpoint /srv/local/ganeshr3/ae598-arp-fp/checkpoints/trigger_real_scalar.pt \
        --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
        --trajectory traj.npz \
        --model_path ../NaVILA/checkpoints/navila-llama3-8b-8f \
        --n_episodes 5 --history_length 9 --headless --enable_cameras

    # Scalar MLP trigger (Baseline 2):
    ... --method threshold --scalar_trigger_checkpoint checkpoints/trigger_real_scalar.pt

    # No VLA:
    ... --method no_vla
"""

"""Launch Isaac Sim first."""

import argparse
import os
import subprocess
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Section V baseline evaluation.")
parser.add_argument("--method", required=True, choices=["hybrid", "threshold", "no_vla"],
                    help="hybrid=learned trigger, threshold=geometric trigger, no_vla=tracker only")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--trajectory", required=True)
parser.add_argument("--model_path",
                    default="../NaVILA/checkpoints/navila-llama3-8b-8f",
                    help="NaVILA model checkpoint (not needed for no_vla)")

# Episode configuration
parser.add_argument("--n_episodes", type=int, default=15,
                    help="Number of episodes to run (obstacle placed at different traj fractions)")
parser.add_argument("--obstacle_fracs", type=float, nargs="+", default=None,
                    help="Exact trajectory fractions for obstacle. Overrides --n_episodes linspace.")
parser.add_argument("--obstacle_lateral_m", type=float, default=0.0,
                    help="Lateral offset of obstacle from trajectory [m] (default: on-path)")
parser.add_argument("--collision_dist", type=float, default=0.4,
                    help="Obstacle contact distance [m]")
parser.add_argument("--max_steps", type=int, default=8000)

# Trigger parameters
parser.add_argument("--trigger_checkpoint",
                    default="/srv/local/ganeshr3/ae598-arp-fp/checkpoints/trigger_real_visual.pt",
                    help="Visual trigger checkpoint (hybrid mode)")
parser.add_argument("--scalar_trigger_checkpoint",
                    default="/srv/local/ganeshr3/ae598-arp-fp/checkpoints/trigger_real_scalar.pt",
                    help="Scalar-only trigger checkpoint (threshold mode, Baseline 2)")
parser.add_argument("--trigger_threshold", type=float, default=0.98,
                    help="Trigger MLP sigmoid threshold (both hybrid and threshold modes)")
parser.add_argument("--trigger_every",     type=int,   default=5)
parser.add_argument("--trigger_cooldown",  type=int,   default=150)

# Avoidance / FSM
parser.add_argument("--handoff_clearance", type=float, default=1.0)
parser.add_argument("--max_avoiding_steps", type=int, default=1000)
parser.add_argument("--navila_server_host", default="localhost")
parser.add_argument("--navila_server_port", type=int, default=15432)

# Cost function weights (Section V.a)
parser.add_argument("--wt", type=float, default=1.0,  help="Time penalty weight")
parser.add_argument("--wd", type=float, default=0.5,  help="Path length weight")
parser.add_argument("--we", type=float, default=0.01, help="Control effort weight (||torques||^2)")
parser.add_argument("--l_fail", type=float, default=50.0,
                    help="Terminal penalty for failed episodes")
parser.add_argument("--v_nominal", type=float, default=0.5,
                    help="Nominal speed [m/s] used to estimate J* = wt*L/v + wd*L")
parser.add_argument("--t_bar_vla", type=float, default=1.5,
                    help="Mean VLA inference latency [s] for T_wall computation")

# Policy
parser.add_argument("--history_length", type=int, default=9)
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true")

# Video
parser.add_argument("--video", action="store_true", help="Record mp4 per episode")
parser.add_argument("--video_fps", type=int, default=50)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.method in ("hybrid", "threshold"):
    args_cli.enable_cameras = True  # camera needed for NaVILA avoidance

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

import csv
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
from omni.isaac.leggedloco.config.go2.go2_trigger_eval_cfg import OBSTACLE_SIZE
from omni.isaac.leggedloco.leggedloco.mdp.commands import (
    HybridCommandGeneratorCfg, TrajectoryCommandGeneratorCfg,
)
from omni.isaac.leggedloco.leggedloco.mdp.commands.return_to_path_command_generator import ReturnToPathCfg
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

sys.path.insert(0, os.path.dirname(__file__))
from optimal_trajectory_solver import load_trajectory, plan_obstacle_free_trajectory

_OBSTACLE_H = OBSTACLE_SIZE[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt"))
        if pts:
            c = os.path.join(path, pts[-1])
            print(f"[INFO] Auto-selected checkpoint: {c}")
            return c
    raise FileNotFoundError(path)


def _teleport_obstacle(env_raw, xy, z):
    obs = env_raw.scene["obstacle"]
    state = obs.data.default_root_state.clone()
    state[:, 0] = float(xy[0]); state[:, 1] = float(xy[1]); state[:, 2] = z
    state[:, 3:7] = torch.tensor([1., 0., 0., 0.]); state[:, 7:] = 0.
    obs.write_root_state_to_sim(state)


def _start_navila_server(model_path, host, port):
    navila_python = "/srv/local/ganeshr3/conda/envs/navila/bin/python"
    script  = os.path.join(os.path.dirname(__file__), "navila_server.py")
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
            raise RuntimeError("navila_server.py exited before printing READY")
    raise RuntimeError("navila_server stdout closed before READY")


def _traj_length(traj) -> float:
    """Total arc-length of optimal trajectory P*."""
    pos = traj.positions  # (M+1, 2) or (M+1, 3)
    return float(sum(np.linalg.norm(pos[k + 1][:2] - pos[k][:2]) for k in range(len(pos) - 1)))


def _obstacle_positions(traj, fracs, lateral_m=0.0):
    """Return list of (xy, z) tuples for obstacle placement."""
    M = traj.M
    positions = []
    for frac in fracs:
        k = int(np.clip(frac * M, 0, M - 1))
        pos_xy = traj.positions[k][:2].copy()
        if lateral_m != 0.0:
            # Perpendicular offset
            if k < M:
                tang = traj.positions[k + 1][:2] - traj.positions[k][:2]
                tang /= (np.linalg.norm(tang) + 1e-8)
                perp = np.array([-tang[1], tang[0]])
                pos_xy += lateral_m * perp
        positions.append((pos_xy.astype(np.float32), _OBSTACLE_H / 2.0))
    return positions


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(episodes: list, L_traj: float) -> dict:
    """Compute aggregate Section V metrics from per-episode result dicts."""
    wt, wd, we, v_nom = args_cli.wt, args_cli.wd, args_cli.we, args_cli.v_nominal
    t_bar_vla, l_fail  = args_cli.t_bar_vla, args_cli.l_fail

    # J* — cost of the reference plan (obstacle-free, optimal speed)
    T_star = L_traj / v_nom
    J_star = wt * T_star + wd * L_traj   # effort term ≈ 0 on optimal path

    rho_J_ep, spl_ep, nvla_ep, twall_ep = [], [], [], []

    for r in episodes:
        T_i     = r["time_s"]
        p_i     = r["path_length_m"]
        eff_i   = r["effort_sum"]
        nvla_i  = r["n_vla"]
        success = r["success"]
        d_goal  = r["d_goal_final"]

        # ρ_J (Eq. 10-11)
        l_term = 0.0 if success else l_fail + d_goal  # penalise residual distance too
        J_hat  = wt * T_i + wd * p_i + we * eff_i + l_term
        rho_J_ep.append(J_hat / J_star)

        # SPL (Eq. 12)
        S_i   = 1.0 if success else 0.0
        spl_i = S_i * L_traj / max(p_i, L_traj)
        spl_ep.append(spl_i)

        # N_vla (Eq. 13)
        nvla_ep.append(float(nvla_i))

        # T_wall (Eq. 14): sim time + estimated VLA stall time
        twall_ep.append(T_i + nvla_i * t_bar_vla)

    def _stats(xs):
        a = np.array(xs)
        return float(np.mean(a)), float(np.std(a))

    rho_mean, rho_std   = _stats(rho_J_ep)
    spl_mean, spl_std   = _stats(spl_ep)
    nv_mean,  nv_std    = _stats(nvla_ep)
    tw_mean,  tw_std    = _stats(twall_ep)
    success_rate        = float(np.mean([r["success"] for r in episodes]))

    return {
        "J_star":       J_star,
        "rho_J":        (rho_mean, rho_std),
        "spl":          (spl_mean, spl_std),
        "n_vla":        (nv_mean,  nv_std),
        "t_wall":       (tw_mean,  tw_std),
        "success_rate": success_rate,
        "per_episode":  episodes,
        "rho_J_list":   rho_J_ep,
        "spl_list":     spl_ep,
        "nvla_list":    nvla_ep,
        "twall_list":   twall_ep,
    }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    ep_idx: int,
    env,
    env_raw,
    cmd_term,
    policy,
    obs_xy: np.ndarray,
    obs_z: float,
    sim_dt: float,
    goal_xy,        # torch.Tensor (2,)
    obs_xy_t,       # torch.Tensor (2,)
    device,
    needs_obstacle: bool,
    is_hybrid: bool,
    log_dir: str | None = None,
) -> dict:
    """Run one episode and return metric dict."""
    # Reset env and teleport obstacle
    env.reset()
    obs, _ = env.get_observations()  # use wrapper path so history buffer is applied
    if needs_obstacle:
        _teleport_obstacle(env_raw, obs_xy, obs_z)

    robot = env_raw.scene["robot"]
    prev_pos = robot.data.root_pos_w[0, :2].cpu().numpy().copy()

    path_length_m = 0.0
    effort_sum    = 0.0
    step          = 0
    terminated_by = "timeout"
    d_goal        = float("inf")
    robot_path    = [prev_pos.copy()]   # accumulate per-step positions
    frames        = []                  # video frames (empty when --video not set)

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        step += 1

        if args_cli.video:
            frames.append(env_raw.render())

        curr_pos  = robot.data.root_pos_w[0, :2].cpu().numpy()
        path_length_m += float(np.linalg.norm(curr_pos - prev_pos))
        prev_pos  = curr_pos.copy()
        robot_path.append(curr_pos.copy())

        # Control effort (||u||^2 per step)
        torques    = robot.data.applied_torque[0]   # (num_joints,)
        effort_sum += float((torques ** 2).sum().cpu()) * sim_dt

        robot_pos_t = robot.data.root_pos_w[0, :2]
        d_goal  = float(torch.norm(robot_pos_t - goal_xy))
        d_obs   = float(torch.norm(robot_pos_t - obs_xy_t))

        # Progress heartbeat every 500 steps
        if step % 500 == 0:
            state = env_raw.extras.get("hybrid_state_name", "?")
            n_vla_so_far = int(env_raw.extras.get("navila_queries", 0)) if is_hybrid else 0
            print(f"    [ep {ep_idx}] step={step}/{args_cli.max_steps}  "
                  f"d_goal={d_goal:.2f}m  d_obs={d_obs:.2f}m  "
                  f"state={state}  n_vla={n_vla_so_far}", flush=True)

        # Termination conditions
        if is_hybrid and bool(getattr(cmd_term, "goal_reached", torch.zeros(1)).all()):
            terminated_by = "goal_reached"; break
        if d_goal < 0.3:
            terminated_by = "goal_reached"; break
        if needs_obstacle and d_obs < args_cli.collision_dist:
            terminated_by = "collision"; break
        if dones[0]:
            terminated_by = "fall"; break
        if step >= args_cli.max_steps:
            break

    success = terminated_by == "goal_reached"
    n_vla   = int(env_raw.extras.get("navila_queries", 0)) if is_hybrid else 0

    print(f"  ep {ep_idx:2d}  {terminated_by:<14}  "
          f"d_goal={d_goal:.3f}m  path={path_length_m:.2f}m  "
          f"steps={step}  n_vla={n_vla}")

    if args_cli.video and frames and log_dir is not None:
        vid_path = os.path.join(log_dir, f"ep{ep_idx:02d}.mp4")
        writer = imageio.get_writer(vid_path, fps=args_cli.video_fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  [video] Saved → {vid_path}")

    return {
        "episode":        ep_idx,
        "success":        success,
        "terminated_by":  terminated_by,
        "steps":          step,
        "time_s":         step * sim_dt,
        "path_length_m":  path_length_m,
        "effort_sum":     effort_sum,
        "n_vla":          n_vla,
        "d_goal_final":   d_goal,
        "obs_xy":         obs_xy.tolist(),
        "robot_path":     np.array(robot_path),   # (steps+1, 2)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    traj = load_trajectory(args_cli.trajectory)
    M    = traj.M
    L_traj = _traj_length(traj)
    print(f"[INFO] Trajectory: {M+1} waypoints, L={L_traj:.2f}m, "
          f"goal={traj.positions[-1][:2].round(2)}")

    # Episode obstacle positions
    if args_cli.obstacle_fracs is not None:
        fracs = args_cli.obstacle_fracs
    else:
        fracs = np.linspace(0.35, 0.65, args_cli.n_episodes).tolist()
    obs_positions = _obstacle_positions(traj, fracs, args_cli.obstacle_lateral_m)
    print(f"[INFO] {len(obs_positions)} episodes, obstacle fracs: "
          f"{[round(f, 2) for f in fracs]}")

    # NaVILA server (not needed for no_vla)
    navila_proc = None
    if args_cli.method != "no_vla":
        navila_proc = _start_navila_server(
            os.path.abspath(args_cli.model_path),
            args_cli.navila_server_host,
            args_cli.navila_server_port,
        )

    # Create log dir early so videos can be saved during the run
    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs",
                           "eval_baselines", args_cli.method, run_id)
    os.makedirs(log_dir, exist_ok=True)

    try:
        results = _run_all(traj, L_traj, obs_positions, navila_proc, log_dir)
    finally:
        if navila_proc is not None:
            navila_proc.terminate(); navila_proc.wait()
            print("[INFO] NaVILA server terminated.")

    # Compute and print metrics
    metrics = compute_metrics(results, L_traj)
    _print_results(metrics)
    out_path = os.path.join(log_dir, "results.json")
    # Strip numpy arrays before JSON serialisation
    results_json = [{k: v.tolist() if hasattr(v, "tolist") else v
                     for k, v in r.items() if k != "robot_path"}
                    for r in results]
    with open(out_path, "w") as f:
        json.dump({"method": args_cli.method, "metrics": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in metrics.items()
            if k != "per_episode"
        }, "episodes": results_json, "args": vars(args_cli)}, f, indent=2)
    print(f"[INFO] Results saved → {out_path}")

    # Per-episode summary CSV
    csv_path = os.path.join(log_dir, "episodes.csv")
    fields = ["episode", "success", "terminated_by", "steps", "time_s",
              "path_length_m", "effort_sum", "n_vla", "d_goal_final"]
    with open(csv_path, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    print(f"[INFO] Per-episode CSV → {csv_path}")

    # Trajectory + obstacle spatial CSV  (one row per waypoint / path point)
    # Columns: type, episode, step, x, y
    #   type=ref_traj   — reference trajectory waypoints (episode=-1)
    #   type=robot_path — actual robot path per episode
    #   type=obstacle   — obstacle xy per episode (single row)
    traj_ref = load_trajectory(args_cli.trajectory)
    spatial_path = os.path.join(log_dir, "spatial.csv")
    with open(spatial_path, "w", newline="") as sf:
        w = csv.writer(sf)
        w.writerow(["type", "episode", "step", "x", "y"])
        # Reference trajectory
        for k, (px, py) in enumerate(traj_ref.positions[:, :2]):
            w.writerow(["ref_traj", -1, k, f"{px:.4f}", f"{py:.4f}"])
        # Per-episode data
        for r in results:
            ep = r["episode"]
            ox, oy = r["obs_xy"]
            w.writerow(["obstacle", ep, 0, f"{ox:.4f}", f"{oy:.4f}"])
            for step_i, (px, py) in enumerate(r["robot_path"]):
                w.writerow(["robot_path", ep, step_i, f"{px:.4f}", f"{py:.4f}"])
    print(f"[INFO] Spatial CSV → {spatial_path}")


def _update_novla_trajectory(cmd_term, traj, obs_xy: np.ndarray, device):
    """Replan an RRT* obstacle-free trajectory and hot-swap it into cmd_term."""
    start    = traj.positions[0][:2].astype(float)
    goal     = traj.positions[-1][:2].astype(float)
    hdg0     = float(traj.headings[0])
    obs_r    = 0.65  # obstacle half-diagonal (≈0.35m) + robot radius (≈0.25m) + margin

    print(f"  [no_vla] Planning RRT* detour around obstacle at {obs_xy.round(2)} ...",
          end="", flush=True)
    ep_traj = plan_obstacle_free_trajectory(
        start=start, goal=goal, start_heading=hdg0,
        obs_xy=obs_xy, obs_radius=obs_r,
        n_waypoints=traj.M,      # same resolution as reference
        v_max=args_cli.v_nominal,
        rrt_max_iter=4000,
    )
    print(f" done  L={np.linalg.norm(np.diff(ep_traj.positions, axis=0), axis=1).sum():.2f}m"
          f"  T={ep_traj.times[-1]:.1f}s", flush=True)

    # Hot-swap trajectory positions in-place
    new_pos = torch.tensor(ep_traj.positions[:, :2], dtype=torch.float32, device=device)
    cmd_term._traj_pos = new_pos
    cmd_term._M        = len(new_pos) - 1


def _run_all(traj, L_traj, obs_positions, navila_proc, log_dir):
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)
    method = args_cli.method
    is_hybrid   = method in ("hybrid", "threshold")
    needs_obstacle = True  # obstacle always present (needed for collision detection)

    # ── Build env config ───────────────────────────────────────────────────
    env_cfg = Go2HybridEvalCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    if method == "no_vla":
        # Replace command term with pure trajectory tracker
        env_cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(
            robot_attr="robot",
            trajectory_file=args_cli.trajectory,
        )
    else:
        hcfg = env_cfg.commands.base_velocity   # HybridCommandGeneratorCfg
        hcfg.trajectory_file    = args_cli.trajectory
        hcfg.trigger_threshold  = args_cli.trigger_threshold
        hcfg.trigger_every      = args_cli.trigger_every
        hcfg.trigger_cooldown   = args_cli.trigger_cooldown
        hcfg.handoff_clearance  = args_cli.handoff_clearance
        hcfg.max_avoiding_steps = args_cli.max_avoiding_steps
        hcfg.navila_server_host = args_cli.navila_server_host
        hcfg.navila_server_port = args_cli.navila_server_port
        hcfg.return_cfg         = ReturnToPathCfg()
        if method == "threshold":
            # Baseline 2: scalar-only MLP (no visual features)
            hcfg.trigger_checkpoint = args_cli.scalar_trigger_checkpoint
        else:
            # Proposed: visual MLP (ResNet-18 + scalar features)
            hcfg.trigger_checkpoint = args_cli.trigger_checkpoint

    # Extend episode length to cover max_steps
    env_cfg.episode_length_s = (args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation + 60.0)

    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    env_raw = env.unwrapped
    sim_dt  = env_raw.cfg.sim.dt * env_raw.cfg.decimation

    # Position overhead-isometric camera once the scene is ready
    if args_cli.video:
        start_xy = traj.positions[0][:2]
        goal_xy_np = traj.positions[-1][:2]
        mid = (start_xy + goal_xy_np) / 2.0
        span = float(np.linalg.norm(goal_xy_np - start_xy))
        env_raw.sim.set_camera_view(
            eye=(float(mid[0]), float(mid[1]) - span * 0.7, span * 0.55),
            target=(float(mid[0]), float(mid[1]), 0.0),
        )
    device  = env_raw.device

    # ── Load πloco policy ──────────────────────────────────────────────────
    agent_cfg  = Go2RoughPPORunnerCfg()
    agent_yaml = os.path.join(os.path.dirname(checkpoint_path), "..", "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        update_class_from_dict(agent_cfg, load_yaml(agent_yaml))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Loaded policy from {checkpoint_path}")

    cmd_term = env_raw.command_manager._terms["base_velocity"]
    goal_xy  = torch.tensor(traj.positions[-1][:2], dtype=torch.float32, device=device)

    print(f"\n{'='*70}")
    print(f"  Method:      {method.upper()}")
    print(f"  Episodes:    {len(obs_positions)}")
    print(f"  L_traj:      {L_traj:.2f} m")
    print(f"  J*:          {args_cli.wt * L_traj / args_cli.v_nominal + args_cli.wd * L_traj:.3f}")
    if method == "threshold":
        print(f"  trigger:     scalar MLP ({args_cli.scalar_trigger_checkpoint})")
    print(f"{'='*70}")
    print(f"{'ep':>4}  {'result':<14}  {'d_goal':>7}  {'path':>7}  "
          f"{'steps':>6}  {'n_vla':>6}")
    print("-" * 55)

    results = []
    for ep_idx, (obs_xy, obs_z) in enumerate(obs_positions):
        obs_xy_t = torch.tensor(obs_xy, dtype=torch.float32, device=device)

        # For no_vla: replan an optimal obstacle-free path each episode and
        # hot-swap it into the command term before the episode starts.
        if method == "no_vla":
            _update_novla_trajectory(cmd_term, traj, obs_xy, device)

        r = run_episode(
            ep_idx, env, env_raw, cmd_term, policy,
            obs_xy, obs_z, sim_dt, goal_xy, obs_xy_t, device,
            needs_obstacle=needs_obstacle, is_hybrid=is_hybrid,
            log_dir=log_dir,
        )
        results.append(r)

    env.close()
    return results


def _print_results(metrics: dict):
    def _fmt(tup):
        m, s = tup
        return f"{m:.4f} ± {s:.4f}"

    print(f"\n{'='*70}")
    print(f"  METHOD: {args_cli.method.upper()}   ({len(metrics['per_episode'])} episodes)")
    print(f"  J*  =  {metrics['J_star']:.3f}")
    print(f"  ρ_J =  {_fmt(metrics['rho_J'])}   (1.0 = optimal)")
    print(f"  SPL =  {_fmt(metrics['spl'])}     (1.0 = perfect)")
    print(f"  N_vla = {_fmt(metrics['n_vla'])}")
    print(f"  T_wall = {_fmt(metrics['t_wall'])} s")
    print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"{'='*70}\n")

    print("Per-episode breakdown:")
    print(f"{'ep':>4}  {'ρ_J':>8}  {'SPL':>8}  {'N_vla':>6}  {'T_wall':>8}  {'ok':>4}")
    for i, (rj, spl, nv, tw) in enumerate(zip(
            metrics["rho_J_list"], metrics["spl_list"],
            metrics["nvla_list"],  metrics["twall_list"])):
        ok = "✓" if metrics["per_episode"][i]["success"] else "✗"
        print(f"{i:>4}  {rj:>8.4f}  {spl:>8.4f}  {nv:>6.0f}  {tw:>8.2f}  {ok:>4}")


if __name__ == "__main__":
    main()
    simulation_app.close()
