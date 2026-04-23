"""Run a frozen locomotion policy on a pre-computed optimal trajectory.

Architecture (no additional training required):

    P* (from OCP solver)
        │
        ▼  geometric controller (Section IV-A of the paper)
    (v_x, 0, ω_z)  ──►  πloco (frozen)  ──►  joint actions  ──►  Go2 sim
        │
        └── logged as tracking error  e_t = min_k ‖p_t − p*_k‖

Usage:
    python scripts/track_trajectory.py \\
        --checkpoint logs/rsl_rl/go2_base/<run>/model_<iter>.pt \\
        --trajectory path/to/traj.npz \\
        [--num_envs 1] [--video] [--video_length 1000] [--Kp_yaw 2.0]

To generate a trajectory first:
    python scripts/optimal_trajectory_solver.py \\
        --start 0 0 0 --goal 5 3 --save traj.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trajectory tracking with frozen locomotion policy.")
parser.add_argument("--checkpoint", required=True,
                    help="Path to trained πloco checkpoint (.pt) or directory containing model_*.pt")
parser.add_argument("--trajectory", required=True,
                    help="Path to P* .npz file from optimal_trajectory_solver.py")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments (default: 1)")
parser.add_argument("--video", action="store_true", help="Record an mp4 video")
parser.add_argument("--video_length", type=int, default=2000, help="Max frames to record")
parser.add_argument("--Kp_yaw", type=float, default=2.0,
                    help="Proportional gain for heading error in geometric controller")
parser.add_argument("--eps_wp", type=float, default=0.15,
                    help="Waypoint advance threshold ε_wp [m]")
parser.add_argument("--history_length", type=int, default=0,
                    help="Must match the value used during training (0 = no history wrapper)")
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

import imageio
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.leggedloco.config.go2.go2_low_base_cfg import Go2RoughPPORunnerCfg
from omni.isaac.leggedloco.config.go2.go2_trajectory_play_cfg import Go2TrajectoryPlayCfg
from omni.isaac.leggedloco.leggedloco.mdp.commands import TrajectoryCommandGeneratorCfg
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

sys.path.insert(0, os.path.dirname(__file__))
from optimal_trajectory_solver import load_trajectory


def _resolve_checkpoint(path: str) -> str:
    """Accept either a direct .pt path or a run directory."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt"))
        if not pts:
            raise FileNotFoundError(f"No model_*.pt files found in {path}")
        chosen = os.path.join(path, pts[-1])
        print(f"[INFO] Auto-selected checkpoint: {chosen}")
        return chosen
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def main():
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)
    traj_path = args_cli.trajectory

    # ── Verify trajectory exists and print summary ────────────────────────────
    traj = load_trajectory(traj_path)
    print(f"[INFO] Loaded P*: {traj.M + 1} waypoints, T*={traj.total_time:.2f}s, "
          f"goal={traj.positions[-1].round(2)}")

    # ── Build env config ──────────────────────────────────────────────────────
    env_cfg = Go2TrajectoryPlayCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(
        robot_attr="robot",
        trajectory_file=traj_path,
        eps_wp=args_cli.eps_wp,
        Kp_yaw=args_cli.Kp_yaw,
        v_max=0.5,
        omega_max=1.0,
    )

    # ── Create env ────────────────────────────────────────────────────────────
    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    # ── Load agent config from training run (agent.yaml lives next to checkpoint) ──
    agent_cfg: RslRlOnPolicyRunnerCfg = Go2RoughPPORunnerCfg()
    agent_yaml = os.path.join(os.path.dirname(checkpoint_path), "..", "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        update_class_from_dict(agent_cfg, load_yaml(agent_yaml))
        print(f"[INFO] Loaded agent config from {agent_yaml}")
    else:
        print("[WARN] agent.yaml not found — using default Go2RoughPPORunnerCfg")

    # ── Load policy ───────────────────────────────────────────────────────────
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] Loaded policy from {checkpoint_path}")

    # ── Camera setup ──────────────────────────────────────────────────────────
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
    goal = traj.positions[-1]
    midpoint = (robot_pos[:2] + goal) / 2
    env.unwrapped.sim.set_camera_view(
        eye=(midpoint[0], midpoint[1] - 8.0, 6.0),
        target=(midpoint[0], midpoint[1], 0.0),
    )

    # ── Inference loop ────────────────────────────────────────────────────────
    obs, _ = env.get_observations()
    frames = []
    step = 0

    # Grab reference to the command term for live metric logging
    cmd_term = env.unwrapped.command_manager._terms.get("base_velocity")

    print("\n[INFO] Starting trajectory tracking.  Ctrl-C to stop.\n")
    print(f"{'step':>6}  {'e_t [m]':>9}  {'wp':>5}/{traj.M}  "
          f"{'vx_cmd':>8}  {'oz_cmd':>8}  {'goal_reached':>12}")
    print("-" * 65)

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        step += 1

        # ── Log tracking metrics ──────────────────────────────────────────
        if cmd_term is not None and step % 50 == 0:
            e_t = cmd_term.tracking_error()
            k   = cmd_term.active_waypoint_idx
            cmd = cmd_term.command
            gr  = cmd_term.goal_reached
            print(
                f"{step:>6}  {e_t[0].item():>9.3f}  {k[0].item():>5}  "
                f"{cmd[0, 0].item():>8.3f}  {cmd[0, 2].item():>8.3f}  "
                f"{'YES' if gr[0] else 'no':>12}"
            )

        # ── Video capture ─────────────────────────────────────────────────
        if args_cli.video and len(frames) < args_cli.video_length:
            frames.append(env.unwrapped.render())

        # Stop when all envs have reached the goal or episode times out
        if cmd_term is not None and cmd_term.goal_reached.all():
            print("\n[INFO] All environments reached the goal.")
            break

        if args_cli.video and len(frames) >= args_cli.video_length:
            break

    # ── Save video ────────────────────────────────────────────────────────────
    if args_cli.video and frames:
        out_dir = os.path.dirname(checkpoint_path)
        out_path = os.path.join(out_dir, "trajectory_tracking.mp4")
        writer = imageio.get_writer(out_path, fps=50)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"\n[INFO] Video saved to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
