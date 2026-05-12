"""Run trajectory tracking with visual trigger obstacle detection.

The robot follows the optimal trajectory P* until the visual trigger fires
(obstacle detected in the path), at which point velocity is zeroed, the
episode terminates, and a SUCCESS / FAIL summary is printed.

Architecture:
    P* (OCP) → geometric controller → πloco → Go2 sim
                                                   │
                     [static obstacle on path]     │
                                                   ▼
                            VisualTrigger (ResNet-18 + MLP) ──► stop + log

Usage:
    python legged-loco/scripts/run_trigger_eval.py \\
        --checkpoint  logs/rsl_rl/go2_base/001/model_1999.pt \\
        --trajectory  traj.npz \\
        --history_length 9

    # Regression check (no trigger):
    python legged-loco/scripts/run_trigger_eval.py \\
        --checkpoint  logs/rsl_rl/go2_base/001/model_1999.pt \\
        --trajectory  traj.npz \\
        --history_length 9 \\
        --no_trigger

    # Custom obstacle position:
    python legged-loco/scripts/run_trigger_eval.py \\
        --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \\
        --trajectory traj.npz \\
        --history_length 9 \\
        --obstacle_pos 3.0 1.0 0.75
"""

"""Launch Isaac Sim first."""

import argparse
import os
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual trigger evaluation.")
parser.add_argument("--checkpoint", required=True,
                    help="Path to πloco checkpoint (.pt) or run directory")
parser.add_argument("--trajectory", required=True,
                    help="Path to P* .npz file from optimal_trajectory_solver.py")
parser.add_argument("--trigger_checkpoint",
                    default="checkpoints/trigger_real_visual.pt",
                    help="Path to trigger model checkpoint (default: checkpoints/trigger_real_visual.pt)")
parser.add_argument("--trigger_threshold", type=float, default=0.98,
                    help="Sigmoid score threshold for firing (default: 0.5)")
parser.add_argument("--trigger_every", type=int, default=5,
                    help="Run trigger every N control steps (default: 5 → 10 Hz)")
parser.add_argument("--obstacle_pos", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Obstacle centre position in world frame. "
                         "If omitted, placed at 50%% along the trajectory.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=1,
                    help="Number of episodes to run (Isaac Sim started once)")
parser.add_argument("--history_length", type=int, default=0,
                    help="Must match value used during πloco training")
parser.add_argument("--no_trigger", action="store_true",
                    help="Disable trigger (regression test — should match track_trajectory.py)")
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=2000)
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

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
from omni.isaac.leggedloco.config.go2.go2_trigger_eval_cfg import (
    Go2TriggerEvalCfg, OBSTACLE_SIZE,
)
from omni.isaac.leggedloco.leggedloco.mdp.commands import TrajectoryCommandGeneratorCfg
from omni.isaac.leggedloco.leggedloco.mdp.triggers import VisualTrigger, VisualTriggerCfg
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

sys.path.insert(0, os.path.dirname(__file__))
from optimal_trajectory_solver import load_trajectory

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _pick_obstacle_pos(traj, ep_idx: int, num_episodes: int, fixed_pos):
    """Return (obs_xy, obs_z) for this episode."""
    if fixed_pos is not None:
        return np.array(fixed_pos[:2], dtype=np.float32), float(fixed_pos[2])
    # Spread obstacles evenly from 20% to 80% along the trajectory
    M = traj.M
    frac = 0.2 + 0.6 * (ep_idx / max(num_episodes - 1, 1))
    wp = int(frac * M)
    return traj.positions[wp].astype(np.float32), _OBSTACLE_H / 2.0


def main():
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)
    traj = load_trajectory(args_cli.trajectory)
    M = traj.M
    print(f"[INFO] Loaded P*: {M + 1} waypoints, goal={traj.positions[-1].round(2)}")

    # ── Build env ──────────────────────────────────────────────────────────
    env_cfg = Go2TriggerEvalCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(
        robot_attr="robot",
        trajectory_file=args_cli.trajectory,
        eps_wp=0.15,
        lookahead_dist=1.0,
        Kp_yaw=2.0,
        v_max=0.5,
        omega_max=1.0,
    )

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

    # ── Load visual trigger ────────────────────────────────────────────────
    trigger = None
    if not args_cli.no_trigger:
        trigger_cfg = VisualTriggerCfg(
            checkpoint_path=args_cli.trigger_checkpoint,
            threshold=args_cli.trigger_threshold,
            inference_every=args_cli.trigger_every,
            device=str(device),
        )
        trigger = VisualTrigger(trigger_cfg, num_envs=args_cli.num_envs)
        print(f"[INFO] Loaded trigger from {args_cli.trigger_checkpoint}  "
              f"(threshold={args_cli.trigger_threshold}, every={args_cli.trigger_every} steps)")
    else:
        print("[INFO] --no_trigger: running baseline trajectory tracking only")

    cmd_term = env_raw.command_manager._terms.get("base_velocity")
    cam      = env_raw.scene["front_camera"] if not args_cli.no_trigger else None

    # ── Camera view (set once) ─────────────────────────────────────────────
    env.reset()
    robot_start = env_raw.scene["robot"].data.root_pos_w[0].cpu().numpy()
    goal = traj.positions[-1]
    mid = (robot_start[:2] + goal) / 2.0
    env_raw.sim.set_camera_view(
        eye=(mid[0], mid[1] - 8.0, 6.0),
        target=(mid[0], mid[1], 0.0),
    )

    # ── Multi-episode loop ─────────────────────────────────────────────────
    results = []
    num_ep = args_cli.num_episodes

    for ep in range(num_ep):
        obs_xy, obs_z = _pick_obstacle_pos(traj, ep, num_ep, args_cli.obstacle_pos)
        obs_xy_torch  = torch.tensor(obs_xy, dtype=torch.float32, device=device)
        _teleport_obstacle(env_raw, obs_xy, obs_z)

        env.reset()
        obs, _ = env.get_observations()
        if trigger is not None:
            trigger.reset()

        frames = []
        step = 0
        terminated_by       = "timeout"
        trigger_fired_step  = -1
        trigger_fired_score = 0.0
        trigger_dist_at_fire = float("inf")

        print(f"\n{'='*70}")
        print(f"  Episode {ep + 1}/{num_ep}  |  "
              f"Obstacle at ({obs_xy[0]:.2f}, {obs_xy[1]:.2f}, {obs_z:.2f})")
        print(f"  Trigger: {'DISABLED' if args_cli.no_trigger else f'threshold={args_cli.trigger_threshold}'}")
        print(f"{'='*70}\n")
        print(f"{'step':>6}  {'e_t [m]':>8}  {'wp':>5}/{M}  "
              f"{'d_obs [m]':>10}  {'score':>7}  {'fired':>5}")
        print("-" * 55)

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step += 1

            # ── Robot state ────────────────────────────────────────────────
            robot_xy = env_raw.scene["robot"].data.root_pos_w[0, :2]
            e_track  = float(cmd_term.tracking_error()[0])
            wp_idx   = int(cmd_term.active_waypoint_idx[0])
            d_obs    = float(torch.norm(robot_xy - obs_xy_torch))
            t_ep     = step * sim_dt

            # ── Visual trigger ─────────────────────────────────────────────
            score = 0.0
            fired = False
            if trigger is not None and step % args_cli.trigger_every == 0 and cam is not None:
                rgb_raw   = cam.data.output["rgb"][0].cpu().numpy()
                rgb_uint8 = rgb_raw[..., :3].astype(np.uint8)

                fired_t, score_t = trigger.step(
                    rgb         = rgb_uint8[np.newaxis],
                    d_obs       = np.array([d_obs],   dtype=np.float32),
                    e_track     = np.array([e_track], dtype=np.float32),
                    t_since_vla = np.array([t_ep],    dtype=np.float32),
                )
                fired = bool(fired_t[0])
                score = float(score_t[0])
                env_raw.extras["trigger_fired"] = fired_t
                env_raw.extras["trigger_score"] = score_t
                env_raw.extras["trigger_step"]  = step

            if step % 50 == 0:
                print(f"{step:>6}  {e_track:>8.3f}  {wp_idx:>5}  "
                      f"{d_obs:>10.3f}  {score:>7.4f}  {'YES' if fired else 'no':>5}")

            if args_cli.video and len(frames) < args_cli.video_length:
                frames.append(env_raw.render())

            if fired and d_obs < 0.65:  # Safety check to avoid false positives far from the obstacle
                cmd_term._twist[:] = 0.0
                trigger_fired_step   = step
                trigger_fired_score  = score
                trigger_dist_at_fire = d_obs
                terminated_by = "trigger"
                break
            if cmd_term.goal_reached.all():
                terminated_by = "goal_reached"
                break
            if dones[0]:
                terminated_by = "env_reset (fall/contact)"
                break
            if step >= args_cli.max_steps:
                terminated_by = "timeout"
                break
            if args_cli.video and len(frames) >= args_cli.video_length:
                break

        # ── Episode summary ────────────────────────────────────────────────
        success = terminated_by == "trigger"
        print(f"\n{'='*70}")
        if success:
            print(f"  TRIGGER FIRED at step {trigger_fired_step}, "
                  f"score={trigger_fired_score:.4f}, "
                  f"distance_to_obstacle={trigger_dist_at_fire:.3f}m  —  SUCCESS")
        else:
            print(f"  TRIGGER DID NOT FIRE  (terminated by: {terminated_by})  —  FAIL")
        print(f"{'='*70}\n")

        results.append(dict(
            ep=ep + 1,
            obs_xy=obs_xy.copy(),
            success=success,
            fired_step=trigger_fired_step,
            fired_score=trigger_fired_score,
            dist_at_fire=trigger_dist_at_fire,
            terminated_by=terminated_by,
        ))

        # Save per-episode video
        if args_cli.video and frames:
            out_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "trigger_eval")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"trigger_eval_ep{ep + 1:02d}.mp4")
            writer = imageio.get_writer(out_path, fps=50)
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"[INFO] Video saved to {out_path}")

        if not simulation_app.is_running():
            break

    # ── Aggregate summary ──────────────────────────────────────────────────
    n_success = sum(r["success"] for r in results)
    print(f"\n{'#'*70}")
    print(f"  AGGREGATE  —  {n_success}/{len(results)} episodes triggered successfully")
    print(f"{'#'*70}")
    print(f"{'ep':>4}  {'obs_x':>7}  {'obs_y':>7}  {'result':>10}  "
          f"{'step':>6}  {'score':>7}  {'d_fire [m]':>10}")
    print("-" * 58)
    for r in results:
        tag = "SUCCESS" if r["success"] else "FAIL"
        d   = f"{r['dist_at_fire']:.3f}" if r["success"] else "  —  "
        s   = f"{r['fired_score']:.4f}"  if r["success"] else "  —  "
        st  = str(r["fired_step"])        if r["success"] else "  —  "
        print(f"{r['ep']:>4}  {r['obs_xy'][0]:>7.2f}  {r['obs_xy'][1]:>7.2f}  "
              f"{tag:>10}  {st:>6}  {s:>7}  {d:>10}")
    print(f"{'#'*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
