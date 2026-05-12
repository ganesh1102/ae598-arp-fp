"""Tracker evaluation harness — runs the trajectory tracker with no obstacle.

Milestone 1: completion rate ≥ 0.9 over N seeds.

Usage (from legged-loco/, isaaclab env):
    CUDA_VISIBLE_DEVICES=1 \\
    /srv/local/ganeshr3/conda/envs/isaaclab/bin/python scripts/eval_tracker.py \\
        --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \\
        --trajectory traj.npz \\
        --history_length 9 \\
        --headless

Outputs:
    legged-loco/logs/eval_tracker/RESULTS.md
"""

"""Launch Isaac Sim first."""

import argparse
import os
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",     required=True)
parser.add_argument("--trajectory",     required=True)
parser.add_argument("--history_length", type=int, default=0)
parser.add_argument("--num_seeds",      type=int, default=10)
parser.add_argument("--max_steps",      type=int, default=2000)
parser.add_argument("--out_dir",        default="logs/eval_tracker")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.leggedloco.config.go2.go2_low_base_cfg import Go2RoughPPORunnerCfg
from omni.isaac.leggedloco.config.go2.go2_trajectory_play_cfg import Go2TrajectoryPlayCfg
from omni.isaac.leggedloco.leggedloco.mdp.commands import TrajectoryCommandGeneratorCfg
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

sys.path.insert(0, os.path.dirname(__file__))
from optimal_trajectory_solver import load_trajectory


def _resolve_checkpoint(path):
    if os.path.isfile(path): return path
    if os.path.isdir(path):
        pts = sorted(f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt"))
        if pts: return os.path.join(path, pts[-1])
    raise FileNotFoundError(path)


def main():
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)
    traj = load_trajectory(args_cli.trajectory)

    env_cfg = Go2TrajectoryPlayCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(
        robot_attr="robot", trajectory_file=args_cli.trajectory,
    )
    env_cfg.episode_length_s = args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation + 30.

    env = ManagerBasedRLEnv(cfg=env_cfg)
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    env_raw = env.unwrapped
    sim_dt  = env_raw.cfg.sim.dt * env_raw.cfg.decimation
    device  = env_raw.device

    agent_cfg = Go2RoughPPORunnerCfg()
    agent_yaml = os.path.join(os.path.dirname(checkpoint_path), "..", "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        update_class_from_dict(agent_cfg, load_yaml(agent_yaml))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    cmd_term = env_raw.command_manager._terms["base_velocity"]

    results = []
    for seed in range(args_cli.num_seeds):
        print(f"[eval_tracker] seed {seed+1}/{args_cli.num_seeds}")
        env.reset()
        obs, _ = env.get_observations()
        step = 0
        tracking_errors = []
        terminated_by = "max_steps"

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step += 1
            e_t = float(cmd_term.tracking_error()[0])
            tracking_errors.append(e_t)

            if cmd_term.goal_reached.all():
                terminated_by = "goal_reached"; break
            if dones[0]:
                terminated_by = "fall"; break
            if step >= args_cli.max_steps:
                break

        results.append({
            "seed": seed, "steps": step,
            "terminated_by": terminated_by,
            "completed": terminated_by == "goal_reached",
            "mean_e_t": float(np.mean(tracking_errors)),
            "max_e_t":  float(np.max(tracking_errors)),
            "time_s":   step * sim_dt,
        })

    env.close()

    n_done  = sum(r["completed"] for r in results)
    rate    = n_done / len(results)
    mean_e  = np.mean([r["mean_e_t"] for r in results])
    max_e   = np.max([r["max_e_t"] for r in results])
    mean_t  = np.mean([r["time_s"] for r in results if r["completed"]])

    os.makedirs(args_cli.out_dir, exist_ok=True)

    per_seed_rows = "".join(
        f"| {r['seed']} | {r['completed']} | {r['steps']} | "
        f"{r['mean_e_t']:.4f} | {r['max_e_t']:.4f} | {r['terminated_by']} |\n"
        for r in results
    )
    verdict = "PASS" if rate >= 0.9 else "FAIL"

    md = f"""# Tracker Evaluation Results

## Setup
- Checkpoint: `{checkpoint_path}`
- Trajectory: `{args_cli.trajectory}` ({traj.M+1} waypoints)
- Seeds: {args_cli.num_seeds}  |  Max steps: {args_cli.max_steps}

## Results
| Metric | Value |
|---|---|
| Completion rate | {n_done}/{len(results)} = **{rate:.2f}** |
| Mean tracking error | {mean_e:.4f} m |
| Max tracking error (over all runs) | {max_e:.4f} m |
| Mean completion time | {mean_t:.1f} s |

## Per-seed
| seed | completed | steps | mean_e_t [m] | max_e_t [m] | terminated_by |
|---|---|---|---|---|---|
{per_seed_rows}
## Milestone 1 criterion
Completion rate >= 0.9 -> **{verdict} ({rate:.2f})**
"""
    with open(os.path.join(args_cli.out_dir, "RESULTS.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"[eval_tracker] Results → {args_cli.out_dir}/RESULTS.md")


if __name__ == "__main__":
    main()
    simulation_app.close()
