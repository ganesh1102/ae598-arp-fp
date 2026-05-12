"""Return-controller evaluation harness — no IsaacLab required.

Spawns the robot at K offset poses from the reference path and runs only
the ReturnToPathController (pure Python simulation).

Milestone 5: success rate ≥ 0.9 over 9 spawn offsets.

Usage (from repo root, any env with numpy):
    python legged-loco/scripts/eval_return.py \\
        --trajectory legged-loco/traj.npz \\
        --out_dir legged-loco/logs/eval_return
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(
    _REPO_ROOT, "legged-loco", "isaaclab_exts",
    "omni.isaac.leggedloco", "omni", "isaac", "leggedloco",
    "leggedloco", "mdp", "commands",
))
from return_to_path_command_generator import ReturnToPathController, ReturnToPathCfg, _wrap_pi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory",    default="legged-loco/traj.npz")
    p.add_argument("--return_radius", type=float, default=0.3)
    p.add_argument("--heading_tol",   type=float, default=0.2)
    p.add_argument("--lookahead",     type=float, default=0.5)
    p.add_argument("--v_max",         type=float, default=0.4)
    p.add_argument("--omega_max",     type=float, default=1.0)
    p.add_argument("--max_steps",     type=int,   default=500)
    p.add_argument("--dt",            type=float, default=0.02)
    p.add_argument("--out_dir",       default="legged-loco/logs/eval_return")
    return p.parse_args()


def simulate(ctrl, start_xy, start_yaw, dt, max_steps):
    """Kinematic simulation of the return controller."""
    xy  = np.array(start_xy, dtype=np.float64)
    yaw = start_yaw
    trajectory = [xy.copy()]
    converged  = False
    steps      = 0

    for k in range(max_steps):
        twist, on_path, nearest = ctrl.compute(xy, yaw, start_from=0)
        vx, _, omega = twist
        # Kinematic update
        xy  = xy + dt * np.array([vx * math.cos(yaw), vx * math.sin(yaw)])
        yaw = _wrap_pi(yaw + dt * omega)
        trajectory.append(xy.copy())
        steps += 1
        if on_path:
            converged = True
            break

    # Compute max lateral overshoot = max min-dist to traj after convergence
    traj_pos = ctrl.traj_pos
    dists    = [float(np.min(np.linalg.norm(traj_pos - p, axis=1)))
                for p in trajectory]
    return converged, steps, float(max(dists)), np.array(trajectory)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npz      = np.load(args.trajectory)
    traj_pos = npz["positions"].astype(np.float64)   # (M+1, 2)

    cfg = ReturnToPathCfg(
        v_max=args.v_max, omega_max=args.omega_max,
        lookahead=args.lookahead,
        return_radius=args.return_radius, heading_tol=args.heading_tol,
    )
    ctrl = ReturnToPathController(cfg, traj_pos)

    # Test offsets: lateral {0.3, 0.5, 1.0} m × heading {0, π/4, π/2} rad
    lateral_offsets = [0.3, 0.5, 1.0]
    heading_offsets = [0.0, math.pi / 4, math.pi / 2]
    spawn_configs   = list(itertools.product(lateral_offsets, heading_offsets))

    # Spawn mid-way along path
    mid = len(traj_pos) // 2
    base_xy  = traj_pos[mid]
    dx = traj_pos[mid+1][0] - traj_pos[mid][0]
    dy = traj_pos[mid+1][1] - traj_pos[mid][1]
    path_yaw = math.atan2(dy, dx)
    # Perpendicular direction (90° left)
    perp = np.array([-math.sin(path_yaw), math.cos(path_yaw)])

    results = []
    for lat, head in spawn_configs:
        start_xy  = base_xy + lat * perp
        start_yaw = _wrap_pi(path_yaw + head)
        converged, steps, overshoot, _ = simulate(
            ctrl, start_xy, start_yaw, args.dt, args.max_steps
        )
        results.append({
            "lateral_m":  lat,
            "heading_rad": head,
            "heading_deg": math.degrees(head),
            "converged":  converged,
            "steps":      steps,
            "time_s":     steps * args.dt,
            "max_overshoot_m": overshoot,
        })
        print(f"  lat={lat:.1f}m  head={math.degrees(head):.0f}°  "
              f"→ {'OK' if converged else 'FAIL'} in {steps} steps")

    n_ok    = sum(r["converged"] for r in results)
    rate    = n_ok / len(results)
    mean_t  = float(np.mean([r["time_s"] for r in results if r["converged"]] or [0]))
    max_ov  = float(np.max([r["max_overshoot_m"] for r in results]))

    per_config_rows = "".join(
        f"| {r['lateral_m']} | {r['heading_deg']:.0f} | {r['converged']} | "
        f"{r['steps']} | {r['time_s']:.2f} | {r['max_overshoot_m']:.3f} |\n"
        for r in results
    )
    verdict = "PASS" if rate >= 0.9 else "FAIL"

    md = f"""# Return Controller Evaluation Results

## Setup
- Trajectory: `{args.trajectory}`  ({len(traj_pos)} waypoints)
- Spawn offsets: lateral {{0.3, 0.5, 1.0}} m x heading {{0, 45, 90}} deg
- return_radius={args.return_radius} m  |  heading_tol={args.heading_tol} rad
- Max steps: {args.max_steps}  (dt={args.dt} s)

## Results
| Metric | Value |
|---|---|
| Success rate | {n_ok}/{len(results)} = **{rate:.2f}** |
| Mean convergence time (successes) | {mean_t:.2f} s |
| Max overshoot | {max_ov:.3f} m |

## Per-configuration
| lateral [m] | heading [deg] | success | steps | time [s] | max_overshoot [m] |
|---|---|---|---|---|---|
{per_config_rows}
## Milestone 5 criterion
Success rate >= 0.9 -> **{verdict} ({rate:.2f})**
"""
    with open(os.path.join(args.out_dir, "RESULTS.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"[eval_return] Results -> {args.out_dir}/RESULTS.md")

if __name__ == "__main__":
    main()
