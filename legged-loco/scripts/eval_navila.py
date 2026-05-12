"""NaVILA standalone evaluation harness — requires IsaacLab + NaVILA server.

Robot spawns already facing the obstacle (3 m ahead on the X-axis).
No trigger, no tracker — NaVILA drives avoidance from start to STOP.

Success criterion: NaVILA emits STOP without colliding with the obstacle.

Usage (from legged-loco/, isaaclab conda env):
    CUDA_VISIBLE_DEVICES=1 \\
    /srv/local/ganeshr3/conda/envs/isaaclab/bin/python scripts/eval_navila.py \\
        --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \\
        --model_path ../NaVILA/checkpoints/navila-llama3-8b-8f \\
        --history_length 9 \\
        --headless \\
        --enable_cameras

Outputs
-------
    legged-loco/logs/eval_navila/
        RESULTS.md          — metrics summary
        latency.json        — per-query round-trip latency (p50/p95/p99)
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
    "Once you have passed the obstacle, stop. "
)

parser = argparse.ArgumentParser(description="NaVILA standalone evaluation.")
parser.add_argument("--checkpoint", required=True,
                    help="Path to πloco checkpoint (.pt) or run directory")
parser.add_argument("--model_path",
                    default="../NaVILA/checkpoints/navila-llama3-8b-8f")
parser.add_argument("--instruction", default=_DEFAULT_INSTRUCTION)
parser.add_argument("--obstacle_dist", type=float, default=3.0,
                    help="Distance [m] to place obstacle ahead of robot spawn")
parser.add_argument("--server_host", default="localhost")
parser.add_argument("--server_port", type=int, default=15432)
parser.add_argument("--history_length", type=int, default=0)
parser.add_argument("--num_seeds", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--collision_dist", type=float, default=0.4,
                    help="Distance [m] at which a collision is declared")
parser.add_argument("--latency_n", type=int, default=20,
                    help="Number of warmup+timed socket queries for latency benchmark")
parser.add_argument("--out_dir", default="logs/eval_navila")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True   # always required for NaVILA

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

import base64
import io
import json
import math
import socket

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
        pts = sorted(f for f in os.listdir(path)
                     if f.startswith("model_") and f.endswith(".pt"))
        if pts:
            return os.path.join(path, pts[-1])
    raise FileNotFoundError(path)


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
    navila_python = "/srv/local/ganeshr3/conda/envs/navila/bin/python"
    server_script = os.path.join(os.path.dirname(__file__), "navila_server.py")
    navila_repo = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "NaVILA")
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = navila_repo + ":" + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [navila_python, server_script,
         "--model_path", model_path,
         "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    print("[eval_navila] Waiting for NaVILA server...")
    for line in proc.stdout:
        print(f"  [navila_server] {line}", end="")
        if "READY" in line:
            print("[eval_navila] NaVILA server ready.")
            return proc
        if proc.poll() is not None:
            raise RuntimeError("navila_server.py exited unexpectedly")
    raise RuntimeError("navila_server.py closed stdout before READY")


def _benchmark_latency(host: str, port: int, n: int = 20, warmup: int = 5) -> dict:
    """Send synthetic single-frame queries to the running server and measure RTT."""
    from PIL import Image as _PIL

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    rbuf = b""

    def _send_query():
        nonlocal rbuf
        buf = io.BytesIO()
        _PIL.new("RGB", (320, 240), (0, 128, 0)).save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        frames = [b64] * 8
        req = json.dumps({"frames": frames, "instruction": "Move forward."}) + "\n"
        sock.sendall(req.encode())
        while b"\n" not in rbuf:
            rbuf += sock.recv(4096)
        line, rbuf = rbuf.split(b"\n", 1)
        return json.loads(line.decode())

    for _ in range(warmup):
        _send_query()

    times_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        _send_query()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    sock.close()
    times_ms = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "p50_ms":  float(np.percentile(times_ms, 50)),
        "p95_ms":  float(np.percentile(times_ms, 95)),
        "p99_ms":  float(np.percentile(times_ms, 99)),
        "n": n,
        "note": "wall-clock socket round-trip, synthetic 8-frame JPEG payload",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)

    navila_proc = _start_navila_server(
        model_path=os.path.abspath(args_cli.model_path),
        host=args_cli.server_host, port=args_cli.server_port,
    )

    try:
        results = _run_episodes(checkpoint_path)
        print("\n[eval_navila] Running latency benchmark...")
        lat = _benchmark_latency(
            args_cli.server_host, args_cli.server_port, n=args_cli.latency_n,
        )
    finally:
        navila_proc.terminate()
        navila_proc.wait()
        print("[eval_navila] NaVILA server terminated.")

    _write_outputs(results, lat)


def _run_episodes(checkpoint_path: str) -> list:
    # ── Build env ──────────────────────────────────────────────────────────
    env_cfg = Go2NavilaEvalCfg()
    env_cfg.scene.num_envs = 1

    nav_cfg = env_cfg.commands.base_velocity
    nav_cfg.server_host = args_cli.server_host
    nav_cfg.server_port = args_cli.server_port
    nav_cfg.instruction = args_cli.instruction

    env_cfg.episode_length_s = (
        args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation + 60.0
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)

    env_raw = env.unwrapped
    sim_dt  = env_raw.cfg.sim.dt * env_raw.cfg.decimation
    device  = env_raw.device

    # ── Load πloco policy ──────────────────────────────────────────────────
    agent_cfg: RslRlOnPolicyRunnerCfg = Go2RoughPPORunnerCfg()
    agent_yaml = os.path.join(
        os.path.dirname(checkpoint_path), "..", "params", "agent.yaml"
    )
    if os.path.exists(agent_yaml):
        update_class_from_dict(agent_cfg, load_yaml(agent_yaml))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    print(f"[eval_navila] Loaded πloco from {checkpoint_path}")

    cmd_term = env_raw.command_manager._terms["base_velocity"]
    results = []

    for seed in range(args_cli.num_seeds):
        print(f"\n[eval_navila] Seed {seed+1}/{args_cli.num_seeds}")
        env.reset()

        # Obstacle: obstacle_dist m ahead of robot spawn on X-axis
        robot_pos = env_raw.scene["robot"].data.root_pos_w[0].cpu().numpy()
        obs_xy = np.array([robot_pos[0] + args_cli.obstacle_dist, robot_pos[1]])
        obs_z  = _OBSTACLE_H / 2.0
        _teleport_obstacle(env_raw, obs_xy, obs_z)

        obs_xy_t = torch.tensor(obs_xy, dtype=torch.float32, device=device)
        obs, _ = env.get_observations()

        step = 0
        terminated_by = "max_steps"
        _final_queries = 0
        min_clearance  = float("inf")

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step += 1

            robot_pos3 = env_raw.scene["robot"].data.root_pos_w[0]
            d_obs = float(torch.norm(robot_pos3[:2] - obs_xy_t))
            min_clearance = min(min_clearance, d_obs)

            _final_queries = cmd_term.navila_queries

            if cmd_term.done:
                terminated_by = "stop"
                break
            if dones[0]:
                terminated_by = "fall"
                break
            if d_obs < args_cli.collision_dist:
                terminated_by = "collision"
                break
            if step >= args_cli.max_steps:
                terminated_by = "max_steps"
                break

        # Clearance at episode end
        robot_pos3 = env_raw.scene["robot"].data.root_pos_w[0]
        d_obs_final = float(torch.norm(robot_pos3[:2] - obs_xy_t))

        success = (terminated_by == "stop")
        results.append({
            "seed":            seed,
            "success":         success,
            "terminated_by":   terminated_by,
            "steps":           step,
            "time_s":          step * sim_dt,
            "navila_queries":  _final_queries,
            "min_clearance_m": min_clearance,
            "final_d_obs_m":   d_obs_final,
        })
        status = "SUCCESS" if success else f"FAIL ({terminated_by})"
        print(f"  → {status}  steps={step}  queries={_final_queries}"
              f"  min_clearance={min_clearance:.2f}m")

    env.close()
    return results


def _write_outputs(results: list, lat: dict):
    n_ok  = sum(r["success"] for r in results)
    rate  = n_ok / len(results)
    mean_q     = float(np.mean([r["navila_queries"] for r in results]))
    mean_t     = float(np.mean([r["time_s"]         for r in results]))
    mean_cl    = float(np.mean([r["min_clearance_m"] for r in results]))

    stop_steps = [r["steps"] for r in results if r["terminated_by"] == "stop"]
    mean_stop_steps = float(np.mean(stop_steps)) if stop_steps else float("nan")

    # ── latency.json ──────────────────────────────────────────────────────
    lat_data = {"gpu_wall_clock": lat}
    with open(os.path.join(args_cli.out_dir, "latency.json"), "w") as f:
        json.dump(lat_data, f, indent=2)

    # ── RESULTS.md ─────────────────────────────────────────────────────────
    per_seed_rows = "".join(
        f"| {r['seed']} | {r['success']} | {r['terminated_by']} | "
        f"{r['steps']} | {r['time_s']:.1f} | {r['navila_queries']} | "
        f"{r['min_clearance_m']:.3f} |\n"
        for r in results
    )
    md = f"""# NaVILA Evaluation Results

## Setup
- Obstacle: {args_cli.obstacle_dist:.1f} m ahead of robot spawn on X-axis
- Seeds: {len(results)}  |  Max steps: {args_cli.max_steps}  |  Collision dist: {args_cli.collision_dist} m
- Instruction: "{args_cli.instruction[:80]}..."

## Results
| Metric | Value |
|---|---|
| Success rate (STOP without collision) | {n_ok}/{len(results)} = **{rate:.2f}** |
| Mean NaVILA queries per episode | {mean_q:.1f} |
| Mean episode time | {mean_t:.1f} s |
| Mean min clearance | {mean_cl:.3f} m |
| Mean steps to STOP (successful) | {mean_stop_steps:.0f} |

## Per-seed
| seed | success | terminated_by | steps | time [s] | queries | min_clearance [m] |
|---|---|---|---|---|---|---|
{per_seed_rows}
## Latency (GPU wall-clock socket round-trip, N={lat['n']})
| Percentile | Value [ms] |
|---|---|
| p50 | {lat['p50_ms']:.1f} |
| p95 | {lat['p95_ms']:.1f} |
| p99 | {lat['p99_ms']:.1f} |

Note: {lat['note']}

## Milestone criterion
Success rate ≥ 0.7 → **{'PASS' if rate >= 0.7 else 'FAIL'} ({rate:.2f})**

Raw latency: `latency.json`
"""
    results_path = os.path.join(args_cli.out_dir, "RESULTS.md")
    with open(results_path, "w") as f:
        f.write(md)
    print(md)
    print(f"[eval_navila] Results → {results_path}")
    print(f"[eval_navila] Latency → {os.path.join(args_cli.out_dir, 'latency.json')}")


if __name__ == "__main__":
    main()
    simulation_app.close()
