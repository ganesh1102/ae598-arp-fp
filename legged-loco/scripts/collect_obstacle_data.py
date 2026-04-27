"""Collect RGB-D + tabular data for training the obstacle-detection MLP trigger.

Architecture
------------
    P* (OCP)  →  geometric controller  →  πloco  →  Go2 sim
                                                         │
                  [obstacle randomly placed on path]     │
                                                         ▼
                                     HDF5 flat-sample dataset

Each saved frame produces one sample dict:

    episode_id        int     which episode
    step              int     control step within episode
    time              float   sim time within episode [s]

    -- MLP input features (z_t) --
    d_obs             float   min depth in forward FOV [m]
    delta_t_vla       float   time since last VLA call [s]  (grows from 0 each episode)
    tracking_error    float   e_t = ||p_t − p*_k|| [m]
    vx_cmd            float   forward velocity command [m/s]
    wp_progress       float   k / M  ∈ [0, 1]

    -- Ground-truth label --
    label             uint8   1 if obstacle present AND dist(robot,obs) < label_dist

    -- Debug / metadata (not used in training) --
    position          (2,)    robot (x, y) [m]
    heading           float   robot yaw [rad]
    obstacle_present  bool
    obstacle_pos      (2,)    obstacle (x, y), NaN if clear episode
    obstacle_size     float   cylinder radius [m], NaN if clear
    episode_outcome   int8    0=goal_reached  1=collision  2=timeout
    image_idx         int     row index into /images/depth and /images/rgb

HDF5 layout
-----------
    /features/<field>   flat arrays, one entry per saved frame (N total)
    /images/depth       (N, H, W)     float16  gzip-4   [metres]
    /images/rgb         (N, H, W, 3)  uint8    gzip-4
    root attrs: trajectory_file, sim_dt, save_every, label_dist, trigger_dist,
                episode_outcome_map

Usage
-----
    python scripts/collect_obstacle_data.py \\
        --checkpoint logs/rsl_rl/go2_base/<run>/model_1999.pt \\
        --trajectory traj.npz \\
        --output data/obstacle_dataset.h5 \\
        --num_episodes 500 \\
        --history_length 9
"""

"""Launch Isaac Sim first."""

import argparse
import os
import sys
import random

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect obstacle-detection data.")
parser.add_argument("--checkpoint", required=True,
                    help="Path to trained πloco checkpoint (.pt) or run directory")
parser.add_argument("--trajectory", required=True,
                    help="Path to P* .npz file")
parser.add_argument("--output", default="data/obstacle_dataset.h5")
parser.add_argument("--num_episodes", type=int, default=500)
parser.add_argument("--history_length", type=int, default=0,
                    help="Must match training value (0 = no history wrapper)")
parser.add_argument("--max_steps", type=int, default=1500,
                    help="Max control steps per episode before timeout")
parser.add_argument("--save_every", type=int, default=5,
                    help="Save one frame per N control steps  (default 5 → 10 Hz at 50 Hz ctrl)")
parser.add_argument("--trigger_dist", type=float, default=2.0,
                    help="Terminate blocked episode when robot is within this distance [m]")
parser.add_argument("--label_dist", type=float, default=3.0,
                    help="Per-frame label = 1 when dist(robot, obstacle) < label_dist [m]")
parser.add_argument("--obstacle_frac", type=float, default=0.5,
                    help="Fraction of episodes that contain an obstacle")
parser.add_argument("--min_wp_frac", type=float, default=0.15,
                    help="Earliest obstacle placement (fraction of M)")
parser.add_argument("--max_wp_frac", type=float, default=0.80,
                    help="Latest obstacle placement (fraction of M)")
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True   # required for RGB-D rendering

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

import h5py
import numpy as np
import torch
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.leggedloco.config.go2.go2_datacollect_cfg import (
    Go2DataCollectCfg,
    OBSTACLE_RADII,
    OBSTACLE_HEIGHT,
    N_OBSTACLES,
)
from omni.isaac.leggedloco.config.go2.go2_low_base_cfg import Go2RoughPPORunnerCfg
from omni.isaac.leggedloco.leggedloco.mdp.commands import TrajectoryCommandGeneratorCfg
from omni.isaac.leggedloco.utils import RslRlVecEnvHistoryWrapper

sys.path.insert(0, os.path.dirname(__file__))
from optimal_trajectory_solver import load_trajectory

# episode_outcome integer codes
OUTCOME_GOAL      = 0
OUTCOME_COLLISION = 1
OUTCOME_TIMEOUT   = 2
OUTCOME_MAP = {OUTCOME_GOAL: "goal_reached",
               OUTCOME_COLLISION: "collision",
               OUTCOME_TIMEOUT: "timeout"}

_HIDDEN_POS  = torch.tensor([0.0, 0.0, -100.0])
_NAN2        = np.array([float("nan"), float("nan")], dtype=np.float32)

IMG_H, IMG_W = 240, 320

# ---------------------------------------------------------------------------
# HDF5 initialisation
# ---------------------------------------------------------------------------

def _init_hdf5(hf: h5py.File):
    """Create extensible (resizable) datasets for flat-sample storage."""
    feat = hf.create_group("features")
    imgs = hf.create_group("images")

    C1 = (1024,)         # chunk for 1-D scalar arrays
    C2 = (1024, 2)       # chunk for 2-D position arrays

    feat.create_dataset("episode_id",       shape=(0,),    maxshape=(None,),    dtype="i4",  chunks=C1)
    feat.create_dataset("step",             shape=(0,),    maxshape=(None,),    dtype="i4",  chunks=C1)
    feat.create_dataset("time",             shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    # MLP input features (z_t)
    feat.create_dataset("d_obs",            shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    feat.create_dataset("delta_t_vla",      shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    feat.create_dataset("tracking_error",   shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    feat.create_dataset("vx_cmd",           shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    feat.create_dataset("wp_progress",      shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    # Ground-truth label
    feat.create_dataset("label",            shape=(0,),    maxshape=(None,),    dtype="u1",  chunks=C1)
    # Debug / metadata
    feat.create_dataset("position",         shape=(0, 2),  maxshape=(None, 2),  dtype="f4",  chunks=C2)
    feat.create_dataset("heading",          shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    feat.create_dataset("obstacle_present", shape=(0,),    maxshape=(None,),    dtype="?",   chunks=C1)
    feat.create_dataset("obstacle_pos",     shape=(0, 2),  maxshape=(None, 2),  dtype="f4",  chunks=C2)
    feat.create_dataset("obstacle_size",    shape=(0,),    maxshape=(None,),    dtype="f4",  chunks=C1)
    feat.create_dataset("episode_outcome",  shape=(0,),    maxshape=(None,),    dtype="i1",  chunks=C1)
    feat.create_dataset("image_idx",        shape=(0,),    maxshape=(None,),    dtype="i4",  chunks=C1)

    KW = dict(compression="gzip", compression_opts=4)
    imgs.create_dataset("depth", shape=(0, IMG_H, IMG_W),
                        maxshape=(None, IMG_H, IMG_W), dtype="f2",
                        chunks=(32, IMG_H, IMG_W), **KW)
    imgs.create_dataset("rgb",   shape=(0, IMG_H, IMG_W, 3),
                        maxshape=(None, IMG_H, IMG_W, 3), dtype="u1",
                        chunks=(32, IMG_H, IMG_W, 3), **KW)
    return feat, imgs


def _extend(ds: h5py.Dataset, data: np.ndarray):
    """Append rows to a resizable HDF5 dataset."""
    n_new = len(data)
    n_old = ds.shape[0]
    ds.resize(n_old + n_new, axis=0)
    ds[n_old:] = data


def _flush_episode(feat: h5py.Group, imgs: h5py.Group,
                   ep_buf: list, outcome: int,
                   depths_buf: list, rgbs_buf: list):
    """Write one episode's buffered samples to HDF5."""
    if not ep_buf:
        return

    n = len(ep_buf)
    img_base = imgs["depth"].shape[0]   # current image count before this episode

    # Update outcome and image_idx (unknown until episode ends)
    for i, s in enumerate(ep_buf):
        s["episode_outcome"] = outcome
        s["image_idx"]       = img_base + i

    # Build column arrays
    cols = {k: np.array([s[k] for s in ep_buf]) for k in ep_buf[0]}

    for key, arr in cols.items():
        _extend(feat[key], arr)

    _extend(imgs["depth"], np.array(depths_buf, dtype=np.float16))
    _extend(imgs["rgb"],   np.array(rgbs_buf,   dtype=np.uint8))


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _compute_d_obs(depth: np.ndarray,
                   fov_w: float = 0.40,   # central 40% of width
                   fov_h: float = 0.50,   # central 50% of height
                   d_max: float = 14.9) -> float:
    """Minimum valid depth reading in the forward central FOV."""
    H, W = depth.shape
    r0, r1 = int(H * (1 - fov_h) / 2), int(H * (1 + fov_h) / 2)
    c0, c1 = int(W * (1 - fov_w) / 2), int(W * (1 + fov_w) / 2)
    roi = depth[r0:r1, c0:c1]
    valid = roi[np.isfinite(roi) & (roi > 0.05) & (roi < d_max)]
    return float(np.min(valid)) if valid.size > 0 else d_max


def _yaw_from_quat(q: torch.Tensor) -> float:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return float(torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# ---------------------------------------------------------------------------
# Obstacle management
# ---------------------------------------------------------------------------

def _teleport_obstacle(env_raw, name: str, xy: np.ndarray):
    obs   = env_raw.scene[f"obstacle_{name}"]
    state = obs.data.default_root_state.clone()   # (1, 13)
    state[0, 0] = float(xy[0])
    state[0, 1] = float(xy[1])
    state[0, 2] = OBSTACLE_HEIGHT / 2.0
    state[0, 3:7] = torch.tensor([1., 0., 0., 0.])
    state[0, 7:]   = 0.0
    obs.write_root_state_to_sim(state)


def _hide(env_raw, name: str):
    obs   = env_raw.scene[f"obstacle_{name}"]
    state = obs.data.default_root_state.clone()
    state[0, :3]  = _HIDDEN_POS
    state[0, 3:7] = torch.tensor([1., 0., 0., 0.])
    state[0, 7:]  = 0.0
    obs.write_root_state_to_sim(state)


def _hide_all(env_raw):
    for i in range(N_OBSTACLES):
        _hide(env_raw, f"{i:02d}")


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(f for f in os.listdir(path)
                     if f.startswith("model_") and f.endswith(".pt"))
        if not pts:
            raise FileNotFoundError(f"No model_*.pt in {path}")
        chosen = os.path.join(path, pts[-1])
        print(f"[INFO] Auto-selected checkpoint: {chosen}")
        return chosen
    raise FileNotFoundError(f"Checkpoint not found: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Print before any Isaac Lab output so it can't be swallowed
    print(f"\n{'='*60}", flush=True)
    print(f"  history_length  = {args_cli.history_length}", flush=True)
    print(f"  expected obs dim= {45 * (1 + args_cli.history_length)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)
    traj = load_trajectory(args_cli.trajectory)
    M    = traj.M
    print(f"[INFO] Loaded P*: {M + 1} waypoints, goal={traj.positions[-1].round(2)}")

    # ── Build env ──────────────────────────────────────────────────────────
    env_cfg = Go2DataCollectCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(
        robot_attr="robot",
        trajectory_file=args_cli.trajectory,
        eps_wp=0.15,
        lookahead_dist=1.0,
        Kp_yaw=2.0,
        v_max=0.5,
        omega_max=1.0,
    )

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
        print(f"[INFO] Using RslRlVecEnvHistoryWrapper  history_length={args_cli.history_length}  "
              f"obs_dim={45 * (1 + args_cli.history_length)}")
    else:
        env = RslRlVecEnvWrapper(env)
        print("[INFO] Using RslRlVecEnvWrapper  obs_dim=45")

    env_raw  = env.unwrapped
    sim_dt   = env_raw.cfg.sim.dt * env_raw.cfg.decimation   # control step [s]
    device   = env_raw.device

    # ── Load policy ────────────────────────────────────────────────────────
    agent_cfg: RslRlOnPolicyRunnerCfg = Go2RoughPPORunnerCfg()
    agent_yaml = os.path.join(os.path.dirname(checkpoint_path), "..", "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        update_class_from_dict(agent_cfg, load_yaml(agent_yaml))
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=device)
    print(f"[INFO] Loaded policy from {checkpoint_path}")

    cmd_term = env_raw.command_manager._terms.get("base_velocity")
    cam      = env_raw.scene["front_camera"]

    traj_pos_np = traj.positions  # (M+1, 2)

    # ── HDF5 setup ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    hf = h5py.File(args_cli.output, "w")
    hf.attrs["trajectory_file"]   = args_cli.trajectory
    hf.attrs["sim_dt"]            = sim_dt
    hf.attrs["save_every"]        = args_cli.save_every
    hf.attrs["label_dist"]        = args_cli.label_dist
    hf.attrs["trigger_dist"]      = args_cli.trigger_dist
    hf.attrs["episode_outcome_0"] = "goal_reached"
    hf.attrs["episode_outcome_1"] = "collision"
    hf.attrs["episode_outcome_2"] = "timeout"
    feat, imgs = _init_hdf5(hf)

    print(f"\n[INFO] Collecting {args_cli.num_episodes} episodes → {args_cli.output}")
    print(f"[INFO] Obstacles: {N_OBSTACLES} cylinders, r=[{OBSTACLE_RADII[0]:.2f}, {OBSTACLE_RADII[-1]:.2f}] m\n")
    header = f"{'ep':>5}/{'tot':<5}  {'%':>5}  {'type':>7}  {'frames':>6}  {'outcome':>13}  {'obs details':>28}  {'total_samples':>13}"
    print(header)
    print("-" * len(header))

    total_samples = 0
    pbar = tqdm(total=args_cli.num_episodes, desc="Collecting", unit="ep",
                dynamic_ncols=True)

    for ep_idx in range(args_cli.num_episodes):
        # ── Decide episode type ────────────────────────────────────────────
        has_obstacle = random.random() < args_cli.obstacle_frac

        obs_idx    = -1
        obs_radius = float("nan")
        obs_xy     = _NAN2.copy()
        obs_wp_idx = -1

        _hide_all(env_raw)

        if has_obstacle:
            obs_idx    = random.randint(0, N_OBSTACLES - 1)
            obs_radius = OBSTACLE_RADII[obs_idx]
            obs_wp_idx = random.randint(int(args_cli.min_wp_frac * M),
                                        int(args_cli.max_wp_frac * M))
            obs_xy     = traj_pos_np[obs_wp_idx].astype(np.float32)
            _teleport_obstacle(env_raw, f"{obs_idx:02d}", obs_xy)

        obs_xy_torch = torch.tensor(obs_xy, dtype=torch.float32, device=device)

        if has_obstacle:
            tqdm.write(f"\n── ep {ep_idx+1:>4}  OBSTACLE  idx={obs_idx:02d}  "
                       f"r={obs_radius:.2f}m  wp={obs_wp_idx}  "
                       f"pos=({obs_xy[0]:.2f}, {obs_xy[1]:.2f})")
        else:
            tqdm.write(f"\n── ep {ep_idx+1:>4}  clear")

        # ── Reset robot (obstacle already placed) ──────────────────────────
        # env.reset() does not go through the history wrapper's obs building;
        # call get_observations() afterwards to get the full history-augmented obs.
        env.reset()
        obs, _ = env.get_observations()

        # ── Per-episode buffers ────────────────────────────────────────────
        ep_buf    = []   # list of sample dicts (image_idx/outcome filled later)
        depth_buf = []
        rgb_buf   = []

        step    = 0
        outcome = OUTCOME_TIMEOUT

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)   # outside inference_mode so env tensors stay mutable
            step += 1

            # ── Robot state ───────────────────────────────────────────────
            robot_pos_w  = env_raw.scene["robot"].data.root_pos_w[0]
            robot_xy     = robot_pos_w[:2]
            robot_quat   = env_raw.scene["robot"].data.root_quat_w[0]
            heading      = _yaw_from_quat(robot_quat)
            e_t          = float(cmd_term.tracking_error()[0])
            wp_idx       = int(cmd_term.active_waypoint_idx[0])
            vx_cmd       = float(cmd_term.command[0, 0])
            t_ep         = step * sim_dt

            # ── Proximity / label ─────────────────────────────────────────
            if has_obstacle:
                dist_to_obs = float(torch.norm(robot_xy - obs_xy_torch))
                frame_label = int(dist_to_obs < args_cli.label_dist)
            else:
                dist_to_obs = float("inf")
                frame_label = 0

            # ── Save frame ────────────────────────────────────────────────
            if step % args_cli.save_every == 0:
                rgb_raw   = cam.data.output["rgb"][0].cpu().numpy()
                depth_raw = cam.data.output["distance_to_image_plane"][0].cpu().numpy()

                # Squeeze to (H, W) regardless of trailing singleton dims
                depth_raw = depth_raw.squeeze()
                if depth_raw.ndim == 1:
                    depth_raw = depth_raw.reshape(IMG_H, IMG_W)

                # Squeeze rgb to (H, W, C) and drop alpha if present
                rgb_raw = rgb_raw.squeeze()
                if rgb_raw.ndim == 2:
                    rgb_raw = np.stack([rgb_raw]*3, axis=-1)   # greyscale → rgb
                rgb_raw = rgb_raw[..., :3]

                d_obs = _compute_d_obs(depth_raw)

                sample = {
                    "episode_id":       np.int32(ep_idx),
                    "step":             np.int32(step),
                    "time":             np.float32(t_ep),
                    # MLP input features
                    "d_obs":            np.float32(d_obs),
                    "delta_t_vla":      np.float32(t_ep),    # grows from 0 each episode
                    "tracking_error":   np.float32(e_t),
                    "vx_cmd":           np.float32(vx_cmd),
                    "wp_progress":      np.float32(wp_idx / max(M, 1)),
                    # Label
                    "label":            np.uint8(frame_label),
                    # Debug
                    "position":         robot_xy.cpu().numpy().astype(np.float32),
                    "heading":          np.float32(heading),
                    "obstacle_present": np.bool_(has_obstacle),
                    "obstacle_pos":     obs_xy.copy(),
                    "obstacle_size":    np.float32(obs_radius),
                    # Filled after episode ends:
                    "episode_outcome":  np.int8(OUTCOME_TIMEOUT),
                    "image_idx":        np.int32(-1),
                }

                ep_buf.append(sample)
                depth_buf.append(depth_raw.astype(np.float16))
                rgb_buf.append(rgb_raw[..., :3])   # drop alpha

            # ── Live postfix update every 10 steps ───────────────────────
            if step % 10 == 0:
                rx, ry = float(robot_xy[0]), float(robot_xy[1])
                postfix = dict(
                    ep=f"{ep_idx+1}/{args_cli.num_episodes}",
                    step=f"{step}/{args_cli.max_steps}",
                    robot=f"({rx:.2f},{ry:.2f})",
                    wp=f"{wp_idx}/{M}",
                    e_t=f"{e_t:.2f}m",
                )
                if has_obstacle:
                    postfix["obs_xy"] = f"({obs_xy[0]:.2f},{obs_xy[1]:.2f})"
                    postfix["d_obs"]  = f"{dist_to_obs:.2f}m"
                    postfix["r"]      = f"{obs_radius:.2f}"
                else:
                    postfix["type"] = "clear"
                pbar.set_postfix(postfix, refresh=True)

            # ── Termination ───────────────────────────────────────────────
            if has_obstacle and dist_to_obs < args_cli.trigger_dist:
                outcome = OUTCOME_COLLISION
                break
            if cmd_term.goal_reached.all():
                outcome = OUTCOME_GOAL
                break
            if step >= args_cli.max_steps:
                outcome = OUTCOME_TIMEOUT
                break

        # ── Flush episode to HDF5 ──────────────────────────────────────────
        _flush_episode(feat, imgs, ep_buf, outcome, depth_buf, rgb_buf)
        hf.flush()

        n_frames = len(ep_buf)
        total_samples += n_frames
        pct      = 100.0 * (ep_idx + 1) / args_cli.num_episodes
        type_str = f"obs:{obs_idx:02d}" if has_obstacle else "clear"
        obs_detail = (f"r={obs_radius:.2f} wp={obs_wp_idx} "
                      f"({obs_xy[0]:.1f},{obs_xy[1]:.1f})") if has_obstacle else "-"
        pbar.set_postfix(type=type_str, outcome=OUTCOME_MAP[outcome],
                         frames=n_frames, total_samples=total_samples, refresh=False)
        pbar.update(1)
        tqdm.write(f"{ep_idx+1:>5}/{args_cli.num_episodes:<5}  {pct:>4.1f}%  {type_str:>7}  "
                   f"{n_frames:>6}  {OUTCOME_MAP[outcome]:>13}  {obs_detail:>28}  {total_samples:>13}")

    pbar.close()
    hf.close()
    print(f"\n[INFO] Done. {total_samples} total samples saved to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
