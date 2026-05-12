# ARP-FP: Event-Triggered Sparse VLA Replanning for Quadruped Navigation

Hybrid locomotion + obstacle-avoidance pipeline for the Unitree Go2 in Isaac Lab. A low-level locomotion policy tracks a reference trajectory; a learned visual trigger detects obstacles and invokes NaVILA (a vision-language model) to compute avoidance maneuvers.

<p align="center">
<img src="./src/go2_teaser.gif" alt="Go2 demo" width="45%">
</p>

---

## Installation

### Environment 1 — Isaac Lab (locomotion + evaluation)

```bash
conda create -n isaaclab python=3.10 && conda activate isaaclab

pip install isaacsim-rl==4.1.0 isaacsim-replicator==4.1.0 \
    isaacsim-extscache-physics==4.1.0 isaacsim-extscache-kit-sdk==4.1.0 \
    isaacsim-extscache-kit==4.1.0 isaacsim-app==4.1.0 \
    --extra-index-url https://pypi.nvidia.com

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install imageio imageio-ffmpeg torchvision scipy matplotlib pandas

git clone git@github.com:yang-zj1026/IsaacLab.git
cd IsaacLab/source/extensions
ln -s {THIS_REPO}/legged-loco/isaaclab_exts/omni.isaac.leggedloco .
cd ../..
./isaaclab.sh -i none
./isaaclab.sh -p -m pip install -e {THIS_REPO}/legged-loco/rsl_rl
```

### Environment 2 — NaVILA (VLA inference, separate terminal)

Follow `../NaVILA/README.md`. Model checkpoint: `../NaVILA/checkpoints/navila-llama3-8b-8f`.

---

## Scripts — What to Run and Why

### `scripts/train.py` — Train the locomotion policy

Trains a PPO policy on the Go2 in Isaac Lab. Skip this if using the provided checkpoint.

```bash
conda activate isaaclab
python scripts/train.py \
    --task=go2_base --history_len=9 \
    --run_name=go2_v1 --max_iterations=2000 \
    --save_interval=200 --headless
# Output: logs/rsl_rl/go2_base/go2_v1/model_1999.pt
```

Pre-trained checkpoint: `logs/rsl_rl/go2_base/001/model_1999.pt`

---

### `scripts/navila_server.py` — Start the NaVILA inference server

Run this **first, in a separate terminal**, before any evaluation that uses the hybrid or threshold method. It loads NaVILA into GPU memory and listens for requests over TCP.

```bash
conda activate navila
PYTHONPATH=$PYTHONPATH:../NaVILA \
python scripts/navila_server.py \
    --model_path ../NaVILA/checkpoints/navila-llama3-8b-8f \
    --port 15432
# Wait for "READY" before running eval
```

Not needed for `--method no_vla`.

---

### `scripts/collect_obstacle_data.py` — Collect trigger training data

Rolls out the locomotion policy with a randomly placed obstacle, saving RGB-D images and scalar features with ground-truth labels to HDF5. Skip this — the dataset is already at `data/obstacle_dataset.h5`.

```bash
conda activate isaaclab
python scripts/collect_obstacle_data.py \
    --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
    --trajectory traj.npz \
    --history_length 9 --headless --enable_cameras
# Output: data/obstacle_dataset.h5
```

---

### `scripts/run_eval_baselines.py` — Run the Section V evaluation

The main evaluation script. Runs N episodes with the obstacle at different positions along the trajectory and records per-episode metrics. Run once per method.

**Start the NaVILA server first (see above) for hybrid and threshold.**

```bash
conda activate isaaclab

# Proposed method: learned visual trigger + NaVILA avoidance
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_baselines.py \
    --method hybrid \
    --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
    --trajectory traj.npz \
    --visual_trigger_checkpoint ../checkpoints/trigger_real_visual.pt \
    --model_path ../NaVILA/checkpoints/navila-llama3-8b-8f \
    --n_episodes 15 --history_length 9 \
    --headless --enable_cameras

# Baseline 2: scalar MLP trigger + NaVILA avoidance
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_baselines.py \
    --method threshold \
    --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
    --trajectory traj.npz \
    --scalar_trigger_checkpoint ../checkpoints/trigger_real_scalar.pt \
    --model_path ../NaVILA/checkpoints/navila-llama3-8b-8f \
    --n_episodes 15 --history_length 9 \
    --headless --enable_cameras

# Baseline 4: RRT* oracle path, no VLA (no server needed)
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_baselines.py \
    --method no_vla \
    --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
    --trajectory traj.npz \
    --n_episodes 15 --history_length 9 \
    --headless
```

Add `--video` to save per-episode `.mp4` files.  
Output per run: `logs/eval_baselines/data/<method>/<timestamp>/episodes.csv` + `results.json`

Pre-computed results are already in `logs/eval_baselines/data/`.

---

### `scripts/compute_metrics.py` — Aggregate results and produce paper figures

Reads all run CSVs under `logs/eval_baselines/data/`, combines episodes across runs of the same method, prints a summary table, and saves figures.

```bash
conda activate isaaclab
python scripts/compute_metrics.py \
    --data_dir logs/eval_baselines/data \
    --out results/
```

Output figures in `results/`:

| Figure | What it shows |
|--------|--------------|
| `scatter_rhoJ_nvla.png` | Cost ratio ρ_J vs. VLA calls per episode |
| `spl_bar.png` | Success-weighted path length (SPL) per method |
| `wallclock.png` | Mission time CDF per method |
| `per_episode_rhoJ.png` | Per-episode cost ratio, all methods |
| `per_episode_twall_nvla.png` | Per-episode T_wall and N_vla (two panels) |
| `trajectories.png` | Spatial overlay of reference + robot paths |

Pre-computed figures are in `results/`.

---

### `scripts/plot_eval_comparison.py` — Overlay all three trajectories on one plot

Reads the latest `spatial.csv` per method and draws reference trajectory, robot paths, and obstacle positions on a shared axis.

```bash
conda activate isaaclab
python scripts/plot_eval_comparison.py
# Output: eval_comparison.png
```

---

### `scripts/track_trajectory.py` — Track a trajectory with no obstacles

Runs the locomotion policy on a reference trajectory without any obstacle or VLA. Useful for sanity-checking the locomotion policy in isolation.

```bash
conda activate isaaclab
python scripts/track_trajectory.py \
    --checkpoint logs/rsl_rl/go2_base/001/model_1999.pt \
    --trajectory traj.npz \
    --history_length 9
```

---

### `scripts/eval_trigger.py` — Evaluate trigger checkpoint accuracy offline

Runs the trigger MLP against the collected dataset and prints precision/recall/F1, then saves a PR curve.

```bash
conda activate isaaclab
python scripts/eval_trigger.py \
    --checkpoint ../checkpoints/trigger_real_visual.pt \
    --dataset data/obstacle_dataset.h5
# Output: logs/eval_trigger/
```

---

## Checkpoints

| Path | Description |
|------|-------------|
| `logs/rsl_rl/go2_base/001/model_1999.pt` | Locomotion policy (PPO, 2000 iter) |
| `../checkpoints/trigger_real_visual.pt` | Proposed visual trigger (ResNet-18 + scalars) |
| `../checkpoints/trigger_real_scalar.pt` | Baseline scalar-only trigger MLP |
| `../NaVILA/checkpoints/navila-llama3-8b-8f` | NaVILA VLA model |

---

## Key Results

| Method | SR | SPL | ρ_J | N_vla |
|--------|----|-----|-----|-------|
| No VLA (RRT* oracle) | 100% | 0.994 | 3.76 | 0 |
| Scalar trigger (B2) | 93.3% | 0.675 | 13.52 | 25.1 |
| **Visual trigger (ours)** | **93.3%** | **0.732** | **11.23** | **15.1** |

ρ_J = J_hat / J* (cost overhead vs. oracle). SPL = success × L* / max(L_hat, L*).  
T_wall = sim time + N_vla × 1.5 s/call (GPU).
