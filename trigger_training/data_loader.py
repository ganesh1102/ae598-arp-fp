"""
Loads Ganesh's HDF5 trigger dataset and converts it to the list-of-dicts
format used by the training pipeline.
"""
import h5py
import numpy as np


def load_h5_dataset(h5_path, include_images=False, relabel=None, verbose=True):
    samples = []
    images_out = None

    with h5py.File(h5_path, 'r') as f:
        feats = f['features']
        N = feats['d_obs'].shape[0]

        if verbose:
            print(f"Loading {N} samples from {h5_path}")

        d_obs           = feats['d_obs'][:]
        tracking_error  = feats['tracking_error'][:]
        delta_t_vla     = feats['delta_t_vla'][:]
        episode_id      = feats['episode_id'][:]
        step            = feats['step'][:]
        time            = feats['time'][:]
        label           = feats['label'][:]
        position        = feats['position'][:]
        heading         = feats['heading'][:]
        obstacle_present = feats['obstacle_present'][:]
        obstacle_pos    = feats['obstacle_pos'][:]
        obstacle_size   = feats['obstacle_size'][:]
        episode_outcome = feats['episode_outcome'][:]
        image_idx       = feats['image_idx'][:]
        vx_cmd          = feats['vx_cmd'][:]
        wp_progress     = feats['wp_progress'][:]

        delta_e = np.zeros_like(tracking_error)
        for ep in np.unique(episode_id):
            mask = (episode_id == ep)
            te = tracking_error[mask]
            de = np.diff(te, prepend=te[0])
            delta_e[mask] = de

        attrs = {k: v for k, v in f.attrs.items()}

        outcome_map = {
            0: attrs.get('episode_outcome_0', 'goal_reached'),
            1: attrs.get('episode_outcome_1', 'collision'),
            2: attrs.get('episode_outcome_2', 'timeout'),
        }
        outcome_map = {k: v.decode() if isinstance(v, bytes) else v
                       for k, v in outcome_map.items()}

        if relabel is not None:
            x_min, x_max = relabel
            new_label = ((d_obs >= x_min) & (d_obs <= x_max)).astype(np.int64)
            if verbose:
                old_pos_rate = float(label.mean())
                new_pos_rate = float(new_label.mean())
                print(f"Relabeled with rule [{x_min}, {x_max}]: "
                      f"positive rate {old_pos_rate:.3f} -> {new_pos_rate:.3f}")
            label = new_label

        if include_images:
            if verbose:
                print("Loading images (this may take a moment)...")
            images_out = {
                'rgb':   f['images/rgb'][:],
                'depth': f['images/depth'][:],
            }

    for i in range(N):
        samples.append({
            'episode_id':       int(episode_id[i]),
            'step':             int(step[i]),
            'time':             float(time[i]),
            'd_obs':            float(d_obs[i]),
            'e_track':          float(tracking_error[i]),
            'delta_e':          float(delta_e[i]),
            'delta_t_vla':      float(delta_t_vla[i]),
            'label':            int(label[i]),
            'position':         position[i].copy(),
            'heading':          float(heading[i]),
            'obstacle_present': bool(obstacle_present[i]),
            'obstacle_pos':     obstacle_pos[i].copy() if obstacle_present[i] else None,
            'obstacle_size':    float(obstacle_size[i]) if obstacle_present[i] else None,
            'episode_outcome':  outcome_map[int(episode_outcome[i])],
            'image_idx':        int(image_idx[i]),
            'vx_cmd':           float(vx_cmd[i]),
            'wp_progress':      float(wp_progress[i]),
        })

    if verbose:
        n_episodes = len(set(s['episode_id'] for s in samples))
        pos_rate = float(np.mean([s['label'] for s in samples]))
        print(f"Loaded {len(samples)} samples across {n_episodes} episodes")
        print(f"Positive label rate: {pos_rate:.3f}")

    return samples, attrs, images_out


def quick_inspect(h5_path):
    samples, attrs, _ = load_h5_dataset(h5_path, include_images=False)

    try:
        import pandas as pd
        df = pd.DataFrame(samples)

        print("\n=== Episode summary (first 20) ===")
        ep_summary = df.groupby('episode_id').agg(
            n_steps=('step', 'count'),
            outcome=('episode_outcome', 'first'),
            has_obstacle=('obstacle_present', 'first'),
            n_positive=('label', 'sum'),
        ).reset_index()
        print(ep_summary.head(20).to_string(index=False))
        print(f"\n... ({len(ep_summary)} episodes total)")

        print("\n=== Outcome distribution ===")
        print(df.groupby('episode_id')['episode_outcome'].first().value_counts())

        print("\n=== d_obs distribution by label ===")
        print(df.groupby('label')['d_obs'].describe())

    except ImportError:
        labels = np.array([s['label'] for s in samples])
        d_obs_vals = np.array([s['d_obs'] for s in samples])
        eps = np.array([s['episode_id'] for s in samples])
        print(f"\nUnique episodes: {len(np.unique(eps))}")
        print(f"Positive samples: {labels.sum()} / {len(labels)}")
        print(f"d_obs (label=0): mean={d_obs_vals[labels == 0].mean():.2f}")
        print(f"d_obs (label=1): mean={d_obs_vals[labels == 1].mean():.2f}")

    print("\n=== File attributes ===")
    for k, v in attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        print(f"  {k}: {v}")


if __name__ == '__main__':
    import sys
    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'legged-loco/data/obstacle_dataset.h5'
    quick_inspect(h5_path)
