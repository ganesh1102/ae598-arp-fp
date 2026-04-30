"""
Trains the trigger MLP using both scalar features and visual embeddings.
Supports an ablation flag to disable visual features for comparison.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import argparse
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TriggerMLP(nn.Module):
    def __init__(self, scalar_dim=4, visual_dim=512, h=128, use_visual=True):
        super().__init__()
        self.use_visual = use_visual
        in_dim = scalar_dim + (visual_dim if use_visual else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, scalars, visual=None):
        if self.use_visual:
            x = torch.cat([scalars, visual], dim=-1)
        else:
            x = scalars
        return self.net(x).squeeze(-1)


def split_by_episode(dataset, val_frac=0.1, test_frac=0.1, seed=42):
    episode_ids = sorted(set(d['episode_id'] for d in dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)
    n = len(episode_ids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_ids = set(episode_ids[:n_test])
    val_ids = set(episode_ids[n_test:n_test + n_val])
    train_ids = set(episode_ids[n_test + n_val:])
    return train_ids, val_ids, test_ids


def build_arrays(dataset, ep_ids, features=None):
    """
    Returns:
        X_scalar (N, 4)
        y (N,)
        X_visual (N, 512) or None
    """
    rows = [d for d in dataset if d['episode_id'] in ep_ids]
    X_scalar = np.array([
        [d['d_obs'], d['e_track'], d['delta_e'], d['delta_t_vla']]
        for d in rows
    ], dtype=np.float32)
    y = np.array([d['label'] for d in rows], dtype=np.float32)
    if features is not None:
        idxs = np.array([d['image_idx'] for d in rows])
        X_visual = features[idxs]
    else:
        X_visual = None
    return X_scalar, y, X_visual


def evaluate(model, X_s, X_v, y, criterion, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(X_s, X_v) if X_v is not None else model(X_s)
        loss = criterion(logits, y).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(int)
        y_np = y.cpu().numpy().astype(int)
    metrics = {
        'loss': loss,
        'precision': precision_score(y_np, preds, zero_division=0),
        'recall': recall_score(y_np, preds, zero_division=0),
        'f1': f1_score(y_np, preds, zero_division=0),
        'auc': roc_auc_score(y_np, probs) if len(np.unique(y_np)) > 1 else float('nan'),
    }
    return metrics


def train(args):
    # Load data
    if args.dataset.endswith('.h5') or args.dataset.endswith('.hdf5'):
        from data_loader import load_h5_dataset
        dataset, _, _ = load_h5_dataset(args.dataset)
    else:
        import pickle
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
    print(f"Loaded {len(dataset)} samples from {args.dataset}")

    features = None
    if args.use_visual:
        features = np.load(args.features)
        print(f"Loaded visual features shape={features.shape}")

    train_ids, val_ids, test_ids = split_by_episode(dataset)
    print(f"Episodes: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    X_train_s, y_train, X_train_v = build_arrays(dataset, train_ids, features)
    X_val_s, y_val, X_val_v = build_arrays(dataset, val_ids, features)
    X_test_s, y_test, X_test_v = build_arrays(dataset, test_ids, features)

    print(f"Samples: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test")
    print(f"Positive rates: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

    # Normalize scalar features
    scalar_mean = X_train_s.mean(axis=0)
    scalar_std = X_train_s.std(axis=0) + 1e-6

    def to_tensor(X_s, y, X_v):
        s = torch.tensor((X_s - scalar_mean) / scalar_std, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        v = torch.tensor(X_v, dtype=torch.float32) if X_v is not None else None
        return s, yt, v

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    Xt_s, yt, Xt_v = to_tensor(X_train_s, y_train, X_train_v)
    Xv_s, yv, Xv_v = to_tensor(X_val_s, y_val, X_val_v)
    Xte_s, yte, Xte_v = to_tensor(X_test_s, y_test, X_test_v)

    # Move to device
    Xt_s, yt = Xt_s.to(device), yt.to(device)
    Xv_s, yv = Xv_s.to(device), yv.to(device)
    Xte_s, yte = Xte_s.to(device), yte.to(device)
    if Xt_v is not None:
        Xt_v, Xv_v, Xte_v = Xt_v.to(device), Xv_v.to(device), Xte_v.to(device)

    # Model
    model = TriggerMLP(use_visual=args.use_visual).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)]
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_val_loss = float('inf')
    n_train = len(Xt_s)

    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch_size):
            b = idx[i:i + args.batch_size]
            X_s_b = Xt_s[b]
            y_b = yt[b]
            X_v_b = Xt_v[b] if Xt_v is not None else None

            logits = model(X_s_b, X_v_b) if X_v_b is not None else model(X_s_b)
            loss = criterion(logits, y_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        val_metrics = evaluate(model, Xv_s, Xv_v, yv, criterion)

        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"P={val_metrics['precision']:.3f} "
              f"R={val_metrics['recall']:.3f} "
              f"F1={val_metrics['f1']:.3f} "
              f"AUC={val_metrics['auc']:.3f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scalar_mean': scalar_mean,
                'scalar_std': scalar_std,
                'use_visual': args.use_visual,
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
            }, args.out)

    # Final test evaluation with best checkpoint
    ckpt = torch.load(args.out, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, Xte_s, Xte_v, yte, criterion)
    print(f"\n=== Final TEST metrics ===")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Precision: {test_metrics['precision']:.3f}")
    print(f"Recall: {test_metrics['recall']:.3f}")
    print(f"F1: {test_metrics['f1']:.3f}")
    print(f"AUC: {test_metrics['auc']:.3f}")
    print(f"\nCheckpoint saved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/synthetic_dataset.pkl')
    parser.add_argument('--features', default='data/synthetic_features.npy')
    parser.add_argument('--out', default='checkpoints/trigger_visual.pt')
    parser.add_argument('--use_visual', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)