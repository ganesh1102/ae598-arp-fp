"""Visual obstacle-detection trigger.

Loads checkpoints/trigger_real_visual.pt once and exposes a batched
step() interface.  Intentionally import-clean of IsaacLab so it can
be unit-tested without the simulator.

Model:  ResNet-18 backbone (512-dim) + 4 scalar features
        → TriggerMLP(516 → 128 → 128 → 1) → sigmoid score

Scalar features (order must match training):
    [d_obs, e_track, delta_e, delta_t_vla]

Image preprocessing (must match extract_features_h5.py exactly):
    ToTensor() → Resize((224,224), antialias=True) → ImageNet Normalize
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VisualTriggerCfg:
    checkpoint_path: str = "checkpoints/trigger_real_visual.pt"
    threshold: float = 0.5
    inference_every: int = 5   # run trigger every N control steps
    device: str = "cuda"


# ---------------------------------------------------------------------------
# MLP — must match train_trigger_visual.py architecture exactly
# ---------------------------------------------------------------------------

class _TriggerMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Public trigger class
# ---------------------------------------------------------------------------

class VisualTrigger:
    """Batched visual obstacle-detection trigger.

    Parameters
    ----------
    cfg        : VisualTriggerCfg
    num_envs   : number of parallel environments
    """

    def __init__(self, cfg: VisualTriggerCfg, num_envs: int = 1):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(
            cfg.device if (cfg.device == "cpu" or not torch.cuda.is_available()) else cfg.device
        )

        # ── Load checkpoint ────────────────────────────────────────────────
        # Compatibility: checkpoints saved with NumPy 2.x reference numpy._core,
        # which doesn't exist in NumPy 1.x environments.
        import sys as _sys, numpy as _np
        if not hasattr(_np, '_core'):
            import numpy.core as _np_core
            _sys.modules.setdefault('numpy._core', _np_core)
            _sys.modules.setdefault('numpy._core.multiarray', _np_core.multiarray)
        ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
        self.use_visual: bool = bool(ckpt["use_visual"])
        self._scalar_mean = torch.tensor(
            ckpt["scalar_mean"], dtype=torch.float32, device=self.device
        )
        self._scalar_std = torch.tensor(
            ckpt["scalar_std"], dtype=torch.float32, device=self.device
        )

        # ── MLP head ───────────────────────────────────────────────────────
        input_dim = 4 + (512 if self.use_visual else 0)
        self._mlp = _TriggerMLP(input_dim)
        self._mlp.load_state_dict(ckpt["model_state_dict"])
        self._mlp.to(self.device).eval()

        # ── ResNet-18 backbone ─────────────────────────────────────────────
        if self.use_visual:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Identity()
            self._backbone = backbone.to(self.device).eval()
            # Preprocessing matches extract_features_h5.py exactly
            self._transform = transforms.Compose([
                transforms.ToTensor(),                          # uint8 HWC → float CHW [0,1]
                transforms.Resize((224, 224), antialias=True),  # direct resize, no crop
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        # ── Per-env mutable state ──────────────────────────────────────────
        self._prev_e_track = np.zeros(num_envs, dtype=np.float32)
        self._fired = torch.zeros(num_envs, dtype=torch.bool)
        self._score = torch.zeros(num_envs, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset per-env state (call at episode start)."""
        if env_ids is None:
            self._prev_e_track[:] = 0.0
            self._fired[:] = False
            self._score[:] = 0.0
        else:
            for i in env_ids:
                self._prev_e_track[i] = 0.0
                self._fired[i] = False
                self._score[i] = 0.0

    @torch.no_grad()
    def step(
        self,
        rgb: np.ndarray,          # (num_envs, H, W, 3)  uint8
        d_obs: np.ndarray,        # (num_envs,)           float32
        e_track: np.ndarray,      # (num_envs,)           float32
        t_since_vla: np.ndarray,  # (num_envs,)           float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one inference step.

        Returns
        -------
        fired : BoolTensor  (num_envs,)
        score : FloatTensor (num_envs,)
        """
        delta_e = e_track - self._prev_e_track
        self._prev_e_track = e_track.copy()

        scalars = np.stack([d_obs, e_track, delta_e, t_since_vla], axis=1)
        scalars_t = torch.tensor(scalars, dtype=torch.float32, device=self.device)
        scalars_norm = (scalars_t - self._scalar_mean) / (self._scalar_std + 1e-6)

        if self.use_visual:
            # Manual preprocessing avoids torchvision's torch.from_numpy numpy-version check.
            # Equivalent to: Resize(224) → ToTensor → ImageNet normalize.
            _mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            _std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            batch = []
            for img in rgb:
                # img: (H, W, 3) uint8; torch.tensor() copies without from_numpy
                t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                t = torch.nn.functional.interpolate(
                    t.to(self.device), size=(224, 224), mode="bilinear", align_corners=False
                )
                batch.append(t.squeeze(0))
            imgs = (torch.stack(batch) - _mean) / _std   # (E, 3, 224, 224)
            vis_feats = self._backbone(imgs)   # (E, 512)
            x = torch.cat([scalars_norm, vis_feats], dim=1)
        else:
            x = scalars_norm

        logits = self._mlp(x)
        scores = torch.sigmoid(logits)
        fired = scores >= self.cfg.threshold

        self._fired = fired.cpu()
        self._score = scores.cpu()
        return self._fired, self._score

    @torch.no_grad()
    def predict_from_features(
        self,
        visual_feat: torch.Tensor,  # (num_envs, 512) pre-extracted ResNet features
        d_obs: float | np.ndarray,
        e_track: float | np.ndarray,
        delta_e: float | np.ndarray,
        t_since_vla: float | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MLP on pre-extracted features (bypasses ResNet; useful for unit tests)."""
        d_obs     = np.atleast_1d(np.float32(d_obs))
        e_track   = np.atleast_1d(np.float32(e_track))
        delta_e   = np.atleast_1d(np.float32(delta_e))
        t_since_vla = np.atleast_1d(np.float32(t_since_vla))

        scalars = np.stack([d_obs, e_track, delta_e, t_since_vla], axis=1)
        scalars_t = torch.tensor(scalars, dtype=torch.float32, device=self.device)
        scalars_norm = (scalars_t - self._scalar_mean) / (self._scalar_std + 1e-6)

        vis = visual_feat.to(self.device)
        x = torch.cat([scalars_norm, vis], dim=1) if self.use_visual else scalars_norm
        logits = self._mlp(x)
        scores = torch.sigmoid(logits)
        return scores >= self.cfg.threshold, scores

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fired(self) -> torch.Tensor:
        """Last step's fired mask. (num_envs,) BoolTensor."""
        return self._fired

    @property
    def score(self) -> torch.Tensor:
        """Last step's sigmoid scores. (num_envs,) FloatTensor."""
        return self._score
