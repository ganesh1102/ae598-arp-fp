"""Unit test: verify VisualTrigger preprocessing matches training code.

Runs WITHOUT IsaacLab — uses predict_from_features() to bypass ResNet and
check only the MLP + normalization path against a frozen reference score.

Reference: sample 0 from data/trigger_features_real.npy + zero scalars
    → logit ≈ 5.3546, score ≈ 0.9952959418296814
"""

import os
import sys

import numpy as np
import pytest
import torch

# Make the trigger module importable without installing the extension
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRIGGER_SRC = os.path.join(
    _REPO_ROOT,
    "legged-loco",
    "isaaclab_exts",
    "omni.isaac.leggedloco",
    "omni",
    "isaac",
    "leggedloco",
    "leggedloco",
    "mdp",
    "triggers",
)
sys.path.insert(0, _TRIGGER_SRC)
from visual_trigger import VisualTrigger, VisualTriggerCfg  # noqa: E402

_CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "trigger_real_visual.pt")
_FEATURES   = os.path.join(_REPO_ROOT, "data", "trigger_features_real.npy")
_REFERENCE_SCORE = 0.9952959418296814


@pytest.fixture(scope="module")
def trigger():
    pytest.importorskip("torch")  # skip gracefully if torch missing
    if not os.path.exists(_CHECKPOINT):
        pytest.skip(f"Checkpoint not found: {_CHECKPOINT}")
    cfg = VisualTriggerCfg(checkpoint_path=_CHECKPOINT, threshold=0.5, device="cpu")
    return VisualTrigger(cfg, num_envs=1)


@pytest.fixture(scope="module")
def visual_feat():
    if not os.path.exists(_FEATURES):
        pytest.skip(f"Feature file not found: {_FEATURES}")
    feat = np.load(_FEATURES)   # (N, 512)
    return torch.tensor(feat[0:1], dtype=torch.float32)  # (1, 512)


def test_reference_score(trigger, visual_feat):
    """MLP output on sample-0 features + zero scalars must match training reference."""
    fired, score = trigger.predict_from_features(
        visual_feat=visual_feat,
        d_obs=0.0,
        e_track=0.0,
        delta_e=0.0,
        t_since_vla=0.0,
    )
    assert score.shape == (1,), f"Expected shape (1,), got {score.shape}"
    assert abs(float(score[0]) - _REFERENCE_SCORE) < 1e-5, (
        f"Score {float(score[0]):.10f} differs from reference "
        f"{_REFERENCE_SCORE:.10f} by more than 1e-5"
    )


def test_threshold_fires(trigger, visual_feat):
    """With reference score ~0.9953, threshold=0.5 should fire."""
    fired, _ = trigger.predict_from_features(
        visual_feat=visual_feat,
        d_obs=0.0,
        e_track=0.0,
        delta_e=0.0,
        t_since_vla=0.0,
    )
    assert bool(fired[0]), "Expected trigger to fire (score >> 0.5)"


def test_high_threshold_no_fire(trigger, visual_feat):
    """Threshold above reference score must not fire."""
    original = trigger.cfg.threshold
    trigger.cfg.threshold = 0.9999
    try:
        fired, _ = trigger.predict_from_features(
            visual_feat=visual_feat,
            d_obs=0.0,
            e_track=0.0,
            delta_e=0.0,
            t_since_vla=0.0,
        )
        assert not bool(fired[0]), "Expected no fire at threshold=0.9999"
    finally:
        trigger.cfg.threshold = original


def test_checkpoint_metadata():
    """Checkpoint must contain required keys with correct shapes."""
    if not os.path.exists(_CHECKPOINT):
        pytest.skip(f"Checkpoint not found: {_CHECKPOINT}")
    ckpt = torch.load(_CHECKPOINT, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "scalar_mean" in ckpt
    assert "scalar_std" in ckpt
    assert "use_visual" in ckpt
    assert len(ckpt["scalar_mean"]) == 4, "scalar_mean must have 4 elements"
    assert len(ckpt["scalar_std"]) == 4, "scalar_std must have 4 elements"
    assert bool(ckpt["use_visual"]), "Expected use_visual=True for this checkpoint"
    w = ckpt["model_state_dict"]["net.0.weight"]
    assert w.shape == (128, 516), f"First layer weight shape {w.shape} != (128, 516)"
