"""Unit tests for HybridFSM — no IsaacLab required.

Run with:
    /srv/local/ganeshr3/conda/envs/isaaclab/bin/python -m pytest \
        trigger_training/tests/test_state_machine.py -v
"""

import math
import os
import sys

# ── Path setup: let Python find hybrid_fsm without an IsaacLab install ──────
_MDP_COMMANDS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "legged-loco", "isaaclab_exts",
    "omni.isaac.leggedloco", "omni", "isaac", "leggedloco",
    "leggedloco", "mdp", "commands",
))
sys.path.insert(0, _MDP_COMMANDS)
from hybrid_fsm import HybridFSM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fsm(**kwargs) -> HybridFSM:
    defaults = dict(
        handoff_clearance=1.5,
        max_avoiding_steps=200,
        return_radius=0.3,
        return_heading_tol=0.2,
        trigger_cooldown=5,
    )
    defaults.update(kwargs)
    return HybridFSM(**defaults)


def _step_tracking(fsm, trigger_fired=False):
    return fsm.step(trigger_fired=trigger_fired, navila_done=False,
                    obstacle_behind=False, d_obs=5.0,
                    path_dist=1.0, heading_error=0.0)


def _step_avoiding(fsm, navila_done=False, obstacle_behind=False, d_obs=0.5):
    return fsm.step(trigger_fired=False, navila_done=navila_done,
                    obstacle_behind=obstacle_behind, d_obs=d_obs,
                    path_dist=1.0, heading_error=0.0)


def _step_returning(fsm, path_dist=1.0, heading_error=0.5, trigger_fired=False):
    return fsm.step(trigger_fired=trigger_fired, navila_done=False,
                    obstacle_behind=True, d_obs=3.0,
                    path_dist=path_dist, heading_error=heading_error)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_tracking(self):
        fsm = _make_fsm()
        assert fsm.state == HybridFSM.TRACKING
        assert fsm.state_name == "TRACKING"

    def test_trigger_armed_at_start(self):
        fsm = _make_fsm()
        assert fsm.trigger_armed is True

    def test_no_transition_flags_at_start(self):
        fsm = _make_fsm()
        assert fsm.handoff_this_step is False
        assert fsm.resume_this_step is False


class TestTrackingToAvoiding:
    def test_no_trigger_stays_tracking(self):
        fsm = _make_fsm()
        for _ in range(10):
            state = _step_tracking(fsm, trigger_fired=False)
        assert state == HybridFSM.TRACKING

    def test_trigger_fires_transition(self):
        fsm = _make_fsm()
        state = _step_tracking(fsm, trigger_fired=True)
        assert state == HybridFSM.AVOIDING
        assert fsm.avoiding_steps == 0

    def test_transition_resets_state_step_count(self):
        fsm = _make_fsm()
        for _ in range(5):
            _step_tracking(fsm)
        _step_tracking(fsm, trigger_fired=True)
        # Reset happens inside _step_tracking; then step() increments → 1
        assert fsm.state_step_count == 1

    def test_handoff_flag_not_set_on_tracking_to_avoiding(self):
        fsm = _make_fsm()
        _step_tracking(fsm, trigger_fired=True)
        assert fsm.handoff_this_step is False


class TestAvoidingState:
    def _get_avoiding_fsm(self):
        fsm = _make_fsm()
        _step_tracking(fsm, trigger_fired=True)
        assert fsm.state == HybridFSM.AVOIDING
        return fsm

    def test_avoiding_steps_increments(self):
        fsm = self._get_avoiding_fsm()
        for k in range(5):
            _step_avoiding(fsm)
            assert fsm.avoiding_steps == k + 1

    def test_no_handoff_if_obstacle_ahead(self):
        fsm = self._get_avoiding_fsm()
        state = _step_avoiding(fsm, navila_done=True, obstacle_behind=False, d_obs=2.0)
        assert state == HybridFSM.AVOIDING
        assert fsm.handoff_this_step is False

    def test_no_handoff_if_too_close(self):
        fsm = self._get_avoiding_fsm()
        state = _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=0.5)
        assert state == HybridFSM.AVOIDING
        assert fsm.handoff_this_step is False

    def test_handoff_on_navila_done_obstacle_behind_clear(self):
        fsm = self._get_avoiding_fsm()
        state = _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        assert state == HybridFSM.RETURNING
        assert fsm.handoff_this_step is True

    def test_handoff_on_timeout(self):
        fsm = _make_fsm(max_avoiding_steps=3)
        _step_tracking(fsm, trigger_fired=True)
        for _ in range(2):
            _step_avoiding(fsm, obstacle_behind=True, d_obs=2.0)
        # Step 3 reaches max_avoiding_steps
        state = _step_avoiding(fsm, obstacle_behind=True, d_obs=2.0)
        assert state == HybridFSM.RETURNING
        assert fsm.handoff_this_step is True

    def test_handoff_resets_state_step_count(self):
        fsm = self._get_avoiding_fsm()
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        # Reset happens inside _step_avoiding; then step() increments → 1
        assert fsm.state_step_count == 1


class TestReturningState:
    def _get_returning_fsm(self):
        fsm = _make_fsm(trigger_cooldown=5)
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        assert fsm.state == HybridFSM.RETURNING
        return fsm

    def test_stays_returning_when_far(self):
        fsm = self._get_returning_fsm()
        state = _step_returning(fsm, path_dist=1.0, heading_error=0.5)
        assert state == HybridFSM.RETURNING
        assert fsm.resume_this_step is False

    def test_stays_returning_when_far_regardless_of_heading(self):
        # Heading is no longer checked — only path_dist matters.
        fsm = self._get_returning_fsm()
        state = _step_returning(fsm, path_dist=1.0, heading_error=1.0)
        assert state == HybridFSM.RETURNING

    def test_resume_when_on_path(self):
        fsm = self._get_returning_fsm()
        # Heading is no longer checked — any heading resumes when path_dist < radius.
        state = _step_returning(fsm, path_dist=0.1, heading_error=1.5)
        assert state == HybridFSM.TRACKING
        assert fsm.resume_this_step is True
        assert fsm.handoff_this_step is False

    def test_cooldown_set_after_resume(self):
        fsm = _make_fsm(trigger_cooldown=7)
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        _step_returning(fsm, path_dist=0.1, heading_error=0.05)
        assert fsm.cooldown_remaining == 7

    def test_resume_resets_state_step_count(self):
        fsm = self._get_returning_fsm()
        _step_returning(fsm, path_dist=0.1, heading_error=0.05)
        # Reset happens inside _step_returning; then step() increments → 1
        assert fsm.state_step_count == 1

    def test_trigger_armed_in_returning(self):
        fsm = self._get_returning_fsm()
        assert fsm.state == HybridFSM.RETURNING
        assert fsm.trigger_armed is True

    def test_trigger_in_returning_goes_to_avoiding(self):
        fsm = self._get_returning_fsm()
        state = _step_returning(fsm, path_dist=1.0, trigger_fired=True)
        assert state == HybridFSM.AVOIDING
        assert fsm.handoff_this_step is False
        assert fsm.resume_this_step is False

    def test_trigger_in_returning_resets_avoiding_steps(self):
        fsm = self._get_returning_fsm()
        _step_returning(fsm, path_dist=1.0, trigger_fired=True)
        assert fsm.avoiding_steps == 0

    def test_trigger_in_returning_not_fired_when_on_path(self):
        # on_path takes priority over trigger
        fsm = self._get_returning_fsm()
        state = _step_returning(fsm, path_dist=0.1, trigger_fired=True)
        assert state == HybridFSM.TRACKING

    def test_trigger_not_fired_when_far_and_no_trigger(self):
        fsm = self._get_returning_fsm()
        state = _step_returning(fsm, path_dist=1.0, trigger_fired=False)
        assert state == HybridFSM.RETURNING


class TestCooldownBehavior:
    def test_trigger_suppressed_during_cooldown(self):
        fsm = _make_fsm(trigger_cooldown=3)
        # Full TRACKING→AVOIDING→RETURNING→TRACKING cycle
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        _step_returning(fsm, path_dist=0.1, heading_error=0.05)
        assert fsm.state == HybridFSM.TRACKING
        assert fsm.trigger_armed is False
        # Fire trigger during cooldown — should be ignored
        state = _step_tracking(fsm, trigger_fired=True)
        assert state == HybridFSM.TRACKING  # did not transition

    def test_cooldown_counts_down(self):
        fsm = _make_fsm(trigger_cooldown=3)
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        _step_returning(fsm, path_dist=0.1, heading_error=0.05)
        assert fsm.cooldown_remaining == 3
        _step_tracking(fsm)
        assert fsm.cooldown_remaining == 2
        _step_tracking(fsm)
        assert fsm.cooldown_remaining == 1
        _step_tracking(fsm)
        assert fsm.cooldown_remaining == 0

    def test_trigger_armed_after_cooldown(self):
        fsm = _make_fsm(trigger_cooldown=2)
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        _step_returning(fsm, path_dist=0.1, heading_error=0.05)
        _step_tracking(fsm)
        _step_tracking(fsm)
        assert fsm.trigger_armed is True

    def test_can_fire_again_after_cooldown(self):
        fsm = _make_fsm(trigger_cooldown=2)
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        _step_returning(fsm, path_dist=0.1, heading_error=0.05)
        _step_tracking(fsm)
        _step_tracking(fsm)
        state = _step_tracking(fsm, trigger_fired=True)
        assert state == HybridFSM.AVOIDING


class TestReset:
    def test_reset_clears_state(self):
        fsm = _make_fsm()
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        assert fsm.state == HybridFSM.RETURNING
        fsm.reset()
        assert fsm.state == HybridFSM.TRACKING
        assert fsm.avoiding_steps == 0
        assert fsm.cooldown_remaining == 0
        assert fsm.handoff_this_step is False
        assert fsm.resume_this_step is False

    def test_trigger_armed_after_reset(self):
        fsm = _make_fsm(trigger_cooldown=100)
        _step_tracking(fsm, trigger_fired=True)
        _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.0)
        _step_returning(fsm, path_dist=0.1, heading_error=0.05)
        assert fsm.trigger_armed is False
        fsm.reset()
        assert fsm.trigger_armed is True


class TestFullCycleSequence:
    def test_tracking_avoiding_returning_tracking(self):
        """Drive a complete avoidance cycle and verify state sequence."""
        fsm = _make_fsm(trigger_cooldown=5)

        # TRACKING (4 quiet steps)
        for _ in range(4):
            assert _step_tracking(fsm) == HybridFSM.TRACKING

        # Trigger fires → AVOIDING
        assert _step_tracking(fsm, trigger_fired=True) == HybridFSM.AVOIDING

        # AVOIDING (20 quiet steps)
        for _ in range(20):
            assert _step_avoiding(fsm, obstacle_behind=False, d_obs=0.8) == HybridFSM.AVOIDING

        # Obstacle passes behind, clearance achieved → RETURNING
        assert _step_avoiding(fsm, navila_done=True, obstacle_behind=True, d_obs=2.5) == HybridFSM.RETURNING
        assert fsm.handoff_this_step is True

        # RETURNING (5 steps far)
        for _ in range(5):
            assert _step_returning(fsm, path_dist=0.8, heading_error=0.3) == HybridFSM.RETURNING

        # On path → back to TRACKING
        assert _step_returning(fsm, path_dist=0.1, heading_error=0.05) == HybridFSM.TRACKING
        assert fsm.resume_this_step is True
        assert fsm.cooldown_remaining == 5
