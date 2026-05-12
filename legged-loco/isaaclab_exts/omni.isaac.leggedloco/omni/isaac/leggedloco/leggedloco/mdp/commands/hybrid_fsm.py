"""Pure-Python state machine for HybridCommandGenerator.

Import-clean of IsaacLab and PyTorch so it can be unit-tested without
the simulator.  HybridCommandGenerator delegates transition logic here.

States
------
  TRACKING  (0) — TrajectoryCommandGenerator drives the robot; trigger active.
  AVOIDING  (1) — NavilaCommandGenerator drives the robot; trigger suppressed.
  RETURNING (2) — ReturnToPathController drives the robot back to the path.

Transitions
-----------
  TRACKING  → AVOIDING  : trigger fired (and cooldown expired)
  AVOIDING  → RETURNING : obstacle behind robot AND d_obs > handoff_clearance
                          AND (NaVILA STOP emitted OR avoiding_steps >= max_avoiding_steps)
  RETURNING → TRACKING  : path_dist < return_radius
  RETURNING → AVOIDING  : trigger fires while still stuck near obstacle
                          (allows re-attempt without going through TRACKING)
"""

from __future__ import annotations


class HybridFSM:
    """Finite state machine for one environment instance (not batched)."""

    TRACKING  = 0
    AVOIDING  = 1
    RETURNING = 2

    _STATE_NAMES = {0: "TRACKING", 1: "AVOIDING", 2: "RETURNING"}

    def __init__(
        self,
        handoff_clearance: float = 1.5,
        max_avoiding_steps: int  = 200,
        return_radius: float     = 0.3,
        return_heading_tol: float = 0.2,
        trigger_cooldown: int    = 100,
    ):
        self.handoff_clearance  = handoff_clearance
        self.max_avoiding_steps = max_avoiding_steps
        self.return_radius      = return_radius
        self.return_heading_tol = return_heading_tol
        self.trigger_cooldown   = trigger_cooldown

        self.state             = self.TRACKING
        self.avoiding_steps    = 0
        self.state_step_count  = 0
        self.cooldown_remaining = 0

        # Transition flags (set for exactly one step when a transition occurs)
        self.handoff_this_step  = False
        self.resume_this_step   = False

        # Set by HybridCommandGenerator on near-side timeout handoff so that
        # the subsequent RETURNING→TRACKING transition skips the cooldown.
        self.skip_next_cooldown = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        trigger_fired: bool,
        navila_done: bool,
        obstacle_behind: bool,
        d_obs: float,
        path_dist: float,
        heading_error: float,
    ) -> int:
        """Advance the FSM by one step.  Returns the (possibly updated) state."""
        self.handoff_this_step = False
        self.resume_this_step  = False

        if self.state == self.TRACKING:
            self._step_tracking(trigger_fired)

        elif self.state == self.AVOIDING:
            self._step_avoiding(navila_done, obstacle_behind, d_obs)

        elif self.state == self.RETURNING:
            self._step_returning(path_dist, heading_error, trigger_fired)

        self.state_step_count += 1
        return self.state

    def reset(self):
        self.state             = self.TRACKING
        self.avoiding_steps    = 0
        self.state_step_count  = 0
        self.cooldown_remaining = 0
        self.handoff_this_step  = False
        self.resume_this_step   = False
        self.skip_next_cooldown = False

    @property
    def state_name(self) -> str:
        return self._STATE_NAMES[self.state]

    @property
    def trigger_armed(self) -> bool:
        """True when the trigger should be checked this step.

        Armed in TRACKING (when cooldown has expired) and also in RETURNING
        so that a stuck-near-obstacle robot can re-enter AVOIDING directly.
        """
        in_tracking = self.state == self.TRACKING and self.cooldown_remaining == 0
        in_returning = self.state == self.RETURNING
        return in_tracking or in_returning

    # ------------------------------------------------------------------
    # Private per-state handlers
    # ------------------------------------------------------------------

    def _step_tracking(self, trigger_fired: bool):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return
        if trigger_fired:
            self.state           = self.AVOIDING
            self.avoiding_steps  = 0
            self.state_step_count = 0

    def _step_avoiding(self, navila_done: bool, obstacle_behind: bool, d_obs: float):
        self.avoiding_steps += 1
        cleared   = obstacle_behind and d_obs > self.handoff_clearance
        timed_out = self.avoiding_steps >= self.max_avoiding_steps
        # Timeout is unconditional — don't stay stuck if NaVILA can't clear the obstacle.
        handoff = (cleared and navila_done) or timed_out
        if handoff:
            self.state            = self.RETURNING
            self.state_step_count = 0
            self.handoff_this_step = True

    def _step_returning(self, path_dist: float, heading_error: float,
                        trigger_fired: bool = False):
        # Heading alignment is handled by the tracker; only check path distance.
        on_path = path_dist < self.return_radius
        if on_path:
            self.state             = self.TRACKING
            self.state_step_count  = 0
            if self.skip_next_cooldown:
                self.cooldown_remaining = 0
                self.skip_next_cooldown = False
            else:
                self.cooldown_remaining = self.trigger_cooldown
            self.resume_this_step   = True
        elif trigger_fired:
            # Obstacle still blocking return to path — go directly back to AVOIDING
            # without going through TRACKING so the trigger cooldown doesn't delay
            # another NaVILA attempt.
            self.state            = self.AVOIDING
            self.avoiding_steps   = 0
            self.state_step_count = 0
            self.skip_next_cooldown = False
