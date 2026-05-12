"""Return-to-path pure-pursuit controller.

Given the reference trajectory (world-frame XY waypoints) and the current
robot pose, emits a (v_x, v_y, ω_z) command that steers the robot back
onto the path.

Algorithm
---------
1. Find nearest waypoint index k* = argmin ‖p_traj[k] − p_robot‖.
2. Scan forward from k* to find the furthest waypoint within ``lookahead``
   metres (lookahead point p_L).
3. Compute lateral / heading error to p_L.
4. Emit:
     v_x   = v_max * cos(heading_err)   (reduces speed during sharp turns)
     v_y   = 0
     ω_z   = clip(Kp_angular * heading_err, -omega_max, +omega_max)
5. "On path" when dist(robot, traj[k*]) < return_radius AND
   |heading_err_to_tangent| < heading_tol.

Import-clean of IsaacLab — testable as plain Python / NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReturnToPathCfg:
    """Configuration for the return-to-path controller."""
    v_max: float          = 0.4
    """Maximum forward velocity [m/s]."""
    omega_max: float      = 1.0
    """Maximum yaw rate [rad/s]."""
    Kp_angular: float     = 2.0
    """Proportional gain: heading error → ω_z."""
    lookahead: float      = 0.5
    """Pure-pursuit lookahead distance [m]."""
    return_radius: float  = 0.3
    """Lateral distance threshold for "on path" [m]."""
    heading_tol: float    = 0.2
    """Heading-to-tangent error threshold for "on path" [rad]."""


class ReturnToPathController:
    """Pure-pursuit return-to-path controller.

    Parameters
    ----------
    cfg        : ReturnToPathCfg
    traj_pos   : np.ndarray, shape (M+1, 2), world-frame XY waypoints
    """

    def __init__(self, cfg: ReturnToPathCfg, traj_pos: np.ndarray):
        self.cfg      = cfg
        self.traj_pos = np.array(traj_pos, dtype=np.float64)  # (M+1, 2)
        self._M       = len(self.traj_pos) - 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        robot_xy: np.ndarray,   # (2,) world-frame XY
        robot_yaw: float,       # radians
        start_from: int = 0,    # search from this waypoint onward
    ) -> tuple[np.ndarray, bool, int]:
        """Return (twist_3, on_path, nearest_idx).

        twist_3 : np.ndarray (3,) → [v_x, v_y=0, ω_z]
        on_path : bool — True when within return_radius and heading_tol
        nearest_idx : int — index of nearest waypoint
        """
        nearest_idx = self._nearest_from(robot_xy, start_from)
        lookahead_pt = self._lookahead_point(robot_xy, nearest_idx)

        # Heading error to lookahead
        dx = lookahead_pt[0] - robot_xy[0]
        dy = lookahead_pt[1] - robot_xy[1]
        desired_yaw = math.atan2(dy, dx)
        heading_err = _wrap_pi(desired_yaw - robot_yaw)

        v_x   = self.cfg.v_max * math.cos(heading_err)
        v_x   = max(0.0, v_x)   # don't reverse
        omega  = float(np.clip(self.cfg.Kp_angular * heading_err,
                               -self.cfg.omega_max, self.cfg.omega_max))

        twist = np.array([v_x, 0.0, omega], dtype=np.float32)

        # "On path" check: lateral dist to nearest + heading error to tangent
        lateral_dist = float(np.linalg.norm(robot_xy - self.traj_pos[nearest_idx]))
        tangent_err  = self._heading_to_tangent(robot_yaw, nearest_idx)
        on_path      = (lateral_dist < self.cfg.return_radius
                        and abs(tangent_err) < self.cfg.heading_tol)

        return twist, on_path, nearest_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_from(self, robot_xy: np.ndarray, start: int) -> int:
        """Nearest waypoint index from ``start`` forward."""
        dists = np.linalg.norm(self.traj_pos[start:] - robot_xy, axis=1)
        return int(start + np.argmin(dists))

    def _lookahead_point(self, robot_xy: np.ndarray, k: int) -> np.ndarray:
        """Furthest waypoint within lookahead_dist of robot_xy, starting at k."""
        best = self.traj_pos[k]
        for i in range(k, self._M + 1):
            d = float(np.linalg.norm(self.traj_pos[i] - robot_xy))
            if d <= self.cfg.lookahead:
                best = self.traj_pos[i]
            else:
                break
        return best

    def _heading_to_tangent(self, robot_yaw: float, k: int) -> float:
        """Heading error relative to the path tangent at waypoint k."""
        k_next = min(k + 1, self._M)
        dx = self.traj_pos[k_next][0] - self.traj_pos[k][0]
        dy = self.traj_pos[k_next][1] - self.traj_pos[k][1]
        tangent_yaw = math.atan2(dy, dx)
        return _wrap_pi(tangent_yaw - robot_yaw)


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
