"""Deterministic geometric tracking controller for pre-computed optimal trajectories.

Implements the geometric conversion from Section IV-A of the paper:

    δ_t     = p*_k − p_t
    ψ_des,t = atan2(δ_y, δ_x)
    v_x,t   = ‖δ_t‖ · cos(ψ_des,t − θ_t)     clipped to [0, v_max]
    ω_z,t   = Kp · (ψ_des,t − θ_t)            clipped to ±ω_max

Plugs in as a drop-in replacement for UniformVelocityCommandCfg under the
"base_velocity" command name, so the frozen locomotion policy sees the same
(num_envs, 3) = [v_x, v_y, ω_z] command tensor it was trained on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from .trajectory_command_generator_cfg import TrajectoryCommandGeneratorCfg


class TrajectoryCommandGenerator(CommandTerm):
    """Geometric waypoint-to-velocity converter for optimal trajectory tracking.

    Loads P* = {(p*_k, v*_k, θ*_k)} from an .npz file produced by
    optimal_trajectory_solver.py, then at each control step computes the
    (v_x, 0, ω_z) command that steers the robot toward the active waypoint.

    All parallel environments track the SAME trajectory but maintain
    independent waypoint indices, so they can be at different progress
    levels after desynchronised resets.
    """

    cfg: TrajectoryCommandGeneratorCfg

    def __init__(self, cfg: TrajectoryCommandGeneratorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.robot_attr]

        # Load trajectory P* from .npz
        data = np.load(cfg.trajectory_file, allow_pickle=True)
        self._traj_pos = torch.tensor(
            data["positions"], dtype=torch.float32, device=self.device
        )  # (M+1, 2)  world-frame [px, py]
        self._M = int(self._traj_pos.shape[0]) - 1  # number of intervals

        # Per-environment mutable state
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._twist = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Trajectory waypoints (cyan small spheres, one per waypoint)
        _traj_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrajectoryCmd/waypoints",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
                )
            },
        )
        self._traj_vis_marker = VisualizationMarkers(_traj_cfg)

        # Start (green) and goal (red) larger spheres
        _start_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrajectoryCmd/start",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.25,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        )
        self._start_marker = VisualizationMarkers(_start_cfg)

        _goal_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrajectoryCmd/goal",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.25,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        self._goal_marker = VisualizationMarkers(_goal_cfg)

        # Pre-compute all marker positions (static throughout the episode)
        self._traj_vis_pos = torch.zeros(self._M + 1, 3, device=self.device)
        self._traj_vis_pos[:, :2] = self._traj_pos
        self._traj_vis_pos[:, 2] = 0.1

        self._start_vis_pos = torch.zeros(1, 3, device=self.device)
        self._start_vis_pos[0, :2] = self._traj_pos[0]
        self._start_vis_pos[0, 2] = 0.25

        self._goal_vis_pos = torch.zeros(1, 3, device=self.device)
        self._goal_vis_pos[0, :2] = self._traj_pos[-1]
        self._goal_vis_pos[0, 2] = 0.25

        print(
            f"[TrajectoryCommandGenerator] Loaded P* with {self._M + 1} waypoints "
            f"from '{cfg.trajectory_file}'."
        )

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Velocity command (v_x, v_y=0, ω_z) in robot base frame. Shape (num_envs, 3)."""
        return self._twist

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._wp_idx[env_ids] = 0
        self._goal_reached[env_ids] = False
        self._twist[env_ids] = 0.0
        return {}

    def compute(self, dt: float):
        robot_pos = self.robot.data.root_pos_w[:, :2]          # (E, 2)
        robot_heading = self._yaw_from_quat(self.robot.data.root_quat_w)  # (E,)

        for i in range(self.num_envs):
            if self._goal_reached[i]:
                self._twist[i] = 0.0
                continue

            pos_i = robot_pos[i]
            k = int(self._wp_idx[i])

            # Advance k to the nearest waypoint on the remaining trajectory.
            # This handles the case where the robot skips waypoints (e.g. due
            # to the lookahead) or spawns ahead of waypoint 0.
            dists = torch.norm(self._traj_pos[k:] - pos_i.unsqueeze(0), dim=1)
            k = k + int(torch.argmin(dists))
            self._wp_idx[i] = k

            if k >= self._M:
                if torch.norm(self._traj_pos[-1] - pos_i) < self.cfg.eps_wp * 2:
                    self._goal_reached[i] = True
                self._twist[i] = 0.0
                continue

            # Pure-pursuit lookahead: scan forward from k along the trajectory
            # and target the furthest waypoint within lookahead_dist.
            # This keeps vx near v_max regardless of waypoint spacing.
            target_k = k
            for j in range(k, self._M + 1):
                if torch.norm(self._traj_pos[j] - pos_i) <= self.cfg.lookahead_dist:
                    target_k = j
                elif target_k > k:
                    break  # exited the lookahead bubble

            # Geometric conversion (Section IV-A)
            delta = self._traj_pos[target_k] - pos_i
            psi_des = torch.atan2(delta[1], delta[0])
            heading_err = self._wrap(psi_des - robot_heading[i])

            vx = (self.cfg.v_max * torch.cos(heading_err)).clamp(0.0, self.cfg.v_max)
            oz = (self.cfg.Kp_yaw * heading_err).clamp(-self.cfg.omega_max, self.cfg.omega_max)

            self._twist[i, 0] = vx
            self._twist[i, 1] = 0.0
            self._twist[i, 2] = oz

        # Refresh static markers every step so they survive scene reloads
        self._traj_vis_marker.visualize(self._traj_vis_pos)
        self._start_marker.visualize(self._start_vis_pos)
        self._goal_marker.visualize(self._goal_vis_pos)

    # ------------------------------------------------------------------
    # Required CommandTerm stubs
    # ------------------------------------------------------------------

    def _update_command(self): pass
    def _update_metrics(self): pass
    def _resample_command(self, env_ids: Sequence[int]): pass

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_wp_marker"):
                cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/TrajectoryCmd/waypoint"
                cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self._wp_marker = VisualizationMarkers(cfg)

                cfg2 = BLUE_ARROW_X_MARKER_CFG.copy()
                cfg2.prim_path = "/Visuals/TrajectoryCmd/vel_cmd"
                cfg2.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self._cmd_marker = VisualizationMarkers(cfg2)
            self._wp_marker.set_visibility(True)
            self._cmd_marker.set_visibility(True)
        else:
            if hasattr(self, "_wp_marker"):
                self._wp_marker.set_visibility(False)
                self._cmd_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        base_pos = self.robot.data.root_pos_w.clone()
        base_pos[:, 2] += 0.5

        # Show active waypoint
        wp_pos = torch.zeros_like(base_pos)
        for i in range(self.num_envs):
            k = int(self._wp_idx[i].clamp(max=self._M))
            wp_pos[i, :2] = self._traj_pos[k]
            wp_pos[i, 2] = base_pos[i, 2]

        # Arrow from base toward waypoint
        delta_xy = wp_pos[:, :2] - base_pos[:, :2]
        heading = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
        zeros = torch.zeros_like(heading)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
        scale = torch.ones(self.num_envs, 3, device=self.device) * 0.3

        self._wp_marker.visualize(wp_pos, quat, scale)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def goal_reached(self) -> torch.Tensor:
        """Boolean mask: which envs have reached the trajectory goal."""
        return self._goal_reached

    @property
    def active_waypoint_idx(self) -> torch.Tensor:
        return self._wp_idx

    def tracking_error(self) -> torch.Tensor:
        """e_t = ‖p_t − p*_k‖  (Eq. 4). Shape (num_envs,)."""
        pos = self.robot.data.root_pos_w[:, :2]
        errors = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            k = int(self._wp_idx[i].clamp(max=self._M))
            errors[i] = torch.norm(pos[i] - self._traj_pos[k])
        return errors

    @staticmethod
    def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
        """Extract yaw from (w, x, y, z) quaternions. Shape: (E,)."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2.0 * torch.pi) - torch.pi
