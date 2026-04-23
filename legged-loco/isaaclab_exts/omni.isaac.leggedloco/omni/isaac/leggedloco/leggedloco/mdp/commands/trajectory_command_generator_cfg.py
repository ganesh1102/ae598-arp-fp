from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils.configclass import configclass

from .trajectory_command_generator import TrajectoryCommandGenerator


@configclass
class TrajectoryCommandGeneratorCfg(CommandTermCfg):
    """Configuration for the geometric trajectory tracking controller.

    Drop-in replacement for UniformVelocityCommandCfg under the "base_velocity"
    command name.  Set trajectory_file to the .npz produced by:

        python scripts/optimal_trajectory_solver.py --save traj.npz ...
    """

    class_type: type = TrajectoryCommandGenerator

    robot_attr: str = MISSING
    """Scene attribute name for the robot articulation (e.g. "robot")."""

    trajectory_file: str = MISSING
    """Path to .npz file with keys: positions (M+1,2), speeds (M+1,), headings (M+1,), times (M+1,)."""

    # Geometric controller parameters — see Section IV-A of the paper
    eps_wp: float = 0.15
    """Waypoint-advance threshold ε_wp [m]. Matches Table I of the paper."""

    lookahead_dist: float = 1.0
    """Pure-pursuit lookahead radius [m]. Target the furthest waypoint within
    this distance so vx stays near v_max regardless of waypoint spacing."""

    Kp_yaw: float = 2.0
    """Proportional gain for heading error → ω_z."""

    v_max: float = 0.5
    """Maximum forward velocity [m/s]. Must match πloco training range."""

    omega_max: float = 1.0
    """Maximum yaw rate [rad/s]. Must match πloco training range."""

    # Disable automatic resampling — trajectory is fixed for the episode
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
