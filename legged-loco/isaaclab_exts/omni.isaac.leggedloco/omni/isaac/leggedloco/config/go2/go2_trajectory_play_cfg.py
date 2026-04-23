"""Go2 environment configuration for trajectory-tracking inference.

Inherits everything from Go2BaseRoughEnvCfg_PLAY and only overrides what is
needed for trajectory tracking:
  - Flat terrain (OCP P* is computed on flat ground)
  - TrajectoryCommandGenerator replaces UniformVelocityCommandCfg so the
    locomotion policy receives geometric tracking commands instead of
    randomly sampled velocity setpoints
  - Domain randomisation disabled (inference mode)

Usage:
    env_cfg = Go2TrajectoryPlayCfg()
    env_cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(
        robot_attr="robot",
        trajectory_file="path/to/traj.npz",
    )
"""

from omni.isaac.lab.utils import configclass

from omni.isaac.leggedloco.config.go2.go2_low_base_cfg import (
    Go2BaseRoughEnvCfg_PLAY,
)
from omni.isaac.leggedloco.leggedloco.mdp.commands import (
    TrajectoryCommandGeneratorCfg,
)


@configclass
class Go2TrajectoryPlayCfg(Go2BaseRoughEnvCfg_PLAY):
    """Go2 trajectory-tracking env (inference only, locomotion policy frozen)."""

    def __post_init__(self):
        super().__post_init__()

        # ── Terrain: flat, so P* computed offline remains valid ──────────────
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

        # ── Single env for inference; bump up for parallelised eval ─────────
        self.scene.num_envs = 1

        # ── Disable remaining randomisation ──────────────────────────────────
        self.events.actuator_gains = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False

        # ── Swap velocity command source: random → geometric tracking ────────
        # trajectory_file must be set by the caller before env creation.
        self.commands.base_velocity = TrajectoryCommandGeneratorCfg(
            robot_attr="robot",
            trajectory_file="UNSET",   # overridden in track_trajectory.py
            eps_wp=0.15,
            lookahead_dist=1.0,
            Kp_yaw=2.0,
            v_max=0.5,
            omega_max=1.0,
        )
