"""Go2 environment configuration for obstacle-data collection.

Inherits Go2TrajectoryPlayCfg and adds:
  - Front-facing RGB-D camera (320x240, attached to Head_lower)
  - N_OBSTACLES pre-spawned kinematic cylinders spanning [OBS_R_MIN, OBS_R_MAX]
    initialised underground; the collection script activates one per episode.

Usage:
    cfg = Go2DataCollectCfg()
    cfg.commands.base_velocity = TrajectoryCommandGeneratorCfg(...)
    env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array")
"""

import numpy as np

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from .go2_trajectory_play_cfg import Go2TrajectoryPlayCfg

# Continuous radius range — N pre-spawned cylinders covering [R_MIN, R_MAX]
OBS_R_MIN   = 0.15   # [m]
OBS_R_MAX   = 0.75   # [m]
N_OBSTACLES = 12
OBSTACLE_RADII = list(np.linspace(OBS_R_MIN, OBS_R_MAX, N_OBSTACLES))  # length N_OBSTACLES
OBSTACLE_HEIGHT = 1.6   # tall enough to fully occlude the camera [m]
_HIDDEN_Z = -100.0


def _cylinder_obstacle_cfg(idx: int, radius: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obstacle_{idx:02d}",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=OBSTACLE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.20, 0.10)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, _HIDDEN_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class Go2DataCollectCfg(Go2TrajectoryPlayCfg):
    """Go2 trajectory-tracking env augmented with camera + obstacles for data collection."""

    def __post_init__(self):
        super().__post_init__()

        # ── Front-facing RGB-D camera on the robot head ─────────────────────
        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Head_lower/front_camera",
            update_period=0,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 15.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),  # forward-facing, ROS convention
                convention="ros",
            ),
        )

        # ── Pre-spawn N cylinders covering the full radius range ─────────────
        for i, radius in enumerate(OBSTACLE_RADII):
            setattr(self.scene, f"obstacle_{i:02d}", _cylinder_obstacle_cfg(i, radius))
