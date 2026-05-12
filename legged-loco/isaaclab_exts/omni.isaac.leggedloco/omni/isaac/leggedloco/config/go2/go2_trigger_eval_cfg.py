"""Go2 environment configuration for visual-trigger evaluation.

Inherits Go2TrajectoryPlayCfg and adds:
  - Front-facing RGB-D camera (identical to go2_datacollect_cfg.py — do not
    change any camera parameters; the trigger's training distribution depends
    on this exact setup)
  - One static kinematic cuboid obstacle (hidden underground initially;
    the run script teleports it to the desired path position)
"""

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from .go2_trajectory_play_cfg import Go2TrajectoryPlayCfg

OBSTACLE_SIZE   = (0.5, 0.5, 1.5)   # (width, depth, height) [m]
_OBSTACLE_H     = OBSTACLE_SIZE[2]
_HIDDEN_Z       = -100.0


@configclass
class Go2TriggerEvalCfg(Go2TrajectoryPlayCfg):
    """Go2 trajectory-tracking env with camera + static obstacle for trigger eval."""

    def __post_init__(self):
        super().__post_init__()

        # ── Camera: verbatim copy from go2_datacollect_cfg.py ───────────────
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
                rot=(0.5, -0.5, 0.5, -0.5),   # forward-facing, ROS convention
                convention="ros",
            ),
        )

        # ── Static obstacle: kinematic cuboid, hidden underground initially ──
        self.scene.obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/obstacle",
            spawn=sim_utils.CuboidCfg(
                size=OBSTACLE_SIZE,
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
