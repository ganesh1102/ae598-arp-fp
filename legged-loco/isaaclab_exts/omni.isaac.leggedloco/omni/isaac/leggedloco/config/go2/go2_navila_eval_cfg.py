"""Go2 environment configuration for NaVILA evaluation.

Inherits Go2TrajectoryPlayCfg and adds:
  - Front-facing RGB camera (verbatim copy from go2_datacollect_cfg.py —
    same geometry so NaVILA sees the same view it was trained on)
  - Static cuboid obstacle on the planned path (kinematic, hidden underground
    initially; teleported to the desired position by run_navila_eval.py)
  - Replaces the TrajectoryCommandGenerator with NavilaCommandGenerator so
    NaVILA drives the robot instead of a geometric tracker
"""

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from .go2_trajectory_play_cfg import Go2TrajectoryPlayCfg
from omni.isaac.leggedloco.leggedloco.mdp.commands import NavilaCommandGeneratorCfg

OBSTACLE_SIZE = (0.5, 0.5, 1.5)   # (width, depth, height) [m]
_OBSTACLE_H   = OBSTACLE_SIZE[2]
_HIDDEN_Z     = -100.0


@configclass
class Go2NavilaEvalCfg(Go2TrajectoryPlayCfg):
    """Go2 env: NaVILA planner + camera + static obstacle."""

    def __post_init__(self):
        super().__post_init__()

        # ── Camera: verbatim copy from go2_datacollect_cfg.py ───────────────
        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Head_lower/front_camera",
            update_period=0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 15.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
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

        # ── Swap command generator: trajectory tracker → NaVILA ─────────────
        # server_host / server_port / instruction overridden at runtime by
        # run_navila_eval.py after env construction.
        self.commands.base_velocity = NavilaCommandGeneratorCfg(
            camera_attr="front_camera",
            v_forward=0.3,
            omega_turn=0.5,
        )
