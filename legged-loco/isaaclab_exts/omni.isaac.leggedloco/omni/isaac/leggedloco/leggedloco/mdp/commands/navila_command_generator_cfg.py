from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils.configclass import configclass

from .navila_command_generator import NavilaCommandGenerator

_DEFAULT_INSTRUCTION = (
    "You are a navigation module on a quadruped robot. "
    "Avoid the obstacle on your path by going around it on the left. "
    "Then move in front of it, facing the forward direction, and stop."
)


@configclass
class NavilaCommandGeneratorCfg(CommandTermCfg):
    """Configuration for the NaVILA-driven velocity command generator.

    Drop-in replacement for ``TrajectoryCommandGeneratorCfg`` under the
    ``base_velocity`` command name.  The NavilaCommandGenerator connects to a
    running ``navila_server.py`` (navila conda env) via TCP and translates
    NaVILA's discrete action outputs into (v_x, v_y, ω_z) commands held for
    the appropriate number of control steps.
    """

    class_type: type = NavilaCommandGenerator

    # ── NaVILA server connection ───────────────────────────────────────────
    server_host: str = "localhost"
    """Hostname of the navila_server process."""

    server_port: int = 15432
    """TCP port of the navila_server process."""

    # ── Navigation instruction ─────────────────────────────────────────────
    instruction: str = _DEFAULT_INSTRUCTION
    """Natural-language navigation instruction passed verbatim to NaVILA."""

    # ── Scene attributes ───────────────────────────────────────────────────
    camera_attr: str = "front_camera"
    """InteractiveScene key for the front RGB camera."""

    # ── Velocity command parameters ────────────────────────────────────────
    v_forward: float = 0.3
    """Forward velocity [m/s] used when NaVILA says MOVE_FORWARD."""

    omega_turn: float = 0.5
    """Yaw rate [rad/s] used when NaVILA says TURN_LEFT / TURN_RIGHT."""

    # ── Frame buffer ───────────────────────────────────────────────────────
    num_frames: int = 8
    """Rolling history length sent to NaVILA (must match model training)."""

    # ── Disable automatic command resampling ──────────────────────────────
    resampling_time_range: tuple = (1.0e9, 1.0e9)
