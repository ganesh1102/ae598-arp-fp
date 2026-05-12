from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils.configclass import configclass

from .hybrid_command_generator import HybridCommandGenerator
from .return_to_path_command_generator import ReturnToPathCfg

_AVOIDANCE_INSTRUCTION = (
    "You are a navigation module on a quadruped robot. "
    "When you encounter an obstacle, you must avoid it by going around it. "
    "The obstacle is a box located somewhere in front of you. "
    "Turn left until the box is no longer in front of you, then walk forward. "
    "When you are moving around the obstacle, keep a safe distance. "
    "Once you have passed the obstacle, stop."
)


@configclass
class HybridCommandGeneratorCfg(CommandTermCfg):
    """Configuration for the HybridCommandGenerator state machine.

    Wires together:
      TRACKING  → TrajectoryCommandGenerator (geometric tracker)
      AVOIDING  → NavilaCommandGenerator     (VLA avoidance)
      RETURNING → ReturnToPathController     (pure-pursuit back to path)
    with a VisualTrigger gating TRACKING → AVOIDING.
    """

    class_type: type = HybridCommandGenerator

    # ── Reference trajectory ───────────────────────────────────────────────
    trajectory_file: str = "UNSET"
    """Path to .npz produced by optimal_trajectory_solver.py."""

    robot_attr: str = "robot"
    camera_attr: str = "front_camera"

    # ── Trajectory tracker parameters (mirrors TrajectoryCommandGeneratorCfg) ─
    eps_wp: float = 0.15
    lookahead_dist: float = 1.0
    Kp_yaw: float = 2.0
    v_max: float = 0.5
    omega_max: float = 1.0

    # ── Visual trigger ─────────────────────────────────────────────────────
    trigger_checkpoint: str = "checkpoints/trigger_real_visual.pt"
    trigger_threshold: float = 0.5
    trigger_every: int = 5       # steps between trigger checks (→ 10 Hz)
    trigger_cooldown: int = 100  # steps before re-arming after RETURNING→TRACKING

    # ── Threshold trigger (Baseline 2, Section V.e) ────────────────────────
    threshold_trigger: bool = False
    """If True, bypass visual MLP and fire on geometric d_obs/e_track conditions."""
    d_thresh: float = 2.0
    """Threshold trigger fires when obstacle is closer than this [m]."""
    e_thresh: float = 0.3
    """Threshold trigger fires when tracking error exceeds this [m]."""

    # ── NaVILA server ──────────────────────────────────────────────────────
    navila_server_host: str = "localhost"
    navila_server_port: int = 15432
    avoidance_instruction: str = _AVOIDANCE_INSTRUCTION
    navila_v_forward: float = 0.3
    navila_omega_turn: float = 0.5
    navila_num_frames: int = 8

    # ── State machine parameters ───────────────────────────────────────────
    handoff_clearance: float = 1.5
    """AVOIDING→RETURNING: obstacle must be > this far behind robot [m]."""
    max_avoiding_steps: int = 1000
    """Force AVOIDING→RETURNING after this many steps regardless."""
    min_trigger_dist: float = 0.3
    """Trigger is suppressed when obstacle is closer than this [m].
    Below this distance the robot is too close to identify the obstacle reliably."""
    max_trigger_dist: float = 3.5
    """Trigger is suppressed when obstacle is farther than this [m].
    NaVILA's avoidance arc (45° turn + 75cm forward) clears a ~2m obstacle
    when triggered at ~3m; below ~2m the arc still converges toward the obstacle."""
    resume_clearance: float = 1.2
    """On AVOIDING→RETURNING, resume tracker at the first waypoint that is
    more than this distance from the obstacle centre [m].  Must exceed the
    obstacle half-width + robot radius so the path is physically clear."""

    # ── Avoidance maneuver geometry ────────────────────────────────────────
    # NaVILA's first turn decides direction; the rest is deterministic.
    avoid_turn_deg:   float = 90.0
    """Degrees to turn (left or right as NaVILA decides) to clear the obstacle."""
    avoid_forward_m:  float = 1.0
    """Metres to move forward after the initial turn."""
    avoid_realign_deg: float = 90.0
    """Degrees to turn back (opposite direction) to re-align with original heading."""
    avoid_fwd2_m:     float = 1.0
    """Metres to move forward after re-aligning before signalling done."""

    # ── Return controller ──────────────────────────────────────────────────
    return_cfg: ReturnToPathCfg = ReturnToPathCfg()

    # ── Disable automatic resampling ───────────────────────────────────────
    resampling_time_range: tuple = (1.0e9, 1.0e9)
