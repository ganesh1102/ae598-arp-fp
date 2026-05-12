"""Go2 environment configuration for hybrid trigger+NaVILA+tracker evaluation.

Inherits Go2TriggerEvalCfg (which already has camera + static obstacle)
and swaps the command generator to HybridCommandGenerator.
"""

from omni.isaac.lab.utils import configclass

from .go2_trigger_eval_cfg import Go2TriggerEvalCfg
from omni.isaac.leggedloco.leggedloco.mdp.commands import HybridCommandGeneratorCfg


@configclass
class Go2HybridEvalCfg(Go2TriggerEvalCfg):
    """Go2 env: hybrid state-machine command generator."""

    def __post_init__(self):
        super().__post_init__()

        # Replace the command generator with the hybrid FSM.
        # trajectory_file is set by the caller (run_hybrid_eval.py).
        self.commands.base_velocity = HybridCommandGeneratorCfg(
            camera_attr="front_camera",
            robot_attr="robot",
        )
