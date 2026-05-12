"""Hybrid command generator — trajectory tracker + NaVILA + return controller.

State machine (one HybridFSM instance per parallel environment):

    TRACKING  ──[trigger fires]──────────────────────────► AVOIDING
                                                               │
                                [obstacle behind + clearance + STOP/timeout]
                                                               ▼
    TRACKING  ◄──[on path + heading aligned]──────────── RETURNING

TRACKING  : TrajectoryCommandGenerator drives robot; visual trigger checked
            every ``trigger_every`` steps (after ``trigger_cooldown`` re-arm).
AVOIDING  : NavilaCommandGenerator drives robot; trigger suppressed.
RETURNING : TrajectoryCommandGenerator drives robot back to path, resuming
            from the first waypoint past the obstacle (not the nearest behind
            it), so the robot never re-encounters the obstacle.

Extras written each step (accessible via env_raw.extras):
    "hybrid_state"              int   (0/1/2)
    "hybrid_state_name"         str
    "trigger_fired_this_step"   bool
    "handoff_this_step"         bool
    "resume_this_step"          bool
    "state_step_count"          int
    "trigger_inference_ms"      float (last trigger wall-time [ms])
    "navila_query_ms"           float (last NaVILA socket round-trip [ms])
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import CommandTerm

from omni.isaac.leggedloco.leggedloco.mdp.triggers import VisualTrigger, VisualTriggerCfg
from .hybrid_fsm import HybridFSM
from .navila_command_generator import NavilaCommandGenerator
from .navila_command_generator_cfg import NavilaCommandGeneratorCfg
from .return_to_path_command_generator import ReturnToPathController, ReturnToPathCfg
from .trajectory_command_generator import TrajectoryCommandGenerator
from .trajectory_command_generator_cfg import TrajectoryCommandGeneratorCfg

if TYPE_CHECKING:
    from .hybrid_command_generator_cfg import HybridCommandGeneratorCfg


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion (w, x, y, z). Returns shape (E,)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class HybridCommandGenerator(CommandTerm):
    """Velocity command generator implementing the TRACKING/AVOIDING/RETURNING FSM."""

    cfg: HybridCommandGeneratorCfg

    def __init__(self, cfg: HybridCommandGeneratorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_attr]

        # ── Load reference trajectory ──────────────────────────────────────
        npz = np.load(cfg.trajectory_file)
        self._traj_pos = torch.tensor(npz["positions"], dtype=torch.float32,
                                      device=self.device)   # (M+1, 2)
        self._traj_pos_np = npz["positions"].astype(np.float64)  # for return ctrl
        self._M = len(self._traj_pos) - 1

        # ── Sub-components ─────────────────────────────────────────────────
        # Geometric tracker (used in TRACKING state)
        tracker_cfg = TrajectoryCommandGeneratorCfg(
            class_type=TrajectoryCommandGenerator,
            robot_attr=cfg.robot_attr,
            trajectory_file=cfg.trajectory_file,
            eps_wp=cfg.eps_wp,
            lookahead_dist=cfg.lookahead_dist,
            Kp_yaw=cfg.Kp_yaw,
            v_max=cfg.v_max,
            omega_max=cfg.omega_max,
        )
        self._tracker = TrajectoryCommandGenerator(tracker_cfg, env)

        # NaVILA command generator (used in AVOIDING state)
        navila_cfg = NavilaCommandGeneratorCfg(
            class_type=NavilaCommandGenerator,
            server_host=cfg.navila_server_host,
            server_port=cfg.navila_server_port,
            instruction=cfg.avoidance_instruction,
            camera_attr=cfg.camera_attr,
            v_forward=cfg.navila_v_forward,
            omega_turn=cfg.navila_omega_turn,
            num_frames=cfg.navila_num_frames,
        )
        self._navila = NavilaCommandGenerator(navila_cfg, env)

        # Return-to-path controller (used in RETURNING state)
        self._return_ctrl = ReturnToPathController(cfg.return_cfg, self._traj_pos_np)

        # Visual trigger
        trigger_cfg = VisualTriggerCfg(
            checkpoint_path=cfg.trigger_checkpoint,
            threshold=cfg.trigger_threshold,
            device=str(self.device),
        )
        self._trigger = VisualTrigger(trigger_cfg, num_envs=self.num_envs)

        # ── Per-env FSM instances ──────────────────────────────────────────
        self._fsm: list[HybridFSM] = [
            HybridFSM(
                handoff_clearance=cfg.handoff_clearance,
                max_avoiding_steps=cfg.max_avoiding_steps,
                return_radius=cfg.return_cfg.return_radius,
                return_heading_tol=cfg.return_cfg.heading_tol,
                trigger_cooldown=cfg.trigger_cooldown,
            )
            for _ in range(self.num_envs)
        ]

        # ── Mutable buffers ────────────────────────────────────────────────
        self._twist = torch.zeros((self.num_envs, 3), device=self.device)
        self._step_count = 0
        self._env0_trigger_fired = False  # saved before FSM step for accurate logging

        # ── Avoidance execution state (per env) ───────────────────────────
        # NaVILA is queried once per phase with a directive instruction.
        # Each "_q" phase awaits one NaVILA response; "_x" phases execute
        # the action with geometrically-enforced magnitude.
        #
        # Phase sequence:
        #   "turn1_q" → query NaVILA "Turn left 90 degrees now."
        #   "turn1_x" → execute turn (enforced magnitude)
        #   "fwd1_q"  → query NaVILA "Move forward 1 meter."
        #   "fwd1_x"  → execute forward
        #   "turn2_q" → query NaVILA "Turn right 90 degrees now."
        #   "turn2_x" → execute turn (opposite to turn1)
        #   "fwd2_q"  → query NaVILA "Move forward 1 meter."
        #   "fwd2_x"  → execute forward, then signal done
        #   "idle"    → complete; wait for FSM timeout
        self._avoid_phase:     list[str]   = ["idle"] * self.num_envs
        self._avoid_remaining: list[int]   = [0]      * self.num_envs
        self._avoid_omega:     list[float] = [0.0]    * self.num_envs  # signed turn rate
        self._avoid_entry_nq:  list[int]   = [0]      * self.num_envs  # nq snapshot before each _q phase

        # Obstacle waypoint index (index of traj_pos nearest to obstacle)
        self._obs_wp_idx: int = self._M // 2   # updated lazily

        # Timing
        self._last_trigger_ms = 0.0
        self._last_navila_ms  = 0.0

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        return self._twist

    # Expose goal_reached from tracker so run scripts can check it
    @property
    def goal_reached(self) -> torch.Tensor:
        return self._tracker._goal_reached

    # Expose tracking_error from tracker
    def tracking_error(self) -> torch.Tensor:
        return self._tracker.tracking_error()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        for i in ids:
            self._fsm[i].reset()
            self._avoid_phase[i]     = "idle"
            self._avoid_remaining[i] = 0
            self._avoid_omega[i]     = 0.0
            self._avoid_entry_nq[i]  = 0
        self._tracker.reset(env_ids)
        self._navila.reset(env_ids)
        self._trigger.reset(env_ids)
        self._twist[ids] = 0.0
        self._step_count = 0
        return {}

    def compute(self, dt: float):
        self._step_count += 1

        robot_pos   = self.robot.data.root_pos_w[:, :2]   # (E, 2)
        robot_quat  = self.robot.data.root_quat_w          # (E, 4) w,x,y,z
        robot_yaw   = _yaw_from_quat(robot_quat)           # (E,)

        cam = self._env.scene.get("front_camera", None) if hasattr(self._env.scene, 'get') else None
        try:
            cam = self._env.scene["front_camera"]
        except Exception:
            cam = None

        for i in range(self.num_envs):
            fsm = self._fsm[i]
            px, py = float(robot_pos[i, 0]), float(robot_pos[i, 1])
            yaw    = float(robot_yaw[i])
            robot_xy_np = np.array([px, py])

            # ── Trigger check ──────────────────────────────────────────────
            trigger_fired = False
            trigger_ms    = 0.0
            if fsm.trigger_armed and self._step_count % self.cfg.trigger_every == 0:
                obs_pos_t = self._get_obstacle_pos()
                d_obs_t   = float(torch.norm(robot_pos[i] - obs_pos_t))
                e_track   = float(self._tracker.tracking_error()[i])
                min_d     = getattr(self.cfg, "min_trigger_dist", 0.65)

                if self.cfg.threshold_trigger:
                    # Baseline 2: geometric threshold (d_obs or e_track)
                    if d_obs_t >= min_d:
                        trigger_fired = (
                            d_obs_t < self.cfg.d_thresh or e_track > self.cfg.e_thresh
                        )
                elif cam is not None and min_d <= d_obs_t <= self.cfg.max_trigger_dist:
                    # Learned visual trigger (MLP + ResNet-18)
                    rgb_uint8 = cam.data.output["rgb"][i].cpu().numpy()[..., :3].astype(np.uint8)
                    t_ep      = self._step_count * dt
                    t0 = time.perf_counter()
                    fired_t, _ = self._trigger.step(
                        rgb         = rgb_uint8[np.newaxis],
                        d_obs       = np.array([d_obs_t],  dtype=np.float32),
                        e_track     = np.array([e_track],  dtype=np.float32),
                        t_since_vla = np.array([t_ep],     dtype=np.float32),
                    )
                    trigger_ms    = (time.perf_counter() - t0) * 1000.0
                    trigger_fired = bool(fired_t[i])
                    self._last_trigger_ms = trigger_ms

                    # During RETURNING, relax threshold: re-trigger if score > 0.5
                    if (not trigger_fired and fsm.state == HybridFSM.RETURNING
                            and hasattr(self._trigger, "_score")):
                        if float(self._trigger._score[i]) > 0.5:
                            trigger_fired = True

            # ── Obstacle geometry for FSM conditions ───────────────────────
            obs_pos_2d = self._get_obstacle_pos()   # (2,) tensor
            d_obs      = float(torch.norm(robot_pos[i] - obs_pos_2d))
            # Obstacle is "behind" robot if robot-forward dot robot→obstacle < 0
            robot_fwd   = np.array([math.cos(yaw), math.sin(yaw)])
            robot_to_obs = (obs_pos_2d.cpu().numpy() - robot_xy_np)
            obstacle_behind = float(np.dot(robot_fwd, robot_to_obs)) < 0.0

            # ── Path geometry (nearest waypoint distance + heading error) ──
            dists       = np.linalg.norm(self._traj_pos_np - robot_xy_np, axis=1)
            nearest_idx = int(np.argmin(dists))
            path_dist   = float(dists[nearest_idx])
            if nearest_idx < self._M:
                tang    = self._traj_pos_np[nearest_idx + 1] - self._traj_pos_np[nearest_idx]
                tang_yaw = math.atan2(tang[1], tang[0])
            else:
                tang_yaw = yaw
            tangent_err = _wrap_pi(yaw - tang_yaw)

            # ── FSM transition ─────────────────────────────────────────────
            # Save trigger_fired BEFORE FSM step — after TRACKING→AVOIDING
            # the trigger_armed property becomes False, so logging it afterward
            # would always produce False.
            if i == 0:
                self._env0_trigger_fired = trigger_fired

            prev_fsm_state = fsm.state
            navila_done = self._navila.done if i == 0 else False
            fsm.step(
                trigger_fired   = trigger_fired,
                navila_done     = navila_done,
                obstacle_behind = obstacle_behind,
                d_obs           = d_obs,
                path_dist       = path_dist,
                heading_error   = tangent_err,
            )

            # ── On AVOIDING→RETURNING: resume tracker past the obstacle ────
            if fsm.handoff_this_step:
                resume_wp, near_side = self._resume_wp(robot_pos[i], obs_pos_2d)
                self._tracker._wp_idx[i] = resume_wp
                self._navila.done = False
                # On near-side timeout (NaVILA didn't clear obstacle), skip the
                # post-return cooldown so the trigger re-arms immediately for
                # another NaVILA attempt.
                if near_side:
                    fsm.skip_next_cooldown = True

            # ── On any entry into AVOIDING: begin 4-phase NaVILA maneuver ──
            if prev_fsm_state != HybridFSM.AVOIDING and fsm.state == HybridFSM.AVOIDING:
                self._navila.done            = False
                self._navila.last_cmd        = {}   # clear cached action from prior cycle
                self._navila.cfg.instruction = "Turn left 90 degrees now."
                self._navila._remaining[i]   = 0    # force immediate query
                self._avoid_entry_nq[i]      = self._navila.navila_queries
                self._avoid_phase[i]         = "turn1_q"
                self._avoid_remaining[i]     = 0
                self._avoid_omega[i]         = 0.0

        # ── Delegate to sub-generators for active states ───────────────────
        # Tracker drives both TRACKING and RETURNING states.
        self._tracker.compute(dt)
        tracker_mask = torch.tensor(
            [fsm.state in (HybridFSM.TRACKING, HybridFSM.RETURNING) for fsm in self._fsm],
            device=self.device, dtype=torch.bool
        )
        self._twist[tracker_mask] = self._tracker._twist[tracker_mask]

        # ── Avoidance executor ─────────────────────────────────────────────────
        # NaVILA is queried once per "_q" phase for direction only.
        # Step counts are computed geometrically from the config so the maneuver
        # is large enough to clear the obstacle reliably.
        avoiding_mask = torch.tensor(
            [fsm.state == HybridFSM.AVOIDING for fsm in self._fsm],
            device=self.device, dtype=torch.bool
        )
        if avoiding_mask.any():
            omega_cfg = self.cfg.navila_omega_turn
            v_cfg     = self.cfg.navila_v_forward

            # Geometric step counts (same for every cycle)
            turn1_steps = max(1, int(math.radians(self.cfg.avoid_turn_deg)    / (omega_cfg * dt)))
            fwd1_steps  = max(1, int(self.cfg.avoid_forward_m                  / (v_cfg     * dt)))
            turn2_steps = max(1, int(math.radians(self.cfg.avoid_realign_deg) / (omega_cfg * dt)))
            fwd2_steps  = max(1, int(self.cfg.avoid_fwd2_m                    / (v_cfg     * dt)))

            navila_needed = any(
                self._avoid_phase[i].endswith("_q")
                for i in range(self.num_envs)
                if self._fsm[i].state == HybridFSM.AVOIDING
            )
            if navila_needed:
                t0 = time.perf_counter()
                self._navila.compute(dt)
                self._last_navila_ms = (time.perf_counter() - t0) * 1000.0

            for i, fsm in enumerate(self._fsm):
                if fsm.state != HybridFSM.AVOIDING:
                    continue
                phase = self._avoid_phase[i]
                nq    = self._navila.navila_queries

                # ── Query phases: wait for NaVILA response, then use
                # NaVILA's direction but geometric magnitude. ───────────────
                if phase == "turn1_q":
                    if nq > self._avoid_entry_nq[i]:
                        action = (self._navila.last_cmd or {}).get("action", "TURN_LEFT")
                        omega = omega_cfg if action != "TURN_RIGHT" else -omega_cfg
                        self._avoid_omega[i] = omega        # save direction for turn2
                        geo_twist = torch.tensor([0.0, 0.0, omega], device=self.device)
                        self._navila._twist[i] = geo_twist  # _x phases copy this
                        self._twist[i] = geo_twist
                        self._avoid_phase[i]     = "turn1_x"
                        self._avoid_remaining[i] = turn1_steps
                    else:
                        self._twist[i] = torch.zeros(3, device=self.device)

                elif phase == "fwd1_q":
                    if nq > self._avoid_entry_nq[i]:
                        geo_twist = torch.tensor([v_cfg, 0.0, 0.0], device=self.device)
                        self._navila._twist[i] = geo_twist
                        self._twist[i] = geo_twist
                        self._avoid_phase[i]     = "fwd1_x"
                        self._avoid_remaining[i] = fwd1_steps
                    else:
                        self._twist[i] = torch.zeros(3, device=self.device)

                elif phase == "turn2_q":
                    if nq > self._avoid_entry_nq[i]:
                        omega = -self._avoid_omega[i]       # invert for realign
                        self._avoid_omega[i] = omega
                        geo_twist = torch.tensor([0.0, 0.0, omega], device=self.device)
                        self._navila._twist[i] = geo_twist
                        self._twist[i] = geo_twist
                        self._avoid_phase[i]     = "turn2_x"
                        self._avoid_remaining[i] = turn2_steps
                    else:
                        self._twist[i] = torch.zeros(3, device=self.device)

                elif phase == "fwd2_q":
                    if nq > self._avoid_entry_nq[i]:
                        geo_twist = torch.tensor([v_cfg, 0.0, 0.0], device=self.device)
                        self._navila._twist[i] = geo_twist
                        self._twist[i] = geo_twist
                        self._avoid_phase[i]     = "fwd2_x"
                        self._avoid_remaining[i] = fwd2_steps
                    else:
                        self._twist[i] = torch.zeros(3, device=self.device)

                # ── Execute phases: hold geometric twist, count down ────────
                elif phase == "turn1_x":
                    self._twist[i] = self._navila._twist[i].clone()
                    self._avoid_remaining[i] -= 1
                    if self._avoid_remaining[i] <= 0:
                        self._avoid_phase[i] = "fwd1_q"
                        self._navila.cfg.instruction = "Move forward 1 meter."
                        self._navila._remaining[i] = 0
                        self._avoid_entry_nq[i] = nq

                elif phase == "fwd1_x":
                    self._twist[i] = self._navila._twist[i].clone()
                    self._avoid_remaining[i] -= 1
                    if self._avoid_remaining[i] <= 0:
                        self._avoid_phase[i] = "turn2_q"
                        self._navila.cfg.instruction = "Turn right 90 degrees now."
                        self._navila._remaining[i] = 0
                        self._avoid_entry_nq[i] = nq

                elif phase == "turn2_x":
                    self._twist[i] = self._navila._twist[i].clone()
                    self._avoid_remaining[i] -= 1
                    if self._avoid_remaining[i] <= 0:
                        self._avoid_phase[i] = "fwd2_q"
                        self._navila.cfg.instruction = "Move forward 1 meter."
                        self._navila._remaining[i] = 0
                        self._avoid_entry_nq[i] = nq

                elif phase == "fwd2_x":
                    self._twist[i] = self._navila._twist[i].clone()
                    self._avoid_remaining[i] -= 1
                    if self._avoid_remaining[i] <= 0:
                        self._avoid_phase[i] = "idle"
                        self._twist[i] = torch.zeros(3, device=self.device)
                        if i == 0:
                            self._navila.done = True

                else:  # idle — wait for FSM timeout
                    self._twist[i] = torch.zeros(3, device=self.device)

        # ── Write extras ───────────────────────────────────────────────────
        # Use env 0 for scalar extras (num_envs=1 in typical eval)
        fsm0 = self._fsm[0]
        extras = self._env.extras
        extras["hybrid_state"]            = fsm0.state
        extras["hybrid_state_name"]       = fsm0.state_name
        extras["trigger_fired_this_step"] = self._env0_trigger_fired
        extras["handoff_this_step"]       = fsm0.handoff_this_step
        extras["resume_this_step"]        = fsm0.resume_this_step
        extras["state_step_count"]        = fsm0.state_step_count
        extras["trigger_inference_ms"]    = self._last_trigger_ms
        extras["navila_query_ms"]         = self._last_navila_ms
        phase0 = self._avoid_phase[0]
        extras["navila_action"]           = phase0
        extras["navila_raw"]              = self._navila.last_raw
        extras["navila_queries"]          = self._navila.navila_queries
        extras["trigger_score"]           = float(self._trigger._score[0]) if hasattr(self._trigger, "_score") else 0.0
        extras["avoiding_steps"]          = fsm0.avoiding_steps
        extras["cooldown_remaining"]      = fsm0.cooldown_remaining

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resume_wp(self, robot_pos_2d: torch.Tensor,
                   obs_pos_2d: torch.Tensor) -> tuple[int, bool]:
        """Choose the tracker waypoint to resume at after NaVILA avoidance.

        Returns (wp_idx, near_side) where near_side=True means NaVILA timed
        out without clearing the obstacle (robot still on approach side).

        Far side: resume at first waypoint past the obstacle that is
        physically clear of the obstacle footprint.

        Near side (timeout): resume at robot's nearest waypoint so tracker
        drives forward again; trigger re-arms for another NaVILA attempt.
        """
        obs_np    = obs_pos_2d.cpu().numpy()
        robot_np  = robot_pos_2d.cpu().numpy()

        obs_dists   = np.linalg.norm(self._traj_pos_np - obs_np,   axis=1)
        robot_dists = np.linalg.norm(self._traj_pos_np - robot_np, axis=1)

        obs_wp   = int(np.argmin(obs_dists))
        robot_wp = int(np.argmin(robot_dists))

        clearance = getattr(self.cfg, "resume_clearance", 1.2)

        if robot_wp > obs_wp:
            # Robot cleared the obstacle — skip to first physically clear wp
            resume = obs_wp
            while resume < self._M and obs_dists[resume] < clearance:
                resume += 1
            near_side = False
        else:
            # Robot still on near side (timeout without clearing) — resume at
            # nearest reachable wp; caller will skip cooldown for re-trigger
            resume = robot_wp
            near_side = True

        return min(resume, self._M), near_side

    def _get_obstacle_pos(self) -> torch.Tensor:
        """Try to get obstacle XY from scene; fall back to a far-away point."""
        try:
            obs = self._env.scene["obstacle"]
            return obs.data.root_pos_w[0, :2]
        except Exception:
            return torch.tensor([1e6, 1e6], device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
