"""NaVILA command generator for the Go2 locomotion pipeline.

Connects to a running ``navila_server.py`` (navila conda env) over TCP.
Maintains a rolling 8-frame RGB history from the robot's front camera.
On each query, sends frames + instruction to the server and holds the
returned velocity command for the duration implied by the discrete action.

NaVILA action → Go2 command mapping
-------------------------------------
  MOVE_FORWARD N cm  →  v_x = cfg.v_forward [m/s]  for  N/100/v_forward/dt  steps
  TURN_LEFT    N deg →  ω_z = +cfg.omega_turn [rad/s]  for  N*π/180/ω/dt  steps
  TURN_RIGHT   N deg →  ω_z = −cfg.omega_turn [rad/s]  for  N*π/180/ω/dt  steps
  STOP               →  zeros, self.done = True

Import-clean of NaVILA / llava — can run inside the isaaclab conda env.
"""

from __future__ import annotations

import base64
import io
import json
import math
import socket
import time
from collections import deque
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import CommandTerm

if TYPE_CHECKING:
    from .navila_command_generator_cfg import NavilaCommandGeneratorCfg


class NavilaCommandGenerator(CommandTerm):
    """Velocity command generator backed by NaVILA VLM queries over TCP."""

    cfg: NavilaCommandGeneratorCfg

    def __init__(self, cfg: NavilaCommandGeneratorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Control timestep (set lazily in first compute() when env is ready)
        self._sim_dt: float | None = None

        # Velocity command buffer: (num_envs, 3) → [v_x, v_y, ω_z]
        self._twist = torch.zeros((self.num_envs, 3), device=self.device)

        # Rolling RGB frame history (PIL Images — one deque per env)
        self._frame_buf: list[deque] = [
            deque(maxlen=cfg.num_frames) for _ in range(self.num_envs)
        ]

        # Steps remaining for the current command (one per env)
        self._remaining: list[int] = [0] * self.num_envs

        # Episode-level done flag (NaVILA emitted STOP)
        self.done: bool = False

        # Total NaVILA queries this episode
        self.navila_queries: int = 0

        # Last raw text + parsed command from NaVILA
        self.last_raw: str = ""
        self.last_cmd: dict = {}

        # TCP socket connection (established lazily)
        self._sock: socket.socket | None = None
        self._rbuf: bytes = b""

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        return self._twist

    def capture_frame(self, env_idx: int, rgb_np: np.ndarray):
        """Append one RGB frame to env_idx's buffer without querying NaVILA.

        Call every step during TRACKING and RETURNING so that by the time
        AVOIDING starts the buffer already holds a real sequential approach
        sequence rather than black padding frames.
        """
        from PIL import Image as _PIL
        self._frame_buf[env_idx].append(_PIL.fromarray(rgb_np.astype(np.uint8)))

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        for i in ids:
            self._frame_buf[i].clear()
            self._remaining[i] = 0
        self._twist[ids] = 0.0
        self.done = False
        self.navila_queries = 0
        self.last_raw = ""
        self.last_cmd = {}
        return {}

    def compute(self, dt: float):
        if self._sim_dt is None:
            self._sim_dt = dt

        cam = self._env.scene[self.cfg.camera_attr]

        for i in range(self.num_envs):
            # ── Capture current RGB frame ──────────────────────────────────
            rgb_raw = cam.data.output["rgb"][i].cpu().numpy()   # (H, W, 4) RGBA
            rgb_uint8 = rgb_raw[..., :3].astype(np.uint8)       # (H, W, 3) RGB

            from PIL import Image as _PIL
            frame = _PIL.fromarray(rgb_uint8)
            self._frame_buf[i].append(frame)

            # ── Query NaVILA when current command expires ──────────────────
            if self._remaining[i] <= 0 and not self.done:
                cmd = self._query(list(self._frame_buf[i]))
                self.last_cmd = cmd
                self.navila_queries += 1
                self._apply_command(i, cmd, dt)

            elif not self.done:
                self._remaining[i] -= 1

    def _apply_command(self, env_idx: int, cmd: dict, dt: float):
        action = cmd.get("action", "STOP")

        if action == "MOVE_FORWARD":
            dist_m  = cmd.get("distance_cm", 25) / 100.0
            steps   = max(1, int(round(dist_m / self.cfg.v_forward / dt)))
            self._twist[env_idx] = torch.tensor(
                [self.cfg.v_forward, 0.0, 0.0], device=self.device
            )
            self._remaining[env_idx] = steps - 1

        elif action == "TURN_LEFT":
            angle_r = math.radians(cmd.get("degree", 15))
            steps   = max(1, int(round(angle_r / self.cfg.omega_turn / dt)))
            self._twist[env_idx] = torch.tensor(
                [0.0, 0.0, self.cfg.omega_turn], device=self.device
            )
            self._remaining[env_idx] = steps - 1

        elif action == "TURN_RIGHT":
            angle_r = math.radians(cmd.get("degree", 15))
            steps   = max(1, int(round(angle_r / self.cfg.omega_turn / dt)))
            self._twist[env_idx] = torch.tensor(
                [0.0, 0.0, -self.cfg.omega_turn], device=self.device
            )
            self._remaining[env_idx] = steps - 1

        else:  # STOP or unrecognised
            self._twist[env_idx] = torch.zeros(3, device=self.device)
            self._remaining[env_idx] = 0
            self.done = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Socket communication
    # ------------------------------------------------------------------

    def _connect(self, retries: int = 30, delay: float = 1.0):
        for attempt in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.cfg.server_host, self.cfg.server_port))
                self._sock = s
                self._rbuf = b""
                print(f"[NavilaCommandGenerator] Connected to server "
                      f"{self.cfg.server_host}:{self.cfg.server_port}")
                return
            except ConnectionRefusedError:
                if attempt < retries - 1:
                    time.sleep(delay)
        raise RuntimeError(
            f"NavilaCommandGenerator: could not connect to server at "
            f"{self.cfg.server_host}:{self.cfg.server_port} after {retries} attempts"
        )

    def _query(self, frames: list) -> dict:
        if self._sock is None:
            self._connect()

        # Encode frames as base64 JPEG
        encoded = []
        for f in frames:
            buf = io.BytesIO()
            f.save(buf, format="JPEG", quality=85)
            encoded.append(base64.b64encode(buf.getvalue()).decode())

        # Pad to num_frames with black frames if buffer not yet full
        from PIL import Image as _PIL
        while len(encoded) < self.cfg.num_frames:
            buf = io.BytesIO()
            _PIL.new("RGB", (320, 240), (0, 0, 0)).save(buf, format="JPEG")
            encoded.insert(0, base64.b64encode(buf.getvalue()).decode())

        req = json.dumps({
            "frames": encoded,
            "instruction": self.cfg.instruction,
        }) + "\n"

        try:
            self._sock.sendall(req.encode())
            # Read one response line
            while b"\n" not in self._rbuf:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Server closed connection")
                self._rbuf += chunk
            line, self._rbuf = self._rbuf.split(b"\n", 1)
            cmd = json.loads(line.decode())
            self.last_raw = cmd.get("raw", "")
            return cmd
        except Exception as e:
            print(f"[NavilaCommandGenerator] Socket error: {e} — defaulting to STOP")
            self._sock = None   # reconnect next time
            return {"action": "STOP", "raw": ""}

    def __del__(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
