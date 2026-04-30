"""Optimal trajectory solver for quadruped navigation.

Implements the OCP from the paper (Eq. 1-3) using direct collocation
(trapezoidal) on a unicycle-with-acceleration model.

    State:   x = [px, py, v, θ]
    Control: u = [a, ω]   (linear acceleration, angular velocity)

    Dynamics (continuous):
        ṗ_x = v · cos θ
        ṗ_y = v · sin θ
        v̇   = a
        θ̇   = ω

    Cost:
        J = w_t · T  +  ∫₀ᵀ (w_e · ‖u‖² + w_d · v) dt

    Output:
        P* = {(p*_k, v*_k, θ*_k)}_{k=0}^{M}  — waypoints consumed by
        the geometric tracking controller (Section IV-A of the paper).

Usage (standalone):
    python optimal_trajectory_solver.py

Usage (as module):
    from scripts.optimal_trajectory_solver import solve_trajectory, SolverConfig
    traj = solve_trajectory(start=[0, 0, 0], goal=[5, 3])
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import minimize, Bounds


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverConfig:
    # Collocation
    N: int = 60             # number of collocation intervals (M = N waypoints)

    # Cost weights (match Eq. 2)
    w_t: float = 1.0        # time weight
    w_e: float = 0.1        # acceleration smoothness weight  a²  (lowered to allow curves)
    w_d: float = 0.1        # path-length (speed) weight  ‖ṗ‖ = v
    w_curv: float = 0.5     # centripetal energy  ω²·v  (turning at speed costs extra)
    w_accel: float = 0.3    # acceleration energy  |a|·v  (propulsive work ∝ force × speed)

    # Robot limits — must stay consistent with πloco feasible ranges
    v_max:     float = 0.5   # m/s   (Go2 / NaVILA nominal max)
    v_min:     float = 0.0   # forward-only unicycle
    a_max:     float = 0.5   # m/s²
    omega_max: float = 1.0   # rad/s

    # Solver time budget
    T_min: float = 1.0      # s
    T_max: float = 180.0    # s

    # Terminal speed: robot should stop at goal
    stop_at_goal: bool = True

    # scipy SLSQP options
    max_iter: int = 1000
    ftol: float = 1e-7


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReferenceTrajectory:
    """P* from the paper: M+1 waypoints, k=0 is start, k=M is goal.

    Arrays are ready to feed directly into the geometric tracking
    controller (Section IV-A):
        δ_t      = positions[k] - p_t
        ψ_des,t  = atan2(δ_y, δ_x)
        v_x,t    = ‖δ_t‖ · cos(ψ_des,t - θ_t)   clipped to v_max
        ω_z,t    = Kp · (ψ_des,t - θ_t)
    """
    positions: np.ndarray   # (M+1, 2)  world-frame [px, py]
    speeds:    np.ndarray   # (M+1,)    reference speed v*_k  [m/s]
    headings:  np.ndarray   # (M+1,)    reference heading θ*_k [rad]
    times:     np.ndarray   # (M+1,)    time at each waypoint  [s]
    cost:      float = 0.0
    success:   bool = False
    message:   str = ""

    @property
    def M(self) -> int:
        return len(self.times) - 1

    @property
    def total_time(self) -> float:
        return float(self.times[-1])

    def to_dict(self) -> dict:
        """Serialise to plain numpy arrays (for saving / passing to IsaacLab)."""
        return {
            "positions": self.positions,
            "speeds":    self.speeds,
            "headings":  self.headings,
            "times":     self.times,
            "cost":      self.cost,
            "success":   self.success,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_trajectory(
    start: np.ndarray | list,
    goal:  np.ndarray | list,
    cfg:   SolverConfig | None = None,
) -> ReferenceTrajectory:
    """Solve the OCP (Eq. 1-3) and return P*.

    Args:
        start: [px, py, θ]  or  [px, py, v0, θ].
               If only 3 elements, initial speed v0 is assumed 0.
        goal:  [px, py]  or  [px, py, θ_goal].
               If only 2 elements, terminal heading is free.
        cfg:   Solver hyperparameters. Uses defaults if None.

    Returns:
        ReferenceTrajectory. Check `.success` before using.
    """
    if cfg is None:
        cfg = SolverConfig()

    start = np.asarray(start, dtype=float)
    goal  = np.asarray(goal,  dtype=float)

    # Parse start state
    if start.size == 3:
        px0, py0, th0 = start
        v0 = 0.0
    elif start.size >= 4:
        px0, py0, v0, th0 = start[:4]
    else:
        raise ValueError(f"start must have 3 or 4 elements, got {start.size}")

    # Parse goal
    goal_pos = goal[:2]
    goal_heading: Optional[float] = float(goal[2]) if goal.size >= 3 else None

    N  = cfg.N
    nx = 4   # [px, py, v, θ]
    nu = 2   # [a,  ω]

    # ── Index helpers ──────────────────────────────────────────────────────────
    # z = [T,  x_0 … x_N,  u_0 … u_N]
    n_x = nx * (N + 1)
    n_u = nu * (N + 1)
    n_z = 1 + n_x + n_u

    ix = lambda k: 1 + nx * k           # start of x_k in z
    iu = lambda k: 1 + n_x + nu * k     # start of u_k in z

    def unpack(z):
        T  = z[0]
        xs = z[1 : 1 + n_x].reshape(N + 1, nx)
        us = z[1 + n_x :].reshape(N + 1, nu)
        return T, xs, us

    # ── Unicycle + acceleration dynamics ──────────────────────────────────────
    def f_vec(xs, us):
        """Vectorised dynamics:  ẋ = f(x, u).  xs: (K, 4), us: (K, 2)."""
        v, th = xs[:, 2], xs[:, 3]
        a, om = us[:, 0], us[:, 1]
        return np.stack([v * np.cos(th), v * np.sin(th), a, om], axis=1)  # (K, 4)

    # ── Objective ─────────────────────────────────────────────────────────────
    # J = w_t·T + ∫( w_e·a²  +  w_d·v  +  w_curv·ω²·v  +  w_accel·|a|·v ) dt
    #
    #  w_e·a²          — input smoothness (jerk-free acceleration)
    #  w_d·v           — path length  (∫v dt = distance)
    #  w_curv·ω²·v     — centripetal energy: turning at speed is expensive;
    #                    encourages slowing in curves (physically correct for legged robots)
    #  w_accel·|a|·v   — propulsive work ≈ force × speed; penalises hard acceleration
    #                    at high speed more than gentle cruise
    def objective(z):
        T, xs, us = unpack(z)
        h = T / N
        a, om, v = us[:, 0], us[:, 1], xs[:, 2]
        L = (
            cfg.w_e     * a**2
          + cfg.w_d     * v
          + cfg.w_curv  * om**2 * v
          + cfg.w_accel * np.abs(a) * v
        )
        integral = h * (0.5 * L[0] + L[1:-1].sum() + 0.5 * L[-1])
        return cfg.w_t * T + integral

    # ── Collocation residuals ─────────────────────────────────────────────────
    #   x_{k+1} - x_k - h/2·(f_k + f_{k+1}) = 0   for k = 0 … N-1
    def eq_collocation(z):
        T, xs, us = unpack(z)
        h = T / N
        F = f_vec(xs, us)                          # (N+1, 4)
        resid = xs[1:] - xs[:-1] - 0.5 * h * (F[:-1] + F[1:])
        return resid.ravel()                        # 4·N

    # ── Initial state ─────────────────────────────────────────────────────────
    def eq_initial(z):
        _, xs, _ = unpack(z)
        return xs[0] - np.array([px0, py0, v0, th0])

    # ── Terminal constraints ───────────────────────────────────────────────────
    def eq_terminal(z):
        _, xs, _ = unpack(z)
        c = [xs[-1, 0] - goal_pos[0],   # px == goal_px
             xs[-1, 1] - goal_pos[1]]   # py == goal_py
        if cfg.stop_at_goal:
            c.append(xs[-1, 2])          # v_N == 0
        if goal_heading is not None:
            c.append(xs[-1, 3] - goal_heading)
        return np.array(c)

    # ── Variable bounds ────────────────────────────────────────────────────────
    lb = np.full(n_z, -np.inf)
    ub = np.full(n_z,  np.inf)

    lb[0], ub[0] = cfg.T_min, cfg.T_max          # T

    for k in range(N + 1):
        lb[ix(k) + 2] = cfg.v_min                # v ≥ 0
        ub[ix(k) + 2] = cfg.v_max                # v ≤ v_max
        lb[iu(k)]     = -cfg.a_max               # a
        ub[iu(k)]     =  cfg.a_max
        lb[iu(k) + 1] = -cfg.omega_max           # ω
        ub[iu(k) + 1] =  cfg.omega_max

    bounds = Bounds(lb, ub)

    # ── Warm start: straight-line path, trapezoidal speed profile ─────────────
    dist     = np.linalg.norm(goal_pos - np.array([px0, py0]))
    th_guess = np.arctan2(goal_pos[1] - py0, goal_pos[0] - px0)

    # Time to cover distance at 60% of v_max, with margin for turn
    angle_diff = abs(_wrap_angle(th_guess - th0))
    T_guess = max(dist / (0.6 * cfg.v_max) + angle_diff / cfg.omega_max + 2.0,
                  cfg.T_min + 1.0)
    T_guess = min(T_guess, cfg.T_max)

    z0 = np.zeros(n_z)
    z0[0] = T_guess
    alpha = np.linspace(0.0, 1.0, N + 1)

    # Interpolate heading from th0 toward th_guess
    th_interp = th0 + alpha * _wrap_angle(th_guess - th0)

    # Bell-shaped speed profile: 0 → v_max → 0
    v_profile = np.sin(np.pi * alpha) * 0.7 * cfg.v_max

    for k in range(N + 1):
        z0[ix(k)]     = px0 + alpha[k] * (goal_pos[0] - px0)
        z0[ix(k) + 1] = py0 + alpha[k] * (goal_pos[1] - py0)
        z0[ix(k) + 2] = v_profile[k]
        z0[ix(k) + 3] = th_interp[k]

    # Finite-difference accelerations as warm start for controls
    h0 = T_guess / N
    for k in range(N):
        z0[iu(k)]     = (z0[ix(k+1) + 2] - z0[ix(k) + 2]) / h0
        z0[iu(k) + 1] = _wrap_angle(z0[ix(k+1) + 3] - z0[ix(k) + 3]) / h0
    z0[iu(N)] = z0[iu(N-1)]    # repeat last control

    # ── Solve ─────────────────────────────────────────────────────────────────
    constraints = [
        {"type": "eq", "fun": eq_collocation},
        {"type": "eq", "fun": eq_initial},
        {"type": "eq", "fun": eq_terminal},
    ]

    t0 = time.perf_counter()
    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg.max_iter, "ftol": cfg.ftol, "disp": False},
    )
    elapsed = time.perf_counter() - t0

    T_sol, xs_sol, _ = unpack(result.x)
    times = np.linspace(0.0, max(T_sol, 0.0), N + 1)

    msg = f"{result.message} (solve time: {elapsed:.2f}s, T*={T_sol:.2f}s, cost={result.fun:.4f})"

    return ReferenceTrajectory(
        positions=xs_sol[:, :2].copy(),
        speeds=np.clip(xs_sol[:, 2].copy(), 0.0, cfg.v_max),
        headings=xs_sol[:, 3].copy(),
        times=times,
        cost=float(result.fun),
        success=bool(result.success),
        message=msg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Geometric tracking controller  (Section IV-A of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def waypoint_to_velocity_command(
    traj: ReferenceTrajectory,
    robot_pos: np.ndarray,      # [px, py]
    robot_heading: float,       # θ  [rad]
    waypoint_idx: int,
    Kp_yaw: float = 2.0,
    eps_wp: float = 0.15,       # advance-waypoint threshold ε_wp [m]
    v_max: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Convert reference waypoint to (vx, vy=0, ωz) command for πloco.

    Implements the geometric conversion from the paper (Section IV-A):
        δ_t     = p*_k − p_t
        ψ_des,t = atan2(δ_y, δ_x)
        vx      = ‖δ_t‖ · cos(ψ_des,t − θ_t)    clipped to [0, v_max]
        ωz      = Kp · (ψ_des,t − θ_t)

    Returns:
        cmd:          np.ndarray([vx, vy, ωz])  in robot base frame
        next_wp_idx:  updated waypoint index (advanced if within ε_wp)
    """
    k = waypoint_idx
    M = traj.M

    # Advance waypoint if within threshold
    while k < M and np.linalg.norm(traj.positions[k] - robot_pos) < eps_wp:
        k += 1

    delta   = traj.positions[k] - robot_pos
    dist    = np.linalg.norm(delta)
    psi_des = np.arctan2(delta[1], delta[0])
    heading_err = _wrap_angle(psi_des - robot_heading)

    vx = dist * np.cos(heading_err)
    vx = float(np.clip(vx, 0.0, v_max))
    oz = float(np.clip(Kp_yaw * heading_err, -1.0, 1.0))

    return np.array([vx, 0.0, oz]), k


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-π, π]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def resample_trajectory(traj: ReferenceTrajectory, M_new: int) -> ReferenceTrajectory:
    """Resample P* to M_new+1 evenly-spaced waypoints (useful for a fixed-size buffer)."""
    t_new = np.linspace(traj.times[0], traj.times[-1], M_new + 1)
    pos  = np.stack([np.interp(t_new, traj.times, traj.positions[:, i]) for i in range(2)], axis=1)
    spd  = np.interp(t_new, traj.times, traj.speeds)
    hdg  = np.interp(t_new, traj.times, np.unwrap(traj.headings))
    return ReferenceTrajectory(pos, spd, hdg, t_new, traj.cost, traj.success, traj.message)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_trajectory(traj: ReferenceTrajectory, path: str) -> None:
    """Save P* to an .npz file consumable by TrajectoryCommandGenerator."""
    np.savez(path, **{k: v for k, v in traj.to_dict().items() if isinstance(v, np.ndarray)},
             cost=np.array(traj.cost), success=np.array(traj.success))
    print(f"Saved trajectory ({traj.M + 1} waypoints) → {path}")


def load_trajectory(path: str) -> ReferenceTrajectory:
    """Load P* from an .npz file saved by save_trajectory."""
    data = np.load(path, allow_pickle=True)
    return ReferenceTrajectory(
        positions=data["positions"],
        speeds=data["speeds"],
        headings=data["headings"],
        times=data["times"],
        cost=float(data["cost"]),
        success=True,
        message=f"loaded from {path}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / visualisation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimal trajectory solver demo")
    parser.add_argument("--start", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=("PX", "PY", "THETA"), help="Start [px py θ]")
    parser.add_argument("--goal",  nargs="+", type=float, default=[5.0, 3.0],
                        metavar="VAL", help="Goal  [px py] or [px py θ_goal]")
    parser.add_argument("--N", type=int, default=60,  help="Collocation intervals")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="Save P* to this .npz path (consumed by track_trajectory.py)")
    args = parser.parse_args()

    cfg = SolverConfig(N=args.N)

    print(f"Solving OCP: start={args.start} → goal={args.goal}")
    print(f"  N={cfg.N}, v_max={cfg.v_max} m/s, w=(t={cfg.w_t}, e={cfg.w_e}, d={cfg.w_d})")

    traj = solve_trajectory(args.start, args.goal, cfg)

    print(f"\nResult:  success={traj.success}")
    print(f"  {traj.message}")
    print(f"  Waypoints: {traj.M + 1}")
    print(f"  Total time: {traj.total_time:.2f} s")
    print(f"  Optimal cost J*: {traj.cost:.4f}")
    print(f"  Start pos: {traj.positions[0]}")
    print(f"  End   pos: {traj.positions[-1]}  (goal: {args.goal})")
    print(f"  Speed range: [{traj.speeds.min():.3f}, {traj.speeds.max():.3f}] m/s")

    if args.save:
        save_trajectory(traj, args.save)

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # -- XY path
        ax = axes[0]
        ax.plot(traj.positions[:, 0], traj.positions[:, 1], "b-o", ms=2, label="P*")
        ax.plot(*args.start[:2], "gs", ms=8, label="start")
        ax.plot(*args.goal,      "r*", ms=12, label="goal")
        # quiver for heading
        skip = max(1, traj.M // 15)
        ax.quiver(traj.positions[::skip, 0], traj.positions[::skip, 1],
                  np.cos(traj.headings[::skip]), np.sin(traj.headings[::skip]),
                  scale=15, width=0.003, color="steelblue", alpha=0.7)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title("Reference path P*")
        ax.legend()
        ax.grid(True)

        # -- Speed profile
        ax = axes[1]
        ax.plot(traj.times, traj.speeds, "b-")
        ax.axhline(cfg.v_max, ls="--", c="r", label="v_max")
        ax.set_xlabel("t [s]"); ax.set_ylabel("speed [m/s]")
        ax.set_title("Speed profile v*(t)")
        ax.legend(); ax.grid(True)

        # -- Heading profile
        ax = axes[2]
        ax.plot(traj.times, np.degrees(traj.headings), "g-")
        ax.set_xlabel("t [s]"); ax.set_ylabel("heading [deg]")
        ax.set_title("Heading profile θ*(t)")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("trajectory_plan.png", dpi=150)
        print("\nPlot saved to trajectory_plan.png")
        plt.show()
