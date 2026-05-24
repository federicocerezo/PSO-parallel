"""
Wind farm layout optimisation — Jensen (Park) wake model.

Decision variables
------------------
x = [x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}]   shape (2N,)

Each pair (xi, yi) is the position of turbine i in the terrain [m].
Dimension d = 2·N, so d=2 → N=1, d=10 → N=5, d=30 → N=15.

Objective
---------
Minimise the wake-loss fraction:

    f(x) = wake_loss + penalty

where wake_loss ∈ [0, 1] is the fraction of AEP lost to wakes and
penalty is a soft constraint for minimum turbine separation.

Physical model
--------------
Jensen (1983) single-wake model with Katić et al. (1986)
sum-of-squares superposition for multiple wakes.

    Velocity at turbine j due to wake from upstream turbine i
    (Jensen, full overlap):
        u_ij = (1 - sqrt(1 - Ct)) · (D / (D + 2·k·x_ij))²

    where x_ij is the downstream distance from i to j.

    Effective velocity at j (Katić superposition):
        V_j = V∞ · (1 - sqrt( Σ_i  u_ij² · m_ij ))

    where m_ij = A_overlap_ij / A_rotor is the fraction of j's rotor
    disk covered by the wake cone of i (circle–circle intersection).

    AEP ∝ Σ_j V_j³   →   wake_loss = 1 - mean_j[(V_j/V∞)³]

Notes on simplifications
-------------------------
* Ct = 0.8 is held constant (no Ct(V) curve).  A real model would use
  the turbine power curve; this simplification is standard in layout
  optimisation benchmarks (Mosetti 1994, Grady 2005).
* Wind rose: default unidireccional from west (math angle 0 rad, +x).
  The interface accepts wind_directions / wind_weights for a full rose;
  the average wake loss is computed as a weighted sum over directions.

Soft constraint
---------------
Minimum separation ≥ min_sep_D · D enforced as a quadratic penalty:

    penalty = penalty_coeff · Σ_{i<j} max(0, min_sep - dist_ij)²

penalty_coeff = 1e-4 is calibrated so that a 10 m violation
(turbines 10 m closer than the threshold) contributes ~0.01 to the
objective — on the same order as a small wake loss.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class WindFarm:
    """
    Wind farm layout optimisation using the Jensen wake model.

    Parameters
    ----------
    n_turbines : int
        Number of turbines N.  Sets decision-variable dimension d = 2N.
    v_inf : float
        Free-stream wind speed [m/s].
    terrain_size : float
        Side length of the square terrain [m].  Default 2000 m (Mosetti).
    penalty_coeff : float
        Coefficient for the soft minimum-separation penalty.
    min_sep_D : float
        Minimum turbine separation in rotor diameters.
    wind_directions : array-like of float, optional
        Wind directions in degrees, meteorological convention
        (0 = north, 90 = east, clockwise).  None → west (270° met).
    wind_weights : array-like of float, optional
        Frequency weights per direction (normalised internally).
        None → equal weight 1.0 for each direction.
    seed : int, optional
        RNG seed.  No effect for deterministic (unidirectional) wind rose;
        reserved for future stochastic wind-rose generation.

    Examples
    --------
    >>> wf = WindFarm(n_turbines=5)          # d = 10
    >>> x  = np.full(10, 1000.0)
    >>> loss = wf(x)
    """

    # ------------------------------------------------------------------ #
    # Turbine physical parameters (class-level constants)                  #
    # ------------------------------------------------------------------ #
    D: float = 80.0    # Rotor diameter [m]
    H: float = 70.0    # Hub height [m]
    Ct: float = 0.8    # Thrust coefficient — constant (see module docstring)
    k: float = 0.075   # Wake decay coefficient (onshore standard)

    def __init__(
        self,
        n_turbines: int = 5,
        v_inf: float = 12.0,
        terrain_size: float = 2000.0,
        penalty_coeff: float = 1e-4,
        min_sep_D: float = 3.0,
        wind_directions: Optional[NDArray[np.float64]] = None,
        wind_weights: Optional[NDArray[np.float64]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.n_turbines = int(n_turbines)
        self.v_inf = float(v_inf)
        self.terrain_size = float(terrain_size)
        self.penalty_coeff = float(penalty_coeff)
        self.min_sep = float(min_sep_D) * self.D
        self._rng = np.random.default_rng(seed)

        # Derived constants (precomputed once)
        self._r = self.D / 2.0
        self._A_rot = np.pi * self._r ** 2
        # Jensen velocity-deficit coefficient: 1 - sqrt(1 - Ct)
        self._deficit_coeff = 1.0 - np.sqrt(1.0 - self.Ct)

        # Wind rose ----------------------------------------------------- #
        if wind_directions is None:
            # Default: unidirectional from west, math angle 0 rad (+x axis)
            self._dirs_rad = np.array([0.0])
            self._weights = np.array([1.0])
        else:
            dirs = np.asarray(wind_directions, dtype=float)
            # Meteorological → math convention: θ_math = 270 - θ_met (mod 360)
            self._dirs_rad = np.deg2rad(270.0 - dirs) % (2.0 * np.pi)
            w = (np.asarray(wind_weights, dtype=float)
                 if wind_weights is not None else np.ones(len(dirs)))
            self._weights = w / w.sum()

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Box constraints for each decision variable.

        Returns
        -------
        List[Tuple[float, float]]
            List of (xmin, xmax) pairs, one per coordinate.
            All coordinates share the range [0, terrain_size].
        """
        return [(0.0, self.terrain_size)] * (2 * self.n_turbines)

    def __call__(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate wake-loss objective for a given turbine layout.

        Parameters
        ----------
        x : NDArray of shape (2 * n_turbines,)
            Flattened turbine coordinates [x0, y0, x1, y1, ...] in metres.

        Returns
        -------
        float
            Wake-loss fraction ∈ [0, 1] plus soft separation penalty.
            0 means no wake losses and all turbines satisfy the minimum
            separation constraint.
        """
        pos = np.asarray(x, dtype=float).reshape(self.n_turbines, 2)
        return float(self._wake_loss_vec(pos) + self._penalty(pos))

    # ------------------------------------------------------------------ #
    # Vectorised implementation — no Python loops over turbines            #
    # ------------------------------------------------------------------ #

    def _wake_loss_vec(self, pos: NDArray[np.float64]) -> float:
        """
        Weighted wake loss over all wind directions (vectorised).

        Parameters
        ----------
        pos : NDArray of shape (N, 2)
            Turbine positions.

        Returns
        -------
        float
            Expected wake-loss fraction.
        """
        total = 0.0
        for angle, w in zip(self._dirs_rad, self._weights):
            total += float(w) * self._wake_loss_one_dir_vec(pos, float(angle))
        return total

    def _wake_loss_one_dir_vec(
        self,
        pos: NDArray[np.float64],
        angle_rad: float,
    ) -> float:
        """
        Wake loss for a single wind direction — fully vectorised.

        Rotates the coordinate system so wind blows along +x, then
        applies Jensen + Katić using NumPy broadcasting.  No Python
        loop over turbines.

        Parameters
        ----------
        pos : NDArray of shape (N, 2)
        angle_rad : float
            Wind *from* direction in math radians (0 = east, CCW).

        Returns
        -------
        float
        """
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        # Rotate so wind blows along +x
        px = pos[:, 0] * cos_a + pos[:, 1] * sin_a   # downstream coord
        py = -pos[:, 0] * sin_a + pos[:, 1] * cos_a  # lateral coord

        # Pairwise downstream / lateral distances — shape (N, N)
        # x_down[i, j] > 0 means j is downstream of i
        x_down = px[np.newaxis, :] - px[:, np.newaxis]
        y_lat = np.abs(py[np.newaxis, :] - py[:, np.newaxis])

        upstream = x_down > 0.0  # bool mask (N, N)

        # Wake radius at j from upstream turbine i
        r_wake = np.where(upstream, self._r + self.k * x_down, 0.0)

        # Rotor–wake overlap area (vectorised circle intersection)
        overlap = self._circle_overlap_vec(self._r, r_wake, y_lat)

        # Jensen deficit coefficient (i → j, full overlap)
        denom = np.where(upstream, self.D + 2.0 * self.k * x_down, 1.0)
        alpha = np.where(
            upstream,
            self._deficit_coeff * (self.D / denom) ** 2,
            0.0,
        )

        # Katić: squared deficit weighted by fractional overlap
        frac = np.where(upstream, overlap / self._A_rot, 0.0)
        deficit_sq = alpha ** 2 * frac  # (N, N)

        # Sum over upstream turbines i → total squared deficit at each j
        total_def_sq = deficit_sq.sum(axis=0)  # (N,)

        # Effective velocity fraction at each turbine
        vel_frac = 1.0 - np.sqrt(np.clip(total_def_sq, 0.0, 1.0))  # (N,)

        return float(1.0 - np.mean(vel_frac ** 3))

    @staticmethod
    def _circle_overlap_vec(
        r1: float,
        r2: NDArray[np.float64],
        d: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Vectorised intersection area of two circles.

        Parameters
        ----------
        r1 : float
            Radius of circle 1 (rotor radius, scalar).
        r2 : NDArray
            Radii of circle 2 (wake cone cross-sections).
        d : NDArray
            Centre-to-centre distances.

        Returns
        -------
        NDArray
            Intersection areas, same shape as r2.
        """
        no_overlap = d >= (r1 + r2)
        full_contain = (~no_overlap) & (d <= np.abs(r1 - r2))
        partial = ~(no_overlap | full_contain)

        # Full containment: area = π · min(r1, r2)²
        result = np.where(full_contain, np.pi * np.minimum(r1, r2) ** 2, 0.0)

        # Partial overlap via classical two-circle intersection formula.
        # Safe denominators to avoid NaN in non-partial entries.
        d_s = np.where(partial, np.maximum(d, 1e-12), 1.0)
        r2_s = np.where(partial, np.maximum(r2, 1e-12), 1.0)

        d1 = (d_s ** 2 + r1 ** 2 - r2_s ** 2) / (2.0 * d_s)
        d2 = d_s - d1

        a1 = np.arccos(np.clip(d1 / r1, -1.0, 1.0))
        a2 = np.arccos(np.clip(d2 / r2_s, -1.0, 1.0))

        area_p = (
            r1 ** 2 * a1 - d1 * np.sqrt(np.clip(r1 ** 2 - d1 ** 2, 0.0, None))
            + r2_s ** 2 * a2 - d2 * np.sqrt(np.clip(r2_s ** 2 - d2 ** 2, 0.0, None))
        )

        return result + np.where(partial, area_p, 0.0)

    # ------------------------------------------------------------------ #
    # Loop implementation — baseline for V0 vs V4 benchmarking            #
    # ------------------------------------------------------------------ #

    def _evaluate_loop(self, x: NDArray[np.float64]) -> float:
        """
        Sequential loop-based implementation (non-vectorised).

        Kept as a reference baseline so that V0 (SequentialEvaluator +
        _evaluate_loop) can be compared directly against V4
        (NumpyEvaluator + vectorised __call__).  The two should agree to
        floating-point precision.  Not used by __call__; call explicitly
        when benchmarking.

        Parameters
        ----------
        x : NDArray of shape (2 * n_turbines,)

        Returns
        -------
        float
        """
        pos = np.asarray(x, dtype=float).reshape(self.n_turbines, 2)
        total = 0.0
        for angle, w in zip(self._dirs_rad, self._weights):
            total += float(w) * self._wake_loss_one_dir_loop(pos, float(angle))
        return total + float(self._penalty(pos))

    def _wake_loss_one_dir_loop(
        self,
        pos: NDArray[np.float64],
        angle_rad: float,
    ) -> float:
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        px = pos[:, 0] * cos_a + pos[:, 1] * sin_a
        py = -pos[:, 0] * sin_a + pos[:, 1] * cos_a
        N = len(px)

        vel_frac = np.ones(N)
        for j in range(N):
            deficit_sq_sum = 0.0
            for i in range(N):
                dx = px[j] - px[i]
                if dx <= 0.0:
                    continue
                dy = abs(py[j] - py[i])
                r_w = self._r + self.k * dx
                ov = self._circle_overlap_scalar(self._r, r_w, dy)
                if ov <= 0.0:
                    continue
                alpha = self._deficit_coeff * (self.D / (self.D + 2.0 * self.k * dx)) ** 2
                deficit_sq_sum += alpha ** 2 * (ov / self._A_rot)
            vel_frac[j] = 1.0 - np.sqrt(max(deficit_sq_sum, 0.0))

        return float(1.0 - np.mean(vel_frac ** 3))

    @staticmethod
    def _circle_overlap_scalar(r1: float, r2: float, d: float) -> float:
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return np.pi * min(r1, r2) ** 2
        d1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2.0 * d)
        d2 = d - d1
        return float(
            r1 ** 2 * np.arccos(np.clip(d1 / r1, -1.0, 1.0))
            - d1 * np.sqrt(max(r1 ** 2 - d1 ** 2, 0.0))
            + r2 ** 2 * np.arccos(np.clip(d2 / r2, -1.0, 1.0))
            - d2 * np.sqrt(max(r2 ** 2 - d2 ** 2, 0.0))
        )

    # ------------------------------------------------------------------ #
    # Penalty                                                               #
    # ------------------------------------------------------------------ #

    def _penalty(self, pos: NDArray[np.float64]) -> float:
        """
        Quadratic soft penalty for minimum turbine separation.

        Returns
        -------
        float
            penalty_coeff · Σ_{i<j} max(0, min_sep − dist_ij)²
        """
        if self.n_turbines < 2:
            return 0.0
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
        dist = np.sqrt((diff ** 2).sum(axis=2))                # (N, N)
        i_idx, j_idx = np.triu_indices(self.n_turbines, k=1)
        violation = np.maximum(0.0, self.min_sep - dist[i_idx, j_idx])
        return float(self.penalty_coeff * np.sum(violation ** 2))

    def __repr__(self) -> str:
        return (
            f"WindFarm(n_turbines={self.n_turbines}, "
            f"terrain={self.terrain_size}m, "
            f"D={self.D}m, Ct={self.Ct}, k={self.k})"
        )


# Convenience instances for the standard PSO benchmark dimensions
wind_farm_d2 = WindFarm(n_turbines=1)   # d = 2
wind_farm_d10 = WindFarm(n_turbines=5)  # d = 10
wind_farm_d30 = WindFarm(n_turbines=15) # d = 30
