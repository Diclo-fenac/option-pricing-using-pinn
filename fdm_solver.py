"""
Finite Difference Method (FDM) solver for Black-Scholes PDE.
Used as classical baseline for FNO benchmarking.

Solves: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
with boundary condition V(S, T) = max(S - K, 0) for calls.
"""
import numpy as np
import torch
import time


# FDM Solver (Crank-Nicolson)

def fdm_black_scholes(S_grid, t_grid, sigma, r, K, option_type='call'):
    """
    Solve Black-Scholes PDE using Crank-Nicolson finite differences.

    Parameters
    ----------
    S_grid : (n_S,) — asset price grid (must be uniform spacing)
    t_grid : (n_t,) — time grid (must be uniform, increasing from 0 to T)
    sigma : float — volatility
    r : float — risk-free rate
    K : float — strike price
    option_type : str — 'call' or 'put'

    Returns
    -------
    V : (n_S, n_t) — option price surface
    """
    n_S = len(S_grid)
    n_t = len(t_grid)
    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]

    V = np.zeros((n_S, n_t), dtype=np.float64)

    # Terminal condition at t = T (expiry)
    if option_type == 'call':
        V[:, -1] = np.maximum(S_grid - K, 0.0)
    else:
        V[:, -1] = np.maximum(K - S_grid, 0.0)

    # Boundary conditions
    if option_type == 'call':
        V[0, :] = 0.0                          # S=0: V=0
        V[-1, :] = S_grid[-1] - K * np.exp(-r * (t_grid[-1] - t_grid))  # S→∞: V ≈ S - Ke^{-r(T-t)}
    else:
        V[0, :] = K * np.exp(-r * (t_grid[-1] - t_grid))  # S=0: V = Ke^{-r(T-t)}
        V[-1, :] = 0.0                         # S→∞: V=0 for put

    # Crank-Nicolson coefficients (stepping backward in time)
    for n in range(n_t - 2, -1, -1):
        T_minus_t = t_grid[-1] - t_grid[n]
        T_minus_t_next = t_grid[-1] - t_grid[n + 1]

        # Build tridiagonal system: A * V^{n} = B * V^{n+1}
        a = np.zeros(n_S - 2)  # lower diagonal
        b = np.zeros(n_S - 2)  # main diagonal
        c = np.zeros(n_S - 2)  # upper diagonal
        rhs = np.zeros(n_S - 2)  # right-hand side

        for i in range(1, n_S - 1):
            S_i = S_grid[i]
            vol_sq = sigma ** 2

            # Coefficients
            alpha = 0.25 * dt * (vol_sq * i**2 - r * i)
            beta = -0.5 * dt * (vol_sq * i**2 + r)
            gamma = 0.25 * dt * (vol_sq * i**2 + r * i)

            a[i - 1] = -alpha
            b[i - 1] = 1.0 - beta
            c[i - 1] = -gamma

            # Right-hand side
            if i == 1:
                rhs[i - 1] = (alpha * V[0, n + 1] + (1.0 + beta) * V[i, n + 1] + gamma * V[i + 1, n + 1])
            elif i == n_S - 2:
                rhs[i - 1] = (alpha * V[i - 1, n + 1] + (1.0 + beta) * V[i, n + 1] + gamma * V[-1, n + 1])
            else:
                rhs[i - 1] = (alpha * V[i - 1, n + 1] + (1.0 + beta) * V[i, n + 1] + gamma * V[i + 1, n + 1])

        # Solve tridiagonal system
        V[1:-1, n] = _solve_tridiagonal(a, b, c, rhs)

    return V.astype(np.float32)


def _solve_tridiagonal(a, b, c, d):
    """
    Solve a tridiagonal system Ax = d using the Thomas algorithm.

    Parameters
    ----------
    a : (n-1,) — lower diagonal
    b : (n,) — main diagonal
    c : (n-1,) — upper diagonal
    d : (n,) — right-hand side

    Returns
    -------
    x : (n,) — solution
    """
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) / (b[n - 1] - a[n - 2] * c_prime[n - 2])

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def fdm_black_scholes_batch(S_grid, t_grid, params, option_type='call'):
    """
    Batch FDM solver for multiple parameter sets.

    Parameters
    ----------
    S_grid : (n_S,)
    t_grid : (n_t,)
    params : (N, 4) — [sigma, r, K, T] columns
    option_type : str

    Returns
    -------
    V : (N, n_S, n_t)
    """
    n = len(params)
    n_S, n_t = len(S_grid), len(t_grid)
    V = np.zeros((n, n_S, n_t), dtype=np.float32)

    # Ensure uniform grids (FDM requires uniform spacing)
    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]

    for i in range(n):
        sigma, r, K, T = params[i]

        # Create uniform sub-grid for this T
        t_uniform = np.linspace(0, T, n_t, dtype=np.float64)
        S_uniform = np.linspace(S_grid[0], S_grid[-1], n_S, dtype=np.float64)

        V[i] = fdm_black_scholes(S_uniform, t_uniform, sigma, r, K, option_type)

    return V


def fdm_timing(S_grid, t_grid, params, option_type='call'):
    """
    Time the FDM solver on a batch of parameters.

    Returns
    -------
    V : (N, n_S, n_t)
    elapsed : float — total time in seconds
    per_sample : float — time per sample in seconds
    """
    t0 = time.perf_counter()
    V = fdm_black_scholes_batch(S_grid, t_grid, params, option_type)
    elapsed = time.perf_counter() - t0
    per_sample = elapsed / len(params)
    return V, elapsed, per_sample


# FDM with interpolation to non-uniform grids

def fdm_interpolate(V_fdm_uniform, S_uniform, t_uniform, S_target, t_target):
    """
    Bilinear interpolation from FDM uniform grid to target (possibly non-uniform) grid.

    Parameters
    ----------
    V_fdm_uniform : (n_S, n_t) — solution on uniform grid
    S_uniform : (n_S,)
    t_uniform : (n_t,)
    S_target : (n_S',)
    t_target : (n_t',)

    Returns
    -------
    V_target : (n_S', n_t')
    """
    from scipy.interpolate import RegularGridInterpolator

    interp_func = RegularGridInterpolator(
        (S_uniform, t_uniform), V_fdm_uniform,
        method='linear', bounds_error=False, fill_value=None
    )

    S_mesh, t_mesh = np.meshgrid(S_target, t_target, indexing='ij')
    points = np.stack([S_mesh.ravel(), t_mesh.ravel()], axis=-1)
    V_target = interp_func(points).reshape(len(S_target), len(t_target))

    return V_target.astype(np.float32)


# PyTorch FDM wrapper for GPU comparison

class FDMSolver:
    """
    Thin wrapper that provides a consistent interface for FDM solving.
    All computation is done in NumPy (CPU-bound).
    """

    def __init__(self, S_grid, t_grid, option_type='call'):
        """
        Parameters
        ----------
        S_grid : (n_S,) — must be uniform
        t_grid : (n_t,) — must be uniform
        """
        self.S_grid = S_grid
        self.t_grid = t_grid
        self.option_type = option_type

        # Verify uniform spacing
        dS = np.diff(S_grid)
        dt = np.diff(t_grid)
        if np.std(dS) > 1e-10 or np.std(dt) > 1e-10:
            raise ValueError("FDM requires uniform grid spacing. Use uniform grids.")

    def solve(self, sigma, r, K, T=None):
        """
        Solve for a single parameter set.

        Parameters
        ----------
        sigma : float
        r : float
        K : float
        T : float (unused — t_grid already defines the time domain)

        Returns
        -------
        V : (n_S, n_t)
        """
        return fdm_black_scholes(self.S_grid, self.t_grid, sigma, r, K, self.option_type)

    def solve_batch(self, params):
        """
        Solve for a batch of parameters.

        Parameters
        ----------
        params : (N, 4) — [sigma, r, K, T]

        Returns
        -------
        V : (N, n_S, n_t)
        """
        return fdm_black_scholes_batch(self.S_grid, self.t_grid, params, self.option_type)

    def solve_and_time(self, params):
        """
        Solve and measure time.

        Returns
        -------
        V : (N, n_S, n_t)
        elapsed : float
        per_sample : float
        """
        return fdm_timing(self.S_grid, self.t_grid, params, self.option_type)
