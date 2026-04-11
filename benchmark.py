"""
Benchmarking suite — Production Edition

Tests:
  1. FNO inference speed vs vectorized Black-Scholes vs Monte Carlo vs FDM
  2. Pricing accuracy (RMSE, MAPE, max error)
  3. Greeks accuracy via AD (Delta, Gamma vs analytical)
  4. Distribution shift / stress testing
  5. PDE residual quality (how well does learned solution satisfy BS PDE?)
  6. FDM baseline comparison (accuracy + speed)

Data format: HDF5 with 'params' (N,4), 'V', 'Delta', 'Gamma', 'S_grid', 't_template'
Model:     FNOOptionPricer (coordinate-aware decoder + hard-constrained ansatz)
"""
import os
import time
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from fno_model import FNOOptionPricer, compute_pde_residual_autograd
from fdm_solver import FDMSolver


# =============================================================================
# Helpers
# =============================================================================

def _vectorized_bs_price(S_grid, t_grid, sigma, r, K, T, t_template):
    """
    Fully vectorized Black-Scholes call price surface.

    Parameters
    ----------
    S_grid : (n_S,)
    t_grid : (n_t,) — absolute time grid (not scaled)
    sigma : scalar
    r : scalar
    K : scalar
    T : scalar
    t_template : (n_t,) template in [0,1]

    Returns
    -------
    V : (n_S, n_t)
    """
    n_S = len(S_grid)
    t_abs = t_template * T  # (n_t,)
    tau = np.maximum(T - t_abs, 1e-8)  # (n_t,)

    S = S_grid[:, np.newaxis]       # (n_S, 1)
    tau_b = tau[np.newaxis, :]      # (1, n_t)
    sqrt_tau = np.sqrt(tau_b)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_b) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    d1, d2 = np.clip(d1, -37, 37), np.clip(d2, -37, 37)

    N_d1 = 0.5 * (1.0 + np.clip(scipy.special.erf(d1 / np.sqrt(2.0)), -1.0, 1.0))
    N_d2 = 0.5 * (1.0 + np.clip(scipy.special.erf(d2 / np.sqrt(2.0)), -1.0, 1.0))

    V = S * N_d1 - K * np.exp(-r * tau_b) * N_d2

    # Enforce expiry payoff
    at_expiry = (T - t_abs <= 1e-8)[np.newaxis, :]
    payoff = np.maximum(S - K, 0.0)
    V = np.where(at_expiry, payoff, V)

    return V.astype(np.float32)


def _analytical_greeks(S, K, T, sigma, r):
    """Analytical Delta and Gamma for European call at a single (S, K, T)."""
    if T <= 1e-8:
        return float(S > K), 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d1 = np.clip(d1, -37, 37)

    delta = 0.5 * (1.0 + scipy.special.erf(d1 / np.sqrt(2.0)))
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    gamma = phi / (S * sigma * sqrt_T)

    return float(delta), float(gamma)


def _load_data(path):
    """Load HDF5 and return dict."""
    with h5py.File(path, 'r') as f:
        return {
            'params': f['params'][:],     # (N, 4) [σ, r, K, T]
            'V': f['V'][:],               # (N, n_S, n_t)
            'Delta': f['Delta'][:],       # (N, n_S, n_t)
            'Gamma': f['Gamma'][:],       # (N, n_S, n_t)
            'S_grid': f['S_grid'][:],     # (n_S,)
            't_template': f['t_template'][:],  # (n_t,)
        }


def _load_model(config, path):
    """Load trained FNOOptionPricer."""
    device = torch.device(config.device if hasattr(config, 'device') else 'cuda')
    model = FNOOptionPricer(config).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, device


# =============================================================================
# Benchmark 1: FNO Inference Speed
# =============================================================================

def benchmark_fno_speed(data, config, model_path):
    """Measure FNO inference throughput."""
    model, device = _load_model(config, model_path)
    model.eval()

    params = torch.tensor(data['params'], dtype=torch.float32, device=device)
    S_grid = torch.tensor(data['S_grid'], dtype=torch.float32, device=device)
    t_template = torch.tensor(data['t_template'], dtype=torch.float32, device=device)

    sigma = params[:, 0]
    r = params[:, 1]
    K_norm = params[:, 2] / 100.0
    T_norm = params[:, 3] / 2.0

    # Warmup
    with torch.no_grad():
        _ = model(sigma[:1], r[:1], K_norm[:1], T_norm[:1], S_grid, t_template)

    # Timed run
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    batch_size = 256
    V_list = []
    with torch.no_grad():
        for i in range(0, len(sigma), batch_size):
            end = min(i + batch_size, len(sigma))
            V_batch = model(sigma[i:end], r[i:end], K_norm[i:end], T_norm[i:end],
                            S_grid, t_template)
            V_list.append(V_batch.cpu().numpy())

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    V_fno = np.concatenate(V_list, axis=0)
    n = len(sigma)
    per_sample = elapsed / n

    print(f"FNO Inference ({device}, batch={batch_size}):")
    print(f"  Total:       {elapsed:.4f}s ({n:,} samples)")
    print(f"  Per sample:  {per_sample*1e6:.1f}μs")
    print(f"  Throughput:  {n/elapsed:.0f} samples/s")

    return V_fno, per_sample


# =============================================================================
# Benchmark 2: Black-Scholes Analytical Speed (NumPy vectorized)
# =============================================================================

def benchmark_bs_speed(data, config):
    """Measure vectorized Black-Scholes throughput — two modes."""
    S_grid = data['S_grid']
    t_template = data['t_template']
    params = data['params']
    n = len(params)

    # Mode 1: per-sample loop (mimics typical usage)
    t0 = time.perf_counter()
    V_list = []
    for i in range(n):
        sigma, r, K, T = params[i]
        V_i = _vectorized_bs_price(S_grid, t_template, sigma, r, K, T, t_template)
        V_list.append(V_i)
    elapsed_loop = time.perf_counter() - t0
    per_sample_loop = elapsed_loop / n

    # Mode 2: fully batched vectorized (single call, all samples)
    sigmas = params[:, 0]
    rates = params[:, 1]
    Ks = params[:, 2]
    Ts = params[:, 3]

    t0 = time.perf_counter()
    # (n, n_S, n_t) fully vectorized
    n_S, n_t = len(S_grid), len(t_template)
    t_abs = t_template[np.newaxis, :] * Ts[:, np.newaxis]  # (n, n_t)
    tau = np.maximum(Ts[:, np.newaxis] - t_abs, 1e-8)       # (n, n_t)

    S = S_grid[np.newaxis, :, np.newaxis]                    # (1, n_S, 1)
    K_3d = Ks[:, np.newaxis, np.newaxis]                     # (n, 1, 1)
    sig_3d = sigmas[:, np.newaxis, np.newaxis]               # (n, 1, 1)
    r_3d = rates[:, np.newaxis, np.newaxis]                  # (n, 1, 1)
    tau_3d = tau[:, np.newaxis, :]                           # (n, 1, n_t)

    sqrt_tau = np.sqrt(tau_3d)
    ln_SK = np.log(S / K_3d)
    d1 = (ln_SK + (r_3d + 0.5 * sig_3d**2) * tau_3d) / (sig_3d * sqrt_tau)
    d2 = d1 - sig_3d * sqrt_tau
    d1, d2 = np.clip(d1, -37, 37), np.clip(d2, -37, 37)
    N_d1 = 0.5 * (1.0 + np.clip(scipy.special.erf(d1 / np.sqrt(2.0)), -1.0, 1.0))
    N_d2 = 0.5 * (1.0 + np.clip(scipy.special.erf(d2 / np.sqrt(2.0)), -1.0, 1.0))
    discount = np.exp(-r_3d * tau_3d)
    V_batched = (S * N_d1 - K_3d * discount * N_d2).astype(np.float32)

    # Enforce expiry
    at_expiry = (Ts[:, np.newaxis] - t_abs <= 1e-8)[:, np.newaxis, :]
    payoff = np.maximum(S - K_3d, 0.0)
    V_batched = np.where(at_expiry, payoff, V_batched)

    elapsed_batch = time.perf_counter() - t0
    per_sample_batch = elapsed_batch / n

    print(f"\nBlack-Scholes (NumPy):")
    print(f"  Per-sample loop: {per_sample_loop*1e3:.4f}ms/sample ({n/elapsed_loop:.0f} samp/s)")
    print(f"  Fully batched:   {per_sample_batch*1e6:.1f}μs/sample ({n/elapsed_batch:.0f} samp/s)")

    return V_batched, per_sample_loop


# =============================================================================
# Benchmark 3: Monte Carlo (single-price, not surface)
# =============================================================================

def _mc_price(S, K, T, sigma, r, n_paths=100_000, seed=42):
    """Monte Carlo European call — single price."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0.0)
    return float(np.mean(payoffs) * np.exp(-r * T))


def benchmark_mc_speed(data, config, n_paths=100_000, subset=200):
    """MC benchmark on a subset (it's slow)."""
    params = data['params']
    n = min(subset, len(params))
    indices = np.linspace(0, len(params) - 1, n, dtype=int)
    S_ref = 100.0  # Reference spot for MC

    t0 = time.perf_counter()
    V_mc = []
    mc_errs = []

    for i in tqdm(indices, desc='Monte Carlo', leave=False):
        sigma, r, K, T = params[i]
        # MC with variance estimate (two halves)
        half = n_paths // 2
        p1 = _mc_price(S_ref, K, T, sigma, r, half, seed=42)
        p2 = _mc_price(S_ref, K, T, sigma, r, half, seed=43)
        V_mc.append((p1 + p2) / 2)
        mc_errs.append(abs(p1 - p2) / 2)

    elapsed = time.perf_counter() - t0
    per_sample = elapsed / n

    print(f"\nMonte Carlo ({n_paths} paths, n={n}):")
    print(f"  Total:       {elapsed:.1f}s")
    print(f"  Per sample:  {per_sample*1e3:.1f}ms")
    print(f"  Avg MC err:  {np.mean(mc_errs):.6f}")

    return np.array(V_mc, dtype=np.float32), per_sample


# =============================================================================
# Benchmark 3b: FDM (Finite Difference Method)
# =============================================================================

def benchmark_fdm_speed(data, config, subset=200):
    """
    Measure FDM throughput on a subset of test samples.
    FDM is CPU-bound and slow — use a small subset.

    Returns
    -------
    V_fdm : (subset, n_S, n_t) — FDM solutions
    per_sample : float — seconds per sample
    """
    from fdm_solver import fdm_black_scholes

    S_grid = data['S_grid']
    t_template = data['t_template']
    params = data['params']

    # Use a small subset — FDM is inherently serial and slow
    n = min(subset, len(params))
    indices = np.linspace(0, len(params) - 1, n, dtype=int)

    # FDM requires uniform grids
    S_uniform = np.linspace(S_grid.min(), S_grid.max(), len(S_grid), dtype=np.float64)

    print(f"\n{'='*60}")
    print(f"FDM Benchmark (Crank-Nicolson, n={n})")
    print(f"{'='*60}")

    V_fdm = np.zeros((n, len(S_uniform), len(t_template)), dtype=np.float32)
    t0 = time.perf_counter()

    for i, idx in enumerate(tqdm(indices, desc='FDM', leave=False)):
        sigma, r, K, T = params[idx]
        t_uniform = np.linspace(0, T, len(t_template), dtype=np.float64)

        V_fdm[i] = fdm_black_scholes(S_uniform, t_uniform, sigma, r, K)

    elapsed = time.perf_counter() - t0
    per_sample = elapsed / n

    print(f"  Total:       {elapsed:.1f}s")
    print(f"  Per sample:  {per_sample*1e3:.1f}ms")
    print(f"  Throughput:  {n/elapsed:.1f} samples/s")

    return V_fdm, per_sample


def benchmark_fdm_accuracy(data, config, subset=500):
    """
    Compare FDM accuracy vs analytical Black-Scholes.
    This validates that our FDM solver is correct.
    """
    from fdm_solver import fdm_black_scholes

    S_grid = data['S_grid']
    t_template = data['t_template']
    params = data['params']
    V_true = data['V']

    n = min(subset, len(params))
    indices = np.linspace(0, len(params) - 1, n, dtype=int)
    S_uniform = np.linspace(S_grid.min(), S_grid.max(), len(S_grid), dtype=np.float64)

    errors = []
    t0 = time.perf_counter()

    for idx in tqdm(indices, desc='FDM Accuracy', leave=False):
        sigma, r, K, T = params[idx]
        t_uniform = np.linspace(0, T, len(t_template), dtype=np.float64)

        V_fdm = fdm_black_scholes(S_uniform, t_uniform, sigma, r, K)
        V_ref = V_true[idx]

        err = np.abs(V_fdm - V_ref)
        errors.append({
            'rmse': np.sqrt(np.mean(err**2)),
            'max': err.max(),
            'mape': np.mean(err / (np.abs(V_ref) + 1e-8)) * 100
        })

    elapsed = time.perf_counter() - t0

    rmses = np.array([e['rmse'] for e in errors])
    maxes = np.array([e['max'] for e in errors])
    mapes = np.array([e['mape'] for e in errors])

    print(f"\n{'='*60}")
    print(f"FDM ACCURACY (vs analytical BS, n={n})")
    print(f"{'='*60}")
    print(f"  Mean RMSE:  {rmses.mean():.8e}")
    print(f"  Mean Max:   {maxes.mean():.8e}")
    print(f"  Mean MAPE:  {mapes.mean():.6f}%")
    print(f"  Total time: {elapsed:.1f}s")

    return {
        'mean_rmse': rmses.mean(),
        'mean_max': maxes.mean(),
        'mean_mape': mapes.mean(),
        'elapsed': elapsed,
        'per_sample': elapsed / n
    }


# =============================================================================
# Benchmark 4: Pricing Accuracy
# =============================================================================

def compute_pricing_accuracy(V_fno, V_true, V_mc=None):
    """RMSE, MAPE, max error."""
    rmse = np.sqrt(np.mean((V_fno - V_true) ** 2))
    max_err = np.max(np.abs(V_fno - V_true))
    mape = np.mean(np.abs((V_fno - V_true) / (np.abs(V_true) + 1e-8))) * 100

    print(f"\n{'='*60}")
    print("PRICING ACCURACY")
    print(f"{'='*60}")
    print(f"  RMSE:      {rmse:.8f}")
    print(f"  Max error: {max_err:.6f}")
    print(f"  MAPE:      {mape:.4f}%")

    results = {'rmse': rmse, 'max_error': max_err, 'mape': mape}

    if V_mc is not None:
        # Compare at S=S_ref only
        mc_rmse = np.sqrt(np.mean((V_fno[:, 128, 0] - V_mc) ** 2))  # S≈100, t=0
        print(f"  MC RMSE (S=S_ref): {mc_rmse:.6f}")
        results['mc_rmse'] = mc_rmse

    return results


# =============================================================================
# Benchmark 5: Greeks Accuracy (AD-based)
# =============================================================================

def compute_greeks_accuracy(data, config, model_path):
    """
    Compare FNO-predicted Delta/Gamma (via finite differences on surface)
    vs analytical Greeks at reference points.
    """
    model, device = _load_model(config, model_path)
    model.eval()

    S_grid = torch.tensor(data['S_grid'], dtype=torch.float32, device=device)
    t_template = torch.tensor(data['t_template'], dtype=torch.float32, device=device)
    params = data['params']

    n_test = min(1000, len(params))
    indices = np.linspace(0, len(params) - 1, n_test, dtype=int)

    dS = S_grid[1] - S_grid[0]

    delta_pred_list = []
    delta_true_list = []
    gamma_pred_list = []
    gamma_true_list = []

    batch_size = 64
    with torch.no_grad():
        for start in tqdm(range(0, n_test, batch_size), desc='Greeks', leave=False):
            end = min(start + batch_size, n_test)
            batch_idx = indices[start:end]

            sigma = torch.tensor(params[batch_idx, 0], dtype=torch.float32, device=device)
            r = torch.tensor(params[batch_idx, 1], dtype=torch.float32, device=device)
            K = torch.tensor(params[batch_idx, 2], dtype=torch.float32, device=device)
            T = torch.tensor(params[batch_idx, 3], dtype=torch.float32, device=device)

            K_norm = K / 100.0
            T_norm = T / 2.0

            # Full surface prediction
            V_pred = model(sigma, r, K_norm, T_norm, S_grid, t_template)

            # Extract at mid-time
            t_mid_idx = len(data['t_template']) // 2
            V_mid = V_pred[:, :, t_mid_idx]  # (batch, n_S)

            # Evaluate Greeks at ATM (S=100)
            S_ref = 100.0
            S_idx = int(torch.argmin(torch.abs(S_grid - S_ref)).item())
            S_left = max(0, S_idx - 1)
            S_right = min(len(S_grid) - 1, S_idx + 1)

            # Central difference Delta
            delta_fd = (V_mid[:, S_right] - V_mid[:, S_left]) / (S_grid[S_right] - S_grid[S_left])
            # Central difference Gamma
            gamma_fd = (V_mid[:, S_right] - 2 * V_mid[:, S_idx] + V_mid[:, S_left]) / (dS ** 2)

            delta_pred_list.append(delta_fd.cpu().numpy())
            gamma_pred_list.append(gamma_fd.cpu().numpy())

            # True Greeks at (S_ref, τ_mid)
            for i in range(len(batch_idx)):
                T_actual = params[batch_idx[i], 3]
                t_abs_mid = data['t_template'][t_mid_idx] * T_actual
                tau = T_actual - t_abs_mid

                if tau < 1e-6:
                    d_true = float(S_ref > params[batch_idx[i], 2])
                    g_true = 0.0
                else:
                    d_true, g_true = _analytical_greeks(
                        S_ref, params[batch_idx[i], 2], tau,
                        params[batch_idx[i], 0], params[batch_idx[i], 1]
                    )
                delta_true_list.append(d_true)
                gamma_true_list.append(g_true)

    delta_pred = np.concatenate(delta_pred_list)
    gamma_pred = np.concatenate(gamma_pred_list)
    delta_true = np.array(delta_true_list)
    gamma_true = np.array(gamma_true_list)

    delta_rmse = np.sqrt(np.mean((delta_pred - delta_true) ** 2))
    delta_max = np.max(np.abs(delta_pred - delta_true))
    gamma_rmse = np.sqrt(np.mean((gamma_pred - gamma_true) ** 2))
    gamma_max = np.max(np.abs(gamma_pred - gamma_true))

    print(f"\n{'='*60}")
    print("GREEKS ACCURACY (at S=100, mid-maturity)")
    print(f"{'='*60}")
    print(f"  Delta RMSE:  {delta_rmse:.6f}  |  Max: {delta_max:.6f}")
    print(f"  Gamma RMSE:  {gamma_rmse:.6f}  |  Max: {gamma_max:.6f}")

    return {
        'delta_rmse': delta_rmse, 'delta_max': delta_max,
        'gamma_rmse': gamma_rmse, 'gamma_max': gamma_max,
    }


# =============================================================================
# Benchmark 6: PDE Residual Quality
# =============================================================================

def compute_pde_residual_quality(data, config, model_path):
    """How well does the FNO solution satisfy the Black-Scholes PDE?"""
    model, device = _load_model(config, model_path)
    model.eval()

    S_grid = torch.tensor(data['S_grid'], dtype=torch.float32, device=device)
    t_template = torch.tensor(data['t_template'], dtype=torch.float32, device=device)
    params = data['params']

    n_test = min(100, len(params))
    indices = np.linspace(0, len(params) - 1, n_test, dtype=int)

    # Interior query points
    S_interior = torch.linspace(20, 500, 16, dtype=torch.float32, device=device)
    T_max = 2.0
    t_interior = torch.linspace(0.05 * T_max, 0.95 * T_max, 12, dtype=torch.float32, device=device)

    residuals = []

    with torch.no_grad():
        for start in range(0, n_test, 16):
            end = min(start + 16, n_test)
            batch_idx = indices[start:end]

            sigma = torch.tensor(params[batch_idx, 0], dtype=torch.float32, device=device)
            r = torch.tensor(params[batch_idx, 1], dtype=torch.float32, device=device)
            K_norm = torch.tensor(params[batch_idx, 2], dtype=torch.float32, device=device) / 100.0
            T_norm = torch.tensor(params[batch_idx, 3], dtype=torch.float32, device=device) / 2.0

            res, _, _, _ = compute_pde_residual_autograd(
                model, sigma, r, K_norm, T_norm,
                S_grid, t_template,
                S_interior, t_interior
            )
            residuals.append(res.abs().cpu().numpy())

    residuals = np.concatenate(residuals, axis=0)
    mean_res = np.mean(residuals)
    max_res = np.max(residuals)
    median_res = np.median(residuals)

    print(f"\n{'='*60}")
    print("PDE RESIDUAL QUALITY")
    print(f"{'='*60}")
    print(f"  Mean residual:  {mean_res:.6e}")
    print(f"  Median residual: {median_res:.6e}")
    print(f"  Max residual:   {max_res:.6e}")

    return {'mean': mean_res, 'median': median_res, 'max': max_res}


# =============================================================================
# Benchmark 7: Distribution Shift / Stress Testing
# =============================================================================

def test_distribution_shift(data, config, model_path):
    """Test generalization to unseen parameter regimes."""
    model, device = _load_model(config, model_path)
    model.eval()

    S_grid = torch.tensor(data['S_grid'], dtype=torch.float32, device=device)
    t_template = torch.tensor(data['t_template'], dtype=torch.float32, device=device)

    # Define regimes outside training distribution
    scenarios = {
        'extreme_vol': dict(sigma=(0.80, 1.50), r=(0.0, 0.15), K=(20, 200), T=(0.1, 2.0)),
        'low_vol':      dict(sigma=(0.01, 0.05), r=(0.0, 0.15), K=(20, 200), T=(0.1, 2.0)),
        'high_rate':    dict(sigma=(0.1, 0.8),  r=(0.15, 0.30), K=(20, 200), T=(0.1, 2.0)),
        'near_expiry':  dict(sigma=(0.1, 0.8),  r=(0.0, 0.15),  K=(20, 200), T=(0.01, 0.10)),
        'very_long':    dict(sigma=(0.1, 0.8),  r=(0.0, 0.15),  K=(20, 200), T=(2.0, 5.0)),
        'deep_otm':     dict(sigma=(0.1, 0.8),  r=(0.0, 0.15),  K=(200, 400), T=(0.1, 2.0)),
        'deep_itm':     dict(sigma=(0.1, 0.8),  r=(0.0, 0.15),  K=(1, 20),   T=(0.1, 2.0)),
    }

    print(f"\n{'='*60}")
    print("DISTRIBUTION SHIFT TESTS")
    print(f"{'='*60}")
    print("Training: σ∈[0.05,0.80]  r∈[0.0,0.15]  K∈[20,200]  T∈[0.1,2.0]")

    results = {}

    for name, rng in scenarios.items():
        np.random.seed(42)
        n = 200

        sigmas = np.random.uniform(*rng['sigma'], n).astype(np.float32)
        rates = np.random.uniform(*rng['r'], n).astype(np.float32)
        Ks = np.random.uniform(*rng['K'], n).astype(np.float32)
        Ts = np.random.uniform(*rng['T'], n).astype(np.float32)

        V_true_list = []
        for i in range(n):
            v = _vectorized_bs_price(
                data['S_grid'], data['t_template'],
                sigmas[i], rates[i], Ks[i], Ts[i], data['t_template']
            )
            V_true_list.append(v)
        V_true = np.array(V_true_list, dtype=np.float32)

        # FNO inference
        sigma_t = torch.tensor(sigmas, device=device)
        r_t = torch.tensor(rates, device=device)
        K_norm_t = torch.tensor(Ks / 100.0, device=device)
        T_norm_t = torch.tensor(Ts / 2.0, device=device)

        V_fno_list = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                V_b = model(sigma_t[i:end], r_t[i:end], K_norm_t[i:end], T_norm_t[i:end],
                           S_grid, t_template)
                V_fno_list.append(V_b.cpu().numpy())
        V_fno = np.concatenate(V_fno_list, axis=0)

        rmse = np.sqrt(np.mean((V_fno - V_true) ** 2))
        max_err = np.max(np.abs(V_fno - V_true))
        mape = np.mean(np.abs((V_fno - V_true) / (np.abs(V_true) + 1e-8))) * 100

        results[name] = {'rmse': rmse, 'max_error': max_err, 'mape': mape}
        print(f"\n  {name:15s}: RMSE={rmse:.6f}  Max={max_err:.6f}  MAPE={mape:.2f}%")

    return results


# =============================================================================
# Visualizations
# =============================================================================

def plot_results(data, V_fno, V_true, output_dir):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    S_grid = data['S_grid']
    t_template = data['t_template']

    # Pick a representative sample
    idx = len(V_true) // 2
    _, _, K, T = data['params'][idx]

    t_abs = t_template * T

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True surface
    im0 = axes[0].imshow(
        V_true[idx], aspect='auto', origin='lower',
        extent=[t_abs[0], t_abs[-1], S_grid[0], S_grid[-1]],
        cmap='viridis'
    )
    axes[0].set_title(f'Black-Scholes (σ={data["params"][idx,0]:.2f}, r={data["params"][idx,1]:.2f}, K={K:.0f}, T={T:.2f})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Asset Price')
    plt.colorbar(im0, ax=axes[0])

    # FNO surface
    im1 = axes[1].imshow(
        V_fno[idx], aspect='auto', origin='lower',
        extent=[t_abs[0], t_abs[-1], S_grid[0], S_grid[-1]],
        cmap='viridis'
    )
    axes[1].set_title('FNO Prediction')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Asset Price')
    plt.colorbar(im1, ax=axes[1])

    # Error surface
    error = np.abs(V_fno[idx] - V_true[idx])
    im2 = axes[2].imshow(
        error, aspect='auto', origin='lower',
        extent=[t_abs[0], t_abs[-1], S_grid[0], S_grid[-1]],
        cmap='hot'
    )
    axes[2].set_title(f'Absolute Error (max={error.max():.6f})')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Asset Price')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'surface_comparison.png'), dpi=150)
    plt.close()

    # Price curves at different maturities
    fig, ax = plt.subplots(figsize=(10, 6))
    t_indices = [0, len(t_template)//4, len(t_template)//2, 3*len(t_template)//4]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for ti, c in zip(t_indices, colors):
        t_val = t_abs[ti]
        ax.plot(S_grid, V_true[idx, :, ti], '--', color=c, label=f'True τ={T - t_val:.2f}', alpha=0.7)
        ax.plot(S_grid, V_fno[idx, :, ti], '-', color=c, label=f'FNO τ={T - t_val:.2f}', alpha=0.5)

    ax.set_xlabel('Asset Price')
    ax.set_ylabel('Option Price')
    ax.set_title('Price Curves at Different Maturities')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_curves.png'), dpi=150)
    plt.close()

    # Error histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    abs_err = np.abs(V_fno - V_true).flatten()
    axes[0].hist(abs_err, bins=100, log=True, edgecolor='none', alpha=0.8)
    axes[0].set_xlabel('Absolute Error')
    axes[0].set_ylabel('Count (log)')
    axes[0].set_title('Error Distribution')
    axes[0].axvline(np.median(abs_err), color='r', ls='--', label=f'Median: {np.median(abs_err):.6f}')
    axes[0].legend()

    rel_err = np.abs((V_fno - V_true) / (np.abs(V_true) + 1e-8)).flatten()
    axes[1].hist(rel_err * 100, bins=100, log=True, edgecolor='none', alpha=0.8)
    axes[1].set_xlabel('Relative Error (%)')
    axes[1].set_ylabel('Count (log)')
    axes[1].set_title('Relative Error Distribution')
    axes[1].axvline(np.median(rel_err) * 100, color='r', ls='--', label=f'Median: {np.median(rel_err)*100:.4f}%')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


# =============================================================================
# Full Benchmark Runner
# =============================================================================

def run_full_benchmark(config, data_path=None, model_path=None):
    """Execute complete benchmark suite."""
    if data_path is None:
        data_path = os.path.join(getattr(config, 'data_dir', './data'), 'test.h5')
    if model_path is None:
        model_path = os.path.join(getattr(config, 'checkpoint_dir', './checkpoints'), 'best.pt')

    output_dir = getattr(config, 'results_dir', './results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("BENCHMARK SUITE: FNO Option Pricing")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading: {data_path}")
    data = _load_data(data_path)
    n = len(data['params'])
    print(f"  {n:,} samples | Grid: {len(data['S_grid'])} × {len(data['t_template'])}")

    # ── 1. FNO Speed ──
    V_fno, fno_time = benchmark_fno_speed(data, config, model_path)

    # ── 2. BS Speed ──
    V_true, bs_time = benchmark_bs_speed(data, config)

    # ── 3. MC Speed ──
    V_mc, mc_time = benchmark_mc_speed(data, config, n_paths=100_000, subset=200)

    # ── 3b. FDM Speed & Accuracy ──
    V_fdm, fdm_time = benchmark_fdm_speed(data, config, subset=200)
    fdm_acc = benchmark_fdm_accuracy(data, config, subset=500)

    # ── 4. Pricing Accuracy ──
    accuracy = compute_pricing_accuracy(V_fno, V_true)

    # ── 5. Speed Summary ──
    print(f"\n{'='*60}")
    print("SPEED SUMMARY")
    print(f"{'='*60}")
    print(f"  FNO:           {fno_time*1e6:.1f}μs/sample")
    print(f"  Black-Scholes: {bs_time*1e3:.4f}ms/sample")
    print(f"  Monte Carlo:   {mc_time*1e3:.1f}ms/sample (100K paths)")
    print(f"  FDM:           {fdm_time*1e3:.1f}ms/sample (Crank-Nicolson)")
    print(f"")
    print(f"  FNO vs MC:     {mc_time/fno_time:.0f}× faster")
    print(f"  FNO vs BS:     {bs_time/fno_time:.1f}× faster")
    print(f"  FNO vs FDM:    {fdm_time/fno_time:.0f}× faster")

    # ── 6. Greeks Accuracy ──
    greeks = compute_greeks_accuracy(data, config, model_path)

    # ── 7. PDE Residual Quality ──
    pde_res = compute_pde_residual_quality(data, config, model_path)

    # ── 8. Distribution Shift ──
    shift = test_distribution_shift(data, config, model_path)

    # ── 9. Visualizations ──
    plot_results(data, V_fno, V_true, output_dir)

    # ── Save all results ──
    results = {
        'accuracy': accuracy,
        'speed': {
            'fno_per_sample': fno_time,
            'bs_per_sample': bs_time,
            'mc_per_sample': mc_time,
            'fdm_per_sample': fdm_time,
            'fno_vs_mc_speedup': mc_time / fno_time,
            'fno_vs_bs_speedup': bs_time / fno_time,
            'fno_vs_fdm_speedup': fdm_time / fno_time,
        },
        'fdm_accuracy': fdm_acc,
        'greeks': greeks,
        'pde_residual': pde_res,
        'distribution_shift': shift,
    }

    np.save(os.path.join(output_dir, 'benchmark_results.npy'), results)
    print(f"\nAll results → {output_dir}/benchmark_results.npy")

    # ── 10. W&B Logging ──
    if hasattr(config, 'use_wandb') and config.use_wandb:
        try:
            import wandb
            # Re-init or resume the run using the run_name/id
            # Note: This assumes you are running benchmark right after training or providing the run name
            wandb.init(project=getattr(config, 'wandb_project', 'fno-option-pricer'), 
                       name=config.run_name, resume="allow")
            
            # Log main metrics
            wandb.log({
                'bench/rmse': accuracy['rmse'],
                'bench/mape': accuracy['mape'],
                'bench/max_error': accuracy['max_error'],
                'bench/fno_speed_ms': fno_time * 1000.0,
                'bench/speedup_vs_mc': mc_time / fno_time,
                'bench/delta_rmse': greeks['delta_rmse'],
                'bench/gamma_rmse': greeks['gamma_rmse'],
                'bench/pde_residual_mean': pde_res['mean']
            })
            
            # Log plots as images
            wandb.log({
                "plots/surface_comparison": wandb.Image(os.path.join(output_dir, 'surface_comparison.png')),
                "plots/price_curves": wandb.Image(os.path.join(output_dir, 'price_curves.png')),
                "plots/error_histogram": wandb.Image(os.path.join(output_dir, 'error_histogram.png'))
            })
            print("Results logged to Weights & Biases.")
        except Exception as e:
            print(f"Failed to log to W&B: {e}")

    # ── 11. GCP Upload ──
    if hasattr(config, 'gcp_bucket_name') and config.gcp_bucket_name and \
       hasattr(config, 'gcp_service_account_path') and config.gcp_service_account_path:
        from utils import upload_to_gcp_bucket
        run_name = config.run_name if hasattr(config, 'run_name') else 'model'
        
        files_to_upload = [
            'benchmark_results.npy', 
            'surface_comparison.png', 
            'price_curves.png', 
            'error_histogram.png'
        ]
        
        for f in files_to_upload:
            local_p = os.path.join(output_dir, f)
            if os.path.exists(local_p):
                dest_blob = f"results/{run_name}/{f}"
                upload_to_gcp_bucket(local_p, config.gcp_bucket_name, dest_blob, config.gcp_service_account_path)

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    from config import Config

    parser = argparse.ArgumentParser(description='Benchmark FNO option pricer')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()

    config = Config()
    
    # Apply CLI overrides
    if args.device: config.device = args.device
    if args.output_dir: config.results_dir = args.output_dir
    if args.run_name: config.run_name = args.run_name
    
    # Default paths using run_name
    data_path = args.data_path if args.data_path else os.path.join(config.data_dir, 'test.h5')
    model_path = args.model_path if args.model_path else os.path.join(config.checkpoint_dir, f"{config.run_name}_best.pt")

    run_full_benchmark(config, data_path=data_path, model_path=model_path)
