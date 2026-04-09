"""
Large-scale synthetic dataset generator for Neural Operator option pricing.
Generates (σ, r, K, T) → V(S, t) datasets using vectorized Black-Scholes.

Sampling: Latin Hypercube + importance near ATM and near-expiry.

Usage:
    python data_generator_large.py --n_samples 100000 --output ./data
"""
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import h5py
from typing import Tuple, Dict, Optional


# Configuration

class DataConfig:
    """Dataset generation configuration."""
    n_S = 256
    n_t = 64
    S_min = 1e-3

    sigma_min = 0.05;  sigma_max = 0.80
    r_min = 0.0;       r_max = 0.15
    K_min = 20.0;      K_max = 200.0
    T_min = 0.1;       T_max = 2.0

    t_sampling_power = 2.0
    n_samples = 100_000
    batch_size = 1024
    output_dir = './data'
    output_filename = 'fno_option_pricing.h5'
    seed = 42
    n_jobs = 4

    # Importance sampling concentration
    # Higher = more samples near ATM / near-expiry
    atm_concentration = 2.0     # K sampling sharpness around S_ref
    expiry_concentration = 1.5  # T sampling sharpness near lower bound


# Sampling: Latin Hypercube + Importance

def latin_hypercube_sample(n_samples: int,
                            bounds: list,
                            seed: int = 42) -> np.ndarray:
    """
    Latin Hypercube Sampling.

    Parameters
    ----------
    n_samples : int
    bounds : list of (lo, hi) for each dimension
    seed : int

    Returns
    -------
    samples : (n_samples, n_dims)
    """
    rng = np.random.default_rng(seed)
    n_dims = len(bounds)
    samples = np.zeros((n_samples, n_dims), dtype=np.float32)

    for d in range(n_dims):
        lo, hi = bounds[d]
        # Stratified bins
        edges = np.linspace(0.0, 1.0, n_samples + 1)
        # Random position within each bin
        u = edges[:-1] + rng.random(n_samples) * (edges[1] - edges[0])
        rng.shuffle(u)
        samples[:, d] = (lo + u * (hi - lo)).astype(np.float32)

    return samples


def importance_sampleK(n_samples: int,
                        K_min: float, K_max: float,
                        S_ref: float = 100.0,
                        concentration: float = 2.0,
                        seed: int = 42) -> np.ndarray:
    """
    Sample strikes with higher density near ATM (K ≈ S_ref).

    Uses a Beta distribution transformed to [K_min, K_max],
    peaked at S_ref.
    """
    rng = np.random.default_rng(seed)

    # Map S_ref to [0,1] in the [K_min, K_max] range
    p_atm = (S_ref - K_min) / (K_max - K_min)
    p_atm = np.clip(p_atm, 0.05, 0.95)

    # Beta parameters to peak at p_atm
    # mode = (α - 1) / (α + β - 2) = p_atm
    # concentration = α + β controls sharpness
    alpha = 1.0 + (p_atm) * concentration
    beta = 1.0 + (1.0 - p_atm) * concentration

    # Sample from Beta, transform to [K_min, K_max]
    u = rng.beta(alpha, beta, n_samples).astype(np.float32)
    K = K_min + u * (K_max - K_min)

    return K


def importance_sample_T(n_samples: int,
                         T_min: float, T_max: float,
                         concentration: float = 1.5,
                         seed: int = 42) -> np.ndarray:
    """
    Sample maturities with higher density near the lower bound (near-expiry).

    Uses a power-law transformation of uniform samples.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n_samples).astype(np.float32)
    # Power transform: cluster near u=0 → T=T_min
    u_transformed = u ** (1.0 / concentration)
    T = T_min + u_transformed * (T_max - T_min)
    return T


def generate_parameters(n_samples: int,
                        config: DataConfig) -> np.ndarray:
    """
    Generate (σ, r, K, T) using LHS for (σ, r) + importance sampling for (K, T).

    Returns
    -------
    params : (n_samples, 4) columns = [σ, r, K, T]
    """
    # LHS for σ and r (uniformly distributed)
    params_lhs = latin_hypercube_sample(
        n_samples,
        [(config.sigma_min, config.sigma_max),
         (config.r_min, config.r_max)],
        seed=config.seed
    )

    # Importance sampling for K and T
    K_samples = importance_sampleK(
        n_samples, config.K_min, config.K_max,
        S_ref=100.0, concentration=config.atm_concentration,
        seed=config.seed + 1
    )
    T_samples = importance_sample_T(
        n_samples, config.T_min, config.T_max,
        concentration=config.expiry_concentration,
        seed=config.seed + 2
    )

    params = np.column_stack([params_lhs, K_samples, T_samples])
    return params.astype(np.float32)


# Black-Scholes (Fully Vectorized, Numerically Stable)

def black_scholes_surface(sigma: np.ndarray,
                           r: np.ndarray,
                           K: np.ndarray,
                           T: np.ndarray,
                           S_grid: np.ndarray,
                           t_grid_template: np.ndarray) -> np.ndarray:
    """
    Compute V(S, t) for a batch of parameters. Fully vectorized.

    Parameters
    ----------
    sigma : (batch,)
    r : (batch,)
    K : (batch,)
    T : (batch,)
    S_grid : (n_S,)
    t_grid_template : (n_t,) values in [0, 1]

    Returns
    -------
    V : (batch, n_S, n_t)
    """
    batch_size = len(sigma)

    # Time grid per sample: t = t_template * T, τ = T - t
    t_per_sample = t_grid_template[np.newaxis, :] * T[:, np.newaxis]  # (batch, n_t)
    tau = T[:, np.newaxis] - t_per_sample
    tau_safe = np.maximum(tau, 1e-8)

    # Broadcasting: (batch, n_S, n_t)
    S = S_grid[np.newaxis, :, np.newaxis]         # (1, n_S, 1)
    K_b = K[:, np.newaxis, np.newaxis]             # (batch, 1, 1)
    sig_b = sigma[:, np.newaxis, np.newaxis]       # (batch, 1, 1)
    r_b = r[:, np.newaxis, np.newaxis]             # (batch, 1, 1)
    tau_b = tau_safe[:, np.newaxis, :]             # (batch, 1, n_t)

    sqrt_tau = np.sqrt(tau_b)
    ln_SK = np.log(S / K_b)

    d1 = (ln_SK + (r_b + 0.5 * sig_b**2) * tau_b) / (sig_b * sqrt_tau)
    d2 = d1 - sig_b * sqrt_tau

    d1 = np.clip(d1, -37.0, 37.0)
    d2 = np.clip(d2, -37.0, 37.0)

    N_d1 = 0.5 * (1.0 + np.clip(np.erf(d1 / np.sqrt(2.0)), -1.0, 1.0))
    N_d2 = 0.5 * (1.0 + np.clip(np.erf(d2 / np.sqrt(2.0)), -1.0, 1.0))

    discount = np.exp(-r_b * tau_b)

    V = S * N_d1 - K_b * discount * N_d2

    # Enforce payoff at expiry
    at_expiry = (tau <= 1e-8)[:, np.newaxis, :]
    payoff = np.maximum(S - K_b, 0.0)
    V = np.where(at_expiry, payoff, V)

    return V.astype(np.float32)


def compute_delta(sigma, r, K, T, S_grid, t_template):
    """Delta = N(d1) — analytical, vectorized."""
    batch_size = len(sigma)
    t_per_sample = t_template[np.newaxis, :] * T[:, np.newaxis]
    tau = np.maximum(T[:, np.newaxis] - t_per_sample, 1e-8)

    S = S_grid[np.newaxis, :, np.newaxis]
    K_b = K[:, np.newaxis, np.newaxis]
    sig_b = sigma[:, np.newaxis, np.newaxis]
    r_b = r[:, np.newaxis, np.newaxis]
    tau_b = tau[:, np.newaxis, :]

    sqrt_tau = np.sqrt(tau_b)
    ln_SK = np.log(S / K_b)
    d1 = (ln_SK + (r_b + 0.5 * sig_b**2) * tau_b) / (sig_b * sqrt_tau)
    d1 = np.clip(d1, -37.0, 37.0)

    delta = 0.5 * (1.0 + np.clip(np.erf(d1 / np.sqrt(2.0)), -1.0, 1.0))

    at_expiry = (T[:, np.newaxis, np.newaxis] * (1.0 - t_template[np.newaxis, np.newaxis, :]) <= 1e-8)
    delta_at_expiry = (S > K_b).astype(np.float32)
    delta = np.where(at_expiry, delta_at_expiry, delta)

    return delta.astype(np.float32)


def compute_gamma(sigma, r, K, T, S_grid, t_template):
    """Gamma = φ(d1) / (S·σ·√τ) — analytical, vectorized."""
    t_per_sample = t_template[np.newaxis, :] * T[:, np.newaxis]
    tau = np.maximum(T[:, np.newaxis] - t_per_sample, 1e-8)

    S = S_grid[np.newaxis, :, np.newaxis]
    K_b = K[:, np.newaxis, np.newaxis]
    sig_b = sigma[:, np.newaxis, np.newaxis]
    r_b = r[:, np.newaxis, np.newaxis]
    tau_b = tau[:, np.newaxis, :]

    sqrt_tau = np.sqrt(tau_b)
    ln_SK = np.log(S / K_b)
    d1 = (ln_SK + (r_b + 0.5 * sig_b**2) * tau_b) / (sig_b * sqrt_tau)
    d1 = np.clip(d1, -37.0, 37.0)

    phi_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    gamma = phi_d1 / (S * sig_b * sqrt_tau)

    at_expiry = (T[:, np.newaxis, np.newaxis] * (1.0 - t_template[np.newaxis, np.newaxis, :]) <= 1e-8)
    gamma = np.where(at_expiry, 0.0, gamma)

    return gamma.astype(np.float32)


# Grid Construction

def create_s_grid(S_min: float, K_max: float, n_S: int) -> np.ndarray:
    """Linear S grid: [S_min, 3·K_max]."""
    S_max = 3.0 * K_max
    return np.linspace(S_min, S_max, n_S, dtype=np.float32)


def create_t_template(n_t: int, power: float = 2.0) -> np.ndarray:
    """Non-uniform time template in [0, 1], clustered near maturity."""
    u = np.linspace(0, 1, n_t, dtype=np.float32)
    return u ** power


# Batch Generation

def generate_batch(params: np.ndarray,
                   S_grid: np.ndarray,
                   t_template: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate V, Delta, Gamma for one batch."""
    sigma, r, K, T = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

    V = black_scholes_surface(sigma, r, K, T, S_grid, t_template)
    Delta = compute_delta(sigma, r, K, T, S_grid, t_template)
    Gamma = compute_gamma(sigma, r, K, T, S_grid, t_template)

    return {'V': V, 'Delta': Delta, 'Gamma': Gamma}


def generate_full_dataset(n_samples: int,
                          batch_size: int,
                          S_grid: np.ndarray,
                          t_template: np.ndarray,
                          config: DataConfig,
                          verbose: bool = True) -> Tuple:
    """Generate full dataset in batches with LHS + importance sampling."""
    n_batches = int(np.ceil(n_samples / batch_size))

    # Preallocate
    params = np.zeros((n_samples, 4), dtype=np.float32)
    V = np.zeros((n_samples, len(S_grid), len(t_template)), dtype=np.float32)
    Delta = np.zeros_like(V, dtype=np.float32)
    Gamma = np.zeros_like(V, dtype=np.float32)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Generating {n_samples:,} samples ({n_batches} batches)")
        print(f"Grid: {len(S_grid)} × {len(t_template)}")
        mem = n_samples * len(S_grid) * len(t_template) * 4 * 3 / 1e9
        print(f"Memory estimate: {mem:.2f} GB (V + Delta + Gamma)")
        print(f"Sampling: LHS(σ,r) + importance(K near ATM, T near expiry)")
        print(f"{'='*60}\n")

    start_time = time.perf_counter()
    iterator = tqdm(range(n_batches), desc='Batches', unit='batch') if verbose else range(n_batches)

    for i in iterator:
        bs = i * batch_size
        be = min((i + 1) * batch_size, n_samples)
        actual = be - bs

        # Sample parameters for this batch (offset seed per batch)
        config_copy = DataConfig()
        for k, v in config.__dict__.items():
            setattr(config_copy, k, v)
        config_copy.seed = config.seed + i * 1000
        batch_params = generate_parameters(actual, config_copy)

        results = generate_batch(batch_params, S_grid, t_template)

        params[bs:be] = batch_params
        V[bs:be] = results['V']
        Delta[bs:be] = results['Delta']
        Gamma[bs:be] = results['Gamma']

    elapsed = time.perf_counter() - start_time
    if verbose:
        print(f"\nDone: {elapsed:.1f}s ({elapsed/n_samples*1e6:.0f}μs/sample, {n_samples/elapsed:.0f} samp/s)")

    return params, V, Delta, Gamma


# Storage

def save_to_hdf5(params, V, Delta, Gamma, S_grid, t_template, filepath, config):
    """Save with compression + metadata."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    print(f"\nSaving → {filepath}")

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('params', data=params, compression='gzip', compression_opts=4)
        f.create_dataset('V', data=V, compression='gzip', compression_opts=4)
        f.create_dataset('Delta', data=Delta, compression='gzip', compression_opts=4)
        f.create_dataset('Gamma', data=Gamma, compression='gzip', compression_opts=4)
        f.create_dataset('S_grid', data=S_grid)
        f.create_dataset('t_template', data=t_template)

        f.attrs['n_samples'] = len(params)
        f.attrs['n_S'] = len(S_grid)
        f.attrs['n_t'] = len(t_template)
        f.attrs['sigma_range'] = [config.sigma_min, config.sigma_max]
        f.attrs['r_range'] = [config.r_min, config.r_max]
        f.attrs['K_range'] = [config.K_min, config.K_max]
        f.attrs['T_range'] = [config.T_min, config.T_max]
        f.attrs['S_range'] = [float(S_grid.min()), float(S_grid.max())]
        f.attrs['seed'] = config.seed
        f.attrs['t_sampling_power'] = config.t_sampling_power
        f.attrs['option_type'] = 'call'
        f.attrs['sampling'] = 'LHS + importance (ATM, near-expiry)'

    print(f"  Size: {os.path.getsize(filepath)/1e6:.1f} MB")
    return filepath


def save_splits(params, V, Delta, Gamma, S_grid, t_template, config,
                train_ratio=0.8, val_ratio=0.1):
    """80/10/10 shuffled splits."""
    n = len(params)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    rng = np.random.default_rng(config.seed + 1)
    idx = rng.permutation(n)
    splits = {
        'train': idx[:n_train],
        'val': idx[n_train:n_train + n_val],
        'test': idx[n_train + n_val:]
    }

    os.makedirs(config.output_dir, exist_ok=True)
    paths = {}

    for name, indices in splits.items():
        path = os.path.join(config.output_dir, f'{name}.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('params', data=params[indices], compression='gzip', compression_opts=4)
            f.create_dataset('V', data=V[indices], compression='gzip', compression_opts=4)
            f.create_dataset('Delta', data=Delta[indices], compression='gzip', compression_opts=4)
            f.create_dataset('Gamma', data=Gamma[indices], compression='gzip', compression_opts=4)
            f.create_dataset('S_grid', data=S_grid)
            f.create_dataset('t_template', data=t_template)
            f.attrs['split'] = name
            f.attrs['n_samples'] = len(indices)
        paths[name] = path
        print(f"  {name:5s}: {len(indices):,} → {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    return paths


# Validation

def validate_dataset(params, V, S_grid, t_template, n_check=100, seed=123):
    """Independent recomputation check."""
    rng = np.random.default_rng(seed)
    n = len(params)
    indices = rng.choice(n, min(n_check, n), replace=False)

    errors = []
    print(f"\n{'='*60}")
    print(f"VALIDATION: {len(indices)} samples")
    print(f"{'='*60}")

    for idx in indices:
        sigma, r, K, T = params[idx]
        t_grid = t_template * T
        tau = np.maximum(T - t_grid, 1e-8)

        S = S_grid[:, np.newaxis]
        tau_b = tau[np.newaxis, :]
        sqrt_tau = np.sqrt(tau_b)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_b) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        d1, d2 = np.clip(d1, -37, 37), np.clip(d2, -37, 37)

        N_d1 = 0.5 * (1.0 + np.erf(d1 / np.sqrt(2.0)))
        N_d2 = 0.5 * (1.0 + np.erf(d2 / np.sqrt(2.0)))
        V_ref = S * N_d1 - K * np.exp(-r * tau_b) * N_d2

        at_expiry = (T - t_grid <= 1e-8)[np.newaxis, :]
        V_ref = np.where(at_expiry, np.maximum(S - K, 0.0), V_ref)

        err = np.abs(V[idx] - V_ref)
        errors.append({
            'idx': idx, 'mean_err': err.mean(), 'max_err': err.max(),
            'rmse': np.sqrt((err**2).mean())
        })

    means = np.array([e['mean_err'] for e in errors])
    maxes = np.array([e['max_err'] for e in errors])
    rmses = np.array([e['rmse'] for e in errors])

    print(f"  Mean error:  {means.mean():.10e} (±{means.std():.10e})")
    print(f"  Median err:  {np.median(means):.10e}")
    print(f"  Max error:   {maxes.max():.10e}")
    print(f"  Mean RMSE:   {rmses.mean():.10e}")
    print(f"  NaN count:   {np.isnan(V).sum()}")
    print(f"  Inf count:   {np.isinf(V).sum()}")

    return {'mean_errors': means, 'max_errors': maxes, 'rmses': rmses}


# Main

def main(config: Optional[DataConfig] = None):
    if config is None:
        config = DataConfig()

    parser = argparse.ArgumentParser(description='FNO option pricing dataset')
    parser.add_argument('--n_samples', type=int, default=config.n_samples)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--output_dir', type=str, default=config.output_dir)
    parser.add_argument('--seed', type=int, default=config.seed)
    args = parser.parse_args()

    config.n_samples = args.n_samples
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir
    config.seed = args.seed

    t0 = time.perf_counter()

    print(f"\n{'='*60}")
    print("NEURAL OPERATOR DATASET GENERATOR")
    print(f"{'='*60}")

    S_grid = create_s_grid(config.S_min, config.K_max, config.n_S)
    t_template = create_t_template(config.n_t, config.t_sampling_power)

    print(f"S_grid: [{S_grid.min():.4f}, {S_grid.max():.4f}] ({config.n_S} pts, linear)")
    print(f"t_template: [0, 1] ({config.n_t} pts, power={config.t_sampling_power})")

    params, V, Delta, Gamma = generate_full_dataset(
        config.n_samples, config.batch_size, S_grid, t_template, config
    )

    validate_dataset(params, V, S_grid, t_template, n_check=100)

    save_to_hdf5(params, V, Delta, Gamma, S_grid, t_template,
                 os.path.join(config.output_dir, config.output_filename), config)

    print(f"\n{'='*60}")
    print("SAVING SPLITS (80/10/10)")
    print(f"{'='*60}")
    save_splits(params, V, Delta, Gamma, S_grid, t_template, config)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Samples:     {config.n_samples:,}")
    print(f"  Grid:        {config.n_S} × {config.n_t}")
    print(f"  Time:        {elapsed:.1f}s ({config.n_samples/elapsed:.0f} samp/s)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
