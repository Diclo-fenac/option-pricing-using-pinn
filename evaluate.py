"""
Evaluation & Visualization Script — Preprint Edition

Produces all figures and metrics needed for the preprint:
  1. Relative L2 error on full test set (target < 1%)
  2. Relative L2 error by sigma range (low / mid / high)
  3. Predicted vs ground-truth surface plots (2-3 samples)
  4. Error surface (difference map)
  5. FDM baseline comparison (accuracy + speed)
  6. Physics loss ablation: PDE vs data-only training curves

Usage:
    python evaluate.py --data_dir ./data --model_path ./checkpoints/best.pt
    python evaluate.py --run_ablation  # Train both PDE and data-only models
"""
import os
import time
import argparse
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from fno_model import FNOOptionPricer
from train import FNOTrainer, load_model
from fdm_solver import fdm_black_scholes


# Dataset compatibility layer (handles both data_generator formats)

class HDF5Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the large-scale data format (data_generator_large.py).

    HDF5 keys: params (N,4=[σ,r,K,T]), V, Delta, Gamma, S_grid, t_template
    """

    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
        if 'params' not in keys:
            raise ValueError(
                f"File {path} is not in large-scale format. "
                "Generate with: python data_generator_large.py --n_samples 10000"
            )
        self.params = f['params'][:]
        self.V = f['V'][:]
        self.Delta = f['Delta'][:] if 'Delta' in f else None
        self.Gamma = f['Gamma'][:] if 'Gamma' in f else None

    def __len__(self):
        return len(self.V)

    def __getitem__(self, i):
        item = {
            'sigma': torch.tensor(self.params[i, 0], dtype=torch.float32),
            'r': torch.tensor(self.params[i, 1], dtype=torch.float32),
            'K': torch.tensor(self.params[i, 2], dtype=torch.float32),
            'T': torch.tensor(self.params[i, 3], dtype=torch.float32),
            'V': torch.tensor(self.V[i], dtype=torch.float32),
            'params': torch.tensor(self.params[i], dtype=torch.float32),
        }
        if self.Delta is not None:
            item['Delta'] = torch.tensor(self.Delta[i], dtype=torch.float32)
        if self.Gamma is not None:
            item['Gamma'] = torch.tensor(self.Gamma[i], dtype=torch.float32)
        return item


# Helper

def _load_test_data(data_dir):
    """Load test set from HDF5. Requires large-scale format."""
    path = os.path.join(data_dir, 'test.h5')
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
    
        if 'params' not in keys:
            raise ValueError(
                f"Data file {path} uses the old format (separate sigma/r/K keys). "
                "Evaluation requires the large-scale format with 'params (N,4)' key. "
                "Generate data with: python data_generator_large.py --n_samples 10000"
            )
    
        return {
            'params': f['params'][:],        # (N, 4) [σ, r, K, T]
            'V': f['V'][:],                   # (N, n_S, n_t)
            'Delta': f['Delta'][:],
            'Gamma': f['Gamma'][:],
            'S_grid': f['S_grid'][:],         # (n_S,)
            't_template': f['t_template'][:], # (n_t,)
        }


def _predict_surfaces(model, data, device, batch_size=128):
    """Run FNO inference on all test samples. Returns V_pred as numpy."""
    params = torch.tensor(data['params'], dtype=torch.float32, device=device)
    S_grid = torch.tensor(data['S_grid'], dtype=torch.float32, device=device)
    t_template = torch.tensor(data['t_template'], dtype=torch.float32, device=device)

    sigma = params[:, 0]
    r = params[:, 1]
    K_norm = params[:, 2] / 100.0
    T_norm = params[:, 3] / 2.0

    V_list = []
    with torch.no_grad():
        for i in range(0, len(sigma), batch_size):
            end = min(i + batch_size, len(sigma))
            V_b = model(sigma[i:end], r[i:end], K_norm[i:end], T_norm[i:end],
                        S_grid, t_template)
            V_list.append(V_b.cpu().numpy())

    return np.concatenate(V_list, axis=0)


def relative_l2(V_pred, V_true):
    """Compute relative L2 error: ||V_pred - V_true|| / ||V_true||"""
    diff = V_pred - V_true
    return float(np.linalg.norm(diff) / (np.linalg.norm(V_true) + 1e-12))


# 1. Relative L2 Error (Full Test Set)

def evaluate_relative_l2(model, data, device):
    """Compute relative L2 error on full test set and by sigma regime."""
    print(f"\n{'='*60}")
    print("RELATIVE L2 ERROR EVALUATION")
    print(f"{'='*60}")

    V_pred = _predict_surfaces(model, data, device)
    V_true = data['V']

    # Full test set
    l2_full = relative_l2(V_pred, V_true)
    print(f"  Full test set (n={len(V_pred)}):  L2 = {l2_full:.6f}  ({l2_full*100:.4f}%)")

    # By sigma regime
    sigmas = data['params'][:, 0]
    n = len(sigmas)

    # Define regimes (adjust thresholds based on training distribution)
    sigma_lo = np.percentile(sigmas, 33)
    sigma_hi = np.percentile(sigmas, 67)

    masks = {
        f'Low σ  (σ < {sigma_lo:.2f})': sigmas < sigma_lo,
        f'Mid σ  ({sigma_lo:.2f} ≤ σ < {sigma_hi:.2f})': (sigmas >= sigma_lo) & (sigmas < sigma_hi),
        f'High σ (σ ≥ {sigma_hi:.2f})': sigmas >= sigma_hi,
    }

    regime_results = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        l2_regime = relative_l2(V_pred[mask], V_true[mask])
        n_regime = mask.sum()
        print(f"  {name:40s}: L2 = {l2_regime:.6f}  ({l2_regime*100:.4f}%, n={n_regime})")
        regime_results[name] = {'l2': l2_regime, 'n': int(n_regime)}

    # RMSE as supplementary
    rmse = np.sqrt(np.mean((V_pred - V_true) ** 2))
    max_err = np.max(np.abs(V_pred - V_true))
    mape = np.mean(np.abs(V_pred - V_true) / (np.abs(V_true) + 1e-8)) * 100
    print(f"\n  RMSE:      {rmse:.8f}")
    print(f"  Max error: {max_err:.6f}")
    print(f"  MAPE:      {mape:.4f}%")

    results = {
        'l2_full': l2_full,
        'regimes': regime_results,
        'rmse': rmse,
        'max_error': max_err,
        'mape': mape,
    }

    return results, V_pred


# 2. Surface Comparison Plots (Predicted vs GT + Error)

def plot_surface_comparison(V_pred, V_true, data, output_dir, n_samples=3):
    """
    Plot predicted vs ground-truth surfaces and error maps.

    For each sample:
      - Left:  Black-Scholes (ground truth)
      - Center: FNO prediction
      - Right: Absolute error surface
    """
    os.makedirs(output_dir, exist_ok=True)
    params = data['params']
    S_grid = data['S_grid']
    t_template = data['t_template']

    # Pick samples with diverse sigma values
    sigmas = params[:, 0]
    lo_idx = np.argmin(np.abs(sigmas - np.percentile(sigmas, 10)))
    mid_idx = np.argmin(np.abs(sigmas - np.percentile(sigmas, 50)))
    hi_idx = np.argmin(np.abs(sigmas - np.percentile(sigmas, 90)))
    sample_indices = [lo_idx, mid_idx, hi_idx][:n_samples]
    labels = ['Low σ', 'Mid σ', 'High σ'][:n_samples]

    for li, idx in enumerate(sample_indices):
        sigma, r, K, T = params[idx]
        t_abs = t_template * T

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Ground truth
        im0 = axes[0].imshow(
            V_true[idx], aspect='auto', origin='lower',
            extent=[t_abs[0], t_abs[-1], S_grid[0], S_grid[-1]],
            cmap='viridis', vmin=V_true[idx].min(), vmax=V_true[idx].max()
        )
        axes[0].set_title(f'Black-Scholes\nσ={sigma:.2f}, r={r:.3f}, K={K:.0f}, T={T:.2f}')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Asset Price')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # FNO prediction
        im1 = axes[1].imshow(
            V_pred[idx], aspect='auto', origin='lower',
            extent=[t_abs[0], t_abs[-1], S_grid[0], S_grid[-1]],
            cmap='viridis', vmin=V_true[idx].min(), vmax=V_true[idx].max()
        )
        axes[1].set_title(f'FNO Prediction ({labels[li]})')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Asset Price')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Error surface
        error = np.abs(V_pred[idx] - V_true[idx])
        im2 = axes[2].imshow(
            error, aspect='auto', origin='lower',
            extent=[t_abs[0], t_abs[-1], S_grid[0], S_grid[-1]],
            cmap='hot'
        )
        axes[2].set_title(f'Absolute Error (max={error.max():.6f})')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Asset Price')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        fname = os.path.join(output_dir, f'surface_comparison_{labels[li].replace(" ", "_").lower()}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

    # Combined price curves at different maturities
    _plot_price_curves(V_pred, V_true, data, output_dir, sample_indices[0], labels[0])


def _plot_price_curves(V_pred, V_true, data, output_dir, idx, label):
    """Plot price vs S at different maturities (predicted vs true)."""
    S_grid = data['S_grid']
    t_template = data['t_template']
    params = data['params'][idx]
    _, _, _, T = params
    t_abs = t_template * T

    fig, ax = plt.subplots(figsize=(10, 6))
    n_t = len(t_template)
    t_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for ti, c in zip(t_indices, colors):
        tau = T - t_abs[ti]
        ax.plot(S_grid, V_true[idx, :, ti], '--', color=c,
                label=f'BS τ={tau:.2f}', alpha=0.7, linewidth=2)
        ax.plot(S_grid, V_pred[idx, :, ti], '-', color=c,
                label=f'FNO τ={tau:.2f}', alpha=0.5, linewidth=1)

    ax.set_xlabel('Asset Price S', fontsize=12)
    ax.set_ylabel('Option Price V', fontsize=12)
    ax.set_title(f'Price Curves — {label} (σ={params[0]:.2f}, K={params[2]:.0f})', fontsize=13)
    ax.legend(ncol=2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_dir, f'price_curves_{label.replace(" ", "_").lower()}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# 3. Error Distribution Plots


def plot_error_distribution(V_pred, V_true, output_dir):
    """Histogram of absolute and relative errors."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    abs_err = np.abs(V_pred - V_true).flatten()
    axes[0].hist(abs_err, bins=100, log=True, edgecolor='none', alpha=0.8, color='#1f77b4')
    axes[0].set_xlabel('Absolute Error')
    axes[0].set_ylabel('Count (log)')
    axes[0].set_title('Error Distribution')
    axes[0].axvline(np.median(abs_err), color='r', ls='--',
                     label=f'Median: {np.median(abs_err):.6f}')
    axes[0].axvline(np.percentile(abs_err, 95), color='orange', ls='--',
                     label=f'95th: {np.percentile(abs_err, 95):.6f}')
    axes[0].legend()

    rel_err = np.abs((V_pred - V_true) / (np.abs(V_true) + 1e-8)).flatten()
    axes[1].hist(rel_err * 100, bins=100, log=True, edgecolor='none', alpha=0.8, color='#ff7f0e')
    axes[1].set_xlabel('Relative Error (%)')
    axes[1].set_ylabel('Count (log)')
    axes[1].set_title('Relative Error Distribution')
    axes[1].axvline(np.median(rel_err) * 100, color='r', ls='--',
                     label=f'Median: {np.median(rel_err)*100:.4f}%')
    axes[1].axvline(np.percentile(rel_err, 95) * 100, color='orange', ls='--',
                     label=f'95th: {np.percentile(rel_err, 95)*100:.4f}%')
    axes[1].legend()

    plt.tight_layout()
    fname = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# 4. FDM Baseline Comparison

def compare_fdm_baseline(data, model, device, subset=100):
    """
    Compare FNO vs FDM on accuracy and speed.

    Returns dict with FDM results and speed comparison table.
    """
    print(f"\n{'='*60}")
    print("FDM BASELINE COMPARISON")
    print(f"{'='*60}")

    S_grid = data['S_grid']
    t_template = data['t_template']
    params = data['params']
    V_true = data['V']

    n = min(subset, len(params))
    indices = np.linspace(0, len(params) - 1, n, dtype=int)
    S_uniform = np.linspace(S_grid.min(), S_grid.max(), len(S_grid), dtype=np.float64)

    # FDM run
    V_fdm = np.zeros((n, len(S_uniform), len(t_template)), dtype=np.float32)
    t0 = time.perf_counter()
    for i, idx in enumerate(tqdm(indices, desc='FDM', leave=False)):
        sigma, r, K, T = params[idx]
        t_uniform = np.linspace(0, T, len(t_template), dtype=np.float64)
        V_fdm[i] = fdm_black_scholes(S_uniform, t_uniform, sigma, r, K)
    fdm_elapsed = time.perf_counter() - t0
    fdm_per_sample = fdm_elapsed / n

    # FDM accuracy vs analytical
    V_true_subset = V_true[indices]
    fdm_rmse = np.sqrt(np.mean((V_fdm - V_true_subset) ** 2))
    fdm_max = np.max(np.abs(V_fdm - V_true_subset))
    fdm_l2 = relative_l2(V_fdm, V_true_subset)

    print(f"  FDM RMSE:      {fdm_rmse:.8f}")
    print(f"  FDM Max Error: {fdm_max:.6f}")
    print(f"  FDM Relative L2: {fdm_l2:.6f}")
    print(f"  FDM Time:      {fdm_elapsed:.1f}s ({fdm_per_sample*1e3:.1f}ms/sample)")

    # FNO run (same subset)
    sub_params = torch.tensor(params[indices], dtype=torch.float32, device=device)
    S_t = torch.tensor(S_grid, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_template, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        V_fno_sub = model(
            sub_params[:, 0], sub_params[:, 1],
            sub_params[:, 2] / 100.0, sub_params[:, 3] / 2.0,
            S_t, t_t
        ).cpu().numpy()
    fno_elapsed = time.perf_counter() - t0
    fno_per_sample = fno_elapsed / n

    # FNO accuracy (same subset)
    fno_rmse = np.sqrt(np.mean((V_fno_sub - V_true[indices]) ** 2))
    fno_max = np.max(np.abs(V_fno_sub - V_true[indices]))
    fno_l2 = relative_l2(V_fno_sub, V_true[indices])

    print(f"\n  FNO RMSE:      {fno_rmse:.8f}")
    print(f"  FNO Max Error: {fno_max:.6f}")
    print(f"  FNO Relative L2: {fno_l2:.6f}")
    print(f"  FNO Time:      {fno_elapsed:.4f}s ({fno_per_sample*1e6:.1f}μs/sample)")

    # Speed comparison table
    speedup = fdm_per_sample / fno_per_sample if fno_per_sample > 0 else float('inf')
    print(f"\n{'='*60}")
    print("SPEED COMPARISON TABLE")
    print(f"{'='*60}")
    sep = '-' * 55
    print(f"  {'Method':<15s} {'Time/sample':>15s} {'RMSE':>12s} {'Rel L2':>10s}")
    print(f"  {sep}")
    print(f"  {'FDM':<15s} {fdm_per_sample*1e3:>12.1f}ms {fdm_rmse:>12.8f} {fdm_l2:>10.6f}")
    print(f"  {'FNO':<15s} {fno_per_sample*1e6:>10.1f}μs {fno_rmse:>12.8f} {fno_l2:>10.6f}")
    print(f"  {sep}")
    print(f"  Speedup: FNO is {speedup:.0f}x faster than FDM")

    results = {
        'fdm_rmse': fdm_rmse,
        'fdm_max': fdm_max,
        'fdm_l2': fdm_l2,
        'fdm_per_sample': fdm_per_sample,
        'fno_rmse': fno_rmse,
        'fno_max': fno_max,
        'fno_l2': fno_l2,
        'fno_per_sample': fno_per_sample,
        'speedup': speedup,
    }

    return results


# 5. Physics Loss Ablation

def run_ablation(config, data_dir, output_dir):
    """
    Train two models:
      1. With PDE loss (physics-informed)
      2. Without PDE loss (data-only)

    Plot both validation loss curves on the same graph.

    Returns
    -------
    history_pde, history_data : dict
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'train.h5')
    val_path = os.path.join(data_dir, 'val.h5')

    print(f"\n{'='*60}")
    print("PHYSICS LOSS ABLATION")
    print(f"{'='*60}")

    # ── Model 1: Physics-Informed (PDE loss) ──
    print("\n[1/2] Training Physics-Informed model (with PDE loss)...")
    config_pde = _make_config(config, use_pde=True)
    trainer_pde = FNOTrainer(config_pde)
    history_pde = trainer_pde.train(
        HDF5Dataset(train_path),
        HDF5Dataset(val_path),
        use_pde=True
    )

    # ── Model 2: Data-Only (no PDE loss) ──
    print("\n[2/2] Training Data-Only model (no PDE loss)...")
    config_data = _make_config(config, use_pde=False)
    trainer_data = FNOTrainer(config_data)
    history_data = trainer_data.train(
        HDF5Dataset(train_path),
        HDF5Dataset(val_path),
        use_pde=False
    )

    # ── Plot comparison ──
    _plot_ablation(history_pde, history_data, output_dir)

    # ── Compare final metrics ──
    final_pde_val = history_pde['val_rmse'][-1]
    final_data_val = history_data['val_rmse'][-1]
    best_pde_val = min(history_pde['val_rmse'])
    best_data_val = min(history_data['val_rmse'])

    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"  Physics-Informed (PDE):")
    print(f"    Final val RMSE: {final_pde_val:.6e}")
    print(f"    Best val RMSE:  {best_pde_val:.6e}")
    print(f"    Epochs:         {len(history_pde['val_rmse'])}")
    print(f"  Data-Only:")
    print(f"    Final val RMSE: {final_data_val:.6e}")
    print(f"    Best val RMSE:  {best_data_val:.6e}")
    print(f"    Epochs:         {len(history_data['val_rmse'])}")
    print(f"  Improvement: {(1 - best_pde_val/best_data_val)*100:.1f}%")

    return history_pde, history_data


def _make_config(base_config, use_pde):
    """Create a config copy with PDE-specific settings."""
    import types
    cfg = types.SimpleNamespace()
    for k, v in vars(base_config).items():
        setattr(cfg, k, v)

    # Override for ablation
    cfg.n_epochs = getattr(base_config, 'ablation_epochs', 50)
    cfg.patience = getattr(base_config, 'ablation_patience', 15)
    cfg.checkpoint_dir = getattr(base_config, 'checkpoint_dir', './checkpoints')
    cfg.results_dir = getattr(base_config, 'results_dir', './results')

    if not use_pde:
        cfg.lambda_pde = 0.0

    return cfg


def _plot_ablation(history_pde, history_data, output_dir):
    """Plot ablation comparison: both val loss curves on same graph."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, max(len(history_pde['val_loss']), len(history_data['val_loss'])) + 1)

    # Validation loss
    ep_pde = list(range(1, len(history_pde['val_loss']) + 1))
    ep_data = list(range(1, len(history_data['val_loss']) + 1))
    axes[0].plot(ep_pde, history_pde['val_loss'], 'b-o', markersize=3, label='Physics-Informed (PDE)', linewidth=1.5)
    axes[0].plot(ep_data, history_data['val_loss'], 'r-s', markersize=3, label='Data-Only', linewidth=1.5)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation RMSE')
    axes[0].set_title('Validation Loss: PDE vs Data-Only')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Training loss components (PDE model only)
    ep_pde_t = list(range(1, len(history_pde['train_data_loss']) + 1))
    axes[1].plot(ep_pde_t, history_pde['train_data_loss'], 'b-', label='Data Loss', linewidth=1.5)
    ep_pde_p = list(range(1, len(history_pde['train_pde_loss']) + 1))
    axes[1].plot(ep_pde_p, history_pde['train_pde_loss'], 'g-', label='PDE Residual Loss', linewidth=1.5)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Physics-Informed: Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Greeks RMSE comparison
    if 'val_delta_rmse' in history_pde and len(history_pde['val_delta_rmse']) > 0:
        ep_dp = list(range(1, len(history_pde['val_delta_rmse']) + 1))
        ep_dd = list(range(1, len(history_data['val_delta_rmse']) + 1))
        axes[2].plot(ep_dp, history_pde['val_delta_rmse'], 'b-o', markersize=3, label='PDE Δ', linewidth=1.5)
        axes[2].plot(ep_dd, history_data['val_delta_rmse'], 'r-s', markersize=3, label='Data-Only Δ', linewidth=1.5)
        axes[2].set_yscale('log')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Delta RMSE')
        axes[2].set_title('Greeks Accuracy: Delta')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'ablation_pde_vs_data_only.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

    # Also save as numpy for programmatic access
    np.save(os.path.join(output_dir, 'ablation_history_pde.npy'), history_pde)
    np.save(os.path.join(output_dir, 'ablation_history_data.npy'), history_data)


# Full Evaluation Runner

def run_full_evaluation(config, data_dir=None, model_path=None):
    """Execute complete evaluation suite and produce all preprint figures."""
    if data_dir is None:
        data_dir = getattr(config, 'data_dir', './data')
    if model_path is None:
        model_path = os.path.join(getattr(config, 'checkpoint_dir', './checkpoints'), 'best.pt')

    output_dir = getattr(config, 'results_dir', './results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("EVALUATION SUITE: FNO Option Pricing")
    print(f"{'='*60}")

    # Load model
    model, device = load_model(config, model_path)
    print(f"Loaded model from: {model_path}")
    print(f"Device: {device}")

    # Load test data
    data = _load_test_data(data_dir)
    n = len(data['params'])
    print(f"Test set: {n:,} samples | Grid: {len(data['S_grid'])} × {len(data['t_template'])}")

    # ── 1. Relative L2 Error ──
    l2_results, V_pred = evaluate_relative_l2(model, data, device)

    # ── 2. Surface Comparison Plots ──
    print(f"\nGenerating surface comparison plots...")
    plot_surface_comparison(V_pred, data['V'], data, output_dir, n_samples=3)

    # ── 3. Error Distribution ──
    print(f"\nGenerating error distribution plots...")
    plot_error_distribution(V_pred, data['V'], output_dir)

    # ── 4. FDM Baseline Comparison ──
    print(f"\nRunning FDM baseline comparison...")
    fdm_results = compare_fdm_baseline(data, model, device, subset=100)

    # ── Save all results ──
    results = {
        'relative_l2': l2_results,
        'fdm_comparison': fdm_results,
    }
    np.save(os.path.join(output_dir, 'evaluation_results.npy'), results)
    print(f"\nAll results → {output_dir}/evaluation_results.npy")

    return results


# CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FNO option pricer')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--run_ablation', action='store_true',
                        help='Train both PDE and data-only models for ablation study')
    parser.add_argument('--ablation_epochs', type=int, default=50)
    parser.add_argument('--ablation_patience', type=int, default=15)
    args = parser.parse_args()

    class EvalConfig:
        device = args.device
        fno_modes = 12
        fno_layers = 3
        fno_width = 64
        S_grid_size = 256
        t_grid_size = 64
        S_min = 1e-3
        S_max = 600.0
        T_max = 2.0
        t_sampling_power = 2.0
        checkpoint_dir = './checkpoints'
        results_dir = args.output_dir
        data_dir = args.data_dir
        ablation_epochs = args.ablation_epochs
        ablation_patience = args.ablation_patience

    config = EvalConfig()

    if args.run_ablation:
        run_ablation(config, args.data_dir, args.output_dir)
    else:
        run_full_evaluation(config, args.data_dir, args.model_path)
