"""
Synthetic data generation for operator learning
Generates (σ, r, K) → V(S, t) datasets on a 256×64 grid
"""
import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count
from functools import partial

from config import Config
from utils import black_scholes_price, black_scholes_greeks


def create_time_grid(t_min, t_max, n_points, power=2.0):
    """
    Create non-uniform time grid with clustering near expiry (t=0).
    """
    u = np.linspace(0, 1, n_points)
    t_grid = t_max * (1 - u**power)
    t_grid = t_grid[::-1]
    return t_grid


def create_spatial_grid(S_min, S_max, n_points, S0=100.0):
    """
    Create asset price grid with slightly denser sampling around S0 (ATM region).
    """
    u = np.linspace(-1, 1, n_points)
    stretch = 0.8
    S_grid = S0 + (S_max - S_min) / 2 * np.tanh(u * stretch) / np.tanh(stretch)
    return S_grid


def generate_single_sample(params, S_grid, t_grid, S0=100.0, option_type='call'):
    """
    Generate a single pricing surface V(S, t) for given parameters.
    Optimized for vectorization.
    """
    sigma, r, K = params
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')
    
    # Vectorized Black-Scholes computation across the entire grid
    V = black_scholes_price(S_mesh, K, t_mesh, sigma, r, option_type)
    
    return V.astype(np.float32)


def generate_dataset(n_samples, S_grid, t_grid, config, seed=42, option_type='call', n_jobs=-1):
    """
    Generate full dataset of pricing surfaces using multiprocessing.
    """
    rng = np.random.default_rng(seed)
    S0 = config.S0
    
    # Sample parameters
    sigmas = rng.uniform(config.sigma_min, config.sigma_max, n_samples)
    rates = rng.uniform(config.r_min, config.r_max, n_samples)
    strikes = rng.uniform(config.K_min * S0, config.K_max * S0, n_samples)
    params_list = list(zip(sigmas, rates, strikes))
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    print(f"Generating {n_samples} surfaces on {len(S_grid)}x{len(t_grid)} grid using {n_jobs} cores...")
    
    # Use multiprocessing Pool
    worker_func = partial(generate_single_sample, S_grid=S_grid, t_grid=t_grid, S0=S0, option_type=option_type)
    
    with Pool(processes=n_jobs) as pool:
        V_surfaces = list(tqdm(pool.imap(worker_func, params_list), total=n_samples))
    
    V_surfaces = np.array(V_surfaces, dtype=np.float32)
    
    # Parallelize Greeks computation as well
    print("Computing Greeks for validation...")
    t_ref = t_grid[len(t_grid) // 2]
    
    def compute_greeks_worker(p):
        sigma, r, K = p
        return black_scholes_greeks(S0, K, t_ref, sigma, r, option_type)
    
    with Pool(processes=n_jobs) as pool:
        greeks_results = list(tqdm(pool.imap(compute_greeks_worker, params_list), total=n_samples))
    
    greeks = {
        'delta': np.array([g['delta'] for g in greeks_results], dtype=np.float32),
        'gamma': np.array([g['gamma'] for g in greeks_results], dtype=np.float32),
        'vega': np.array([g['vega'] for g in greeks_results], dtype=np.float32)
    }
    
    return {
        'sigma': sigmas.astype(np.float32),
        'r': rates.astype(np.float32),
        'K': strikes.astype(np.float32),
        'V': V_surfaces,
        'greeks': greeks,
        'S_grid': S_grid,
        't_grid': t_grid
    }



def save_dataset(data, filepath):
    """Save dataset to HDF5 format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('sigma', data=data['sigma'])
        f.create_dataset('r', data=data['r'])
        f.create_dataset('K', data=data['K'])
        f.create_dataset('V', data=data['V'])
        f.create_dataset('S_grid', data=data['S_grid'])
        f.create_dataset('t_grid', data=data['t_grid'])
        
        # Save Greeks
        for name, values in data['greeks'].items():
            f.create_dataset(f'greeks/{name}', data=values)
    
    print(f"Dataset saved to {filepath}")
    print(f"  Shapes: V={data['V'].shape}, sigma={data['sigma'].shape}")


def load_dataset(filepath):
    """Load dataset from HDF5 format."""
    with h5py.File(filepath, 'r') as f:
        data = {
            'sigma': f['sigma'][:],
            'r': f['r'][:],
            'K': f['K'][:],
            'V': f['V'][:],
            'S_grid': f['S_grid'][:],
            't_grid': f['t_grid'][:],
            'greeks': {}
        }
        
        if 'greeks' in f:
            for name in f['greeks'].keys():
                data['greeks'][name] = f[f'greeks/{name}'][:]
    
    return data


class OptionPricingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for option pricing surfaces."""
    
    def __init__(self, data_path=None, data_dict=None):
        """
        Initialize from file or dict.
        
        Parameters
        ----------
        data_path : str - Path to HDF5 file
        data_dict : dict - Pre-loaded data dictionary
        """
        if data_path is not None:
            self.data = load_dataset(data_path)
        elif data_dict is not None:
            self.data = data_dict
        else:
            raise ValueError("Must provide data_path or data_dict")
        
        self.V = torch.tensor(self.data['V'], dtype=torch.float32)
        self.sigma = torch.tensor(self.data['sigma'], dtype=torch.float32)
        self.r = torch.tensor(self.data['r'], dtype=torch.float32)
        self.K = torch.tensor(self.data['K'], dtype=torch.float32)
        
        # Normalize K relative to S0
        self.K_norm = self.K / Config.S0
    
    def __len__(self):
        return len(self.V)
    
    def __getitem__(self, idx):
        return {
            'sigma': self.sigma[idx],
            'r': self.r[idx],
            'K_norm': self.K_norm[idx],
            'V': self.V[idx]
        }


if __name__ == '__main__':
    config = Config()
    
    # Create grids
    S_grid = create_spatial_grid(config.S_min, config.S_max, config.S_grid_size, config.S0)
    t_grid = create_time_grid(config.t_min, config.t_max, config.t_grid_size, config.t_sampling_power)
    
    print(f"S_grid: {S_grid.min():.1f} → {S_grid.max():.1f} ({len(S_grid)} points)")
    print(f"t_grid: {t_grid.min():.2f} → {t_grid.max():.2f} ({len(t_grid)} points)")
    
    # Generate datasets
    print("\n=== Generating Training Data ===")
    train_data = generate_dataset(
        config.n_train_samples, S_grid, t_grid, config, seed=42
    )
    save_dataset(train_data, os.path.join(config.data_dir, 'train.h5'))
    
    print("\n=== Generating Validation Data ===")
    val_data = generate_dataset(
        config.n_val_samples, S_grid, t_grid, config, seed=123
    )
    save_dataset(val_data, os.path.join(config.data_dir, 'val.h5'))
    
    print("\n=== Generating Test Data ===")
    test_data = generate_dataset(
        config.n_test_samples, S_grid, t_grid, config, seed=456
    )
    save_dataset(test_data, os.path.join(config.data_dir, 'test.h5'))
    
    print("\nData generation complete!")
