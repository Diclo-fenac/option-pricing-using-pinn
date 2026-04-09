import json

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def mk_md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def mk_code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    }

def create_data_gen_nb():
    cells = []
    
    cells.append(mk_md(r"""# Phase 1: Parallel Data Generation for Option Pricing FNO
This notebook generates the synthetic training, validation, and test data for learning the Black-Scholes solution operator.

**Key Enhancement:** This version utilizes **multiprocessing** to parallelize the generation of pricing surfaces across all available CPU cores.

**Goal:** Learn the mapping $G : (\sigma, r, K) \mapsto V(S, t)$"""))
    
    cells.append(mk_code(r"""import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from multiprocessing import cpu_count

# Import local modules
from config import Config
from data_generator import create_time_grid, create_spatial_grid, generate_dataset, save_dataset

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'pdf.fonttype': 42, 'ps.fonttype': 42})

print(f"Available CPU cores: {cpu_count()}")"""))

    cells.append(mk_md(r"""## 1. Configuration & Grid Setup
We use the `Config` class to define parameter bounds. 
- $S \in [50, 150]$
- $t \in [0, 2]$ (time to expiry)
- Volatility $\sigma \in [0.1, 0.8]$
- Risk-free rate $r \in [0.0, 0.1]$
- Strike $K \in [70, 130]$"""))

    cells.append(mk_code(r"""config = Config()

# Create non-uniform grids
S_grid = create_spatial_grid(config.S_min, config.S_max, config.S_grid_size, config.S0)
t_grid = create_time_grid(config.t_min, config.t_max, config.t_grid_size, config.t_sampling_power)

print(f"Spatial Grid (S): {len(S_grid)} points")
print(f"Time Grid (t): {len(t_grid)} points")"""))

    cells.append(mk_md(r"""## 2. Parallel Dataset Generation
We use `generate_dataset` which now leverages `multiprocessing.Pool` to speed up the computation of 10,000+ surfaces.

`n_jobs=-1` will use all available cores."""))

    cells.append(mk_code(r"""os.makedirs(config.data_dir, exist_ok=True)

# Generate Validation Data
print("Generating Validation Data...")
val_data = generate_dataset(config.n_val_samples, S_grid, t_grid, config, seed=123, n_jobs=-1)
save_dataset(val_data, os.path.join(config.data_dir, 'val.h5'))

# Generate Test Data
print("Generating Test Data...")
test_data = generate_dataset(config.n_test_samples, S_grid, t_grid, config, seed=456, n_jobs=-1)
save_dataset(test_data, os.path.join(config.data_dir, 'test.h5'))

# Generate Training Data
print("Generating Training Data...")
train_data = generate_dataset(config.n_train_samples, S_grid, t_grid, config, seed=42, n_jobs=-1)
save_dataset(train_data, os.path.join(config.data_dir, 'train.h5'))"""))

    cells.append(mk_md(r"""## 3. Visualize a Sample Surface"""))

    cells.append(mk_code(r"""sample_idx = 0
V_sample = train_data['V'][sample_idx]
sigma_sample = train_data['sigma'][sample_idx]
r_sample = train_data['r'][sample_idx]
K_sample = train_data['K'][sample_idx]

S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing='ij')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_mesh, t_mesh, V_sample, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_title(f'Black-Scholes Call Surface\n$\sigma={sigma_sample:.2f}, r={r_sample:.2f}, K={K_sample:.1f}$')
plt.show()"""))

    nb = create_nb(cells)
    with open('01_Data_Generation.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)

def create_model_train_nb():
    # Model training notebook remains essentially the same as it uses PyTorch which is already optimized
    cells = []
    
    cells.append(mk_md(r"""# Phase 1: Model Training (FNO)
This notebook sets up the training pipeline for the Fourier Neural Operator.

**Key Features:**
1. **Fourier Positional Encoding**
2. **Coordinate-Aware Decoder**
3. **Hard-Constrained Output**
4. **Physics-Informed Loss (AD-based)**"""))

    cells.append(mk_code(r"""import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from fno_model import FNOOptionPricer
from train import FNOTrainer
from data_generator import OptionPricingDataset

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'pdf.fonttype': 42, 'ps.fonttype': 42})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")"""))

    cells.append(mk_md(r"""## 1. Load Datasets"""))

    cells.append(mk_code(r"""config = Config()
train_dataset = OptionPricingDataset(data_path=os.path.join(config.data_dir, 'train.h5'))
val_dataset = OptionPricingDataset(data_path=os.path.join(config.data_dir, 'val.h5'))
print(f"Loaded {len(train_dataset)} training samples.")"""))

    cells.append(mk_md(r"""## 2. Initialize Model & Trainer"""))

    cells.append(mk_code(r"""config.n_epochs = 10 
trainer = FNOTrainer(config)"""))

    cells.append(mk_md(r"""## 3. Run Training"""))

    cells.append(mk_code(r"""# trainer.train(train_dataset, val_dataset)"""))

    cells.append(mk_md(r"""## 4. Training Curves"""))

    cells.append(mk_code(r"""# Visualization code here as before..."""))

    nb = create_nb(cells)
    with open('02_Model_Training.ipynb', 'w') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    create_data_gen_nb()
    create_model_train_nb()
