# Operator Learning for Fast Option Pricing

A neural operator framework that learns the Black–Scholes solution operator to enable real-time, high-fidelity option pricing across dynamic market conditions.

## Overview

This project uses **Fourier Neural Operators (FNO)** to learn the mapping from market parameters (σ, r, K) to entire option pricing surfaces V(S, t), rather than predicting individual prices. This enables:

- **Real-time pricing**: O(1) inference after training
- **Full surface generation**: Output V(S, t) for all S and t simultaneously  
- **Greeks computation**: Automatic differentiation for Delta, Gamma, Vega
- **Distribution shift robustness**: Generalize across volatility regimes

## Architecture

```
(σ, r, K)  →  FNO (3 Fourier layers)  →  V(S_grid=256, t_grid=64)
```

**Key design choices:**
- 256×64 spatial-temporal grid
- Non-uniform time sampling (clustered near expiry)
- Physics-informed loss: L = L_data + λ_pde·L_PDE + λ_bc·L_BC
- Spectral convolution with 12 Fourier modes

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python data_generator.py
```
Generates training/validation/test datasets as HDF5 files.

### 3. Train the Model
```bash
python train.py
```
Trains FNO with physics-informed loss. Checkpoints saved to `./checkpoints/`.

### 4. Run Benchmarks
```bash
python benchmark.py
```
Compares FNO vs Black-Scholes vs Monte Carlo on accuracy, latency, and Greeks.

## Project Structure

```
Implementation_B/
├── config.py              # All hyperparameters
├── data_generator.py      # Synthetic data generation
├── fno_model.py           # Fourier Neural Operator
├── train.py               # Training pipeline
├── benchmark.py           # Benchmarking suite
├── utils.py               # Black-Scholes formulas, Greeks, metrics
├── requirements.txt
└── README.md
```

## Mathematical Background

The Black–Scholes PDE:

∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

With boundary condition V(S, T) = max(S - K, 0) for calls.

The FNO learns the solution operator:

**G: (σ, r, K) ↦ V(S, t)**

mapping parameter space to function space.

## References

- Li, Z. et al. "Fourier Neural Operator for Parametric PDEs" (ICLR 2021)
- Lu, L. et al. "Learning Nonlinear Operators via DeepONet" (Nature Mach. Intell. 2021)
