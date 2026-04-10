"""
Experiment configuration
"""
import torch
import os

class Config:
    # =============================================================================
    # Grid settings
    # =============================================================================
    S_grid_size = 256          # Asset price grid points
    t_grid_size = 64           # Time grid points
    S_min = 1e-3               # Minimum asset price
    S_max = 600.0              # Maximum asset price
    T_min = 0.1                # Minimum time to expiry
    T_max = 2.0                # Maximum time to expiry (2 years)
    
    # Non-uniform time sampling: cluster points near expiry
    t_sampling_power = 2.0     # Higher = more clustering near t=0 (expiry)
    
    # =============================================================================
    # Parameter ranges for operator training
    # =============================================================================
    sigma_min = 0.05           # Minimum volatility
    sigma_max = 0.80           # Maximum volatility
    r_min = 0.00               # Minimum risk-free rate
    r_max = 0.15               # Maximum risk-free rate
    K_min = 20.0               # Minimum strike
    K_max = 200.0              # Maximum strike
    
    # Dataset Generation settings
    n_samples = 100000         # Number of samples
    batch_size_data = 1024
    output_filename = 'fno_option_pricing.h5'
    atm_concentration = 2.0
    expiry_concentration = 1.5
    seed = 42

    # =============================================================================
    # FNO Architecture
    # =============================================================================
    fno_modes = 24             # Number of Fourier modes
    fno_layers = 4             # Number of Fourier layers
    fno_width = 64             # Channel width
    fno_nonlinearity = 'gelu'  # Activation function
    
    # =============================================================================
    # Training
    # =============================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-4
    n_epochs = 200
    patience = 25
    scheduler_step = 50
    scheduler_gamma = 0.5
    
    # Physics-informed loss weight
    lambda_pde = 0.1           # Weight for PDE residual loss
    lambda_bc = 0.1            # Weight for boundary condition loss
    
    # Dataloader / multiprocessing (set lower/disabled for notebook stability)
    num_workers = 0
    pin_memory = False
    persistent_workers = False
    num_threads = 0
    num_interop_threads = 0
    
    # Logging
    use_wandb = False
    wandb_project = 'fno-option-pricer'
    run_benchmark = True                 # Automatically run benchmark after training

    # =============================================================================
    # Paths and Cloud Storage
    # =============================================================================
    run_name = 'fno_model_v1'  # Identifier for this specific training run
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    results_dir = './results'
    
    # GCP Settings
    gcp_bucket_name = None               # e.g., 'my-option-pricing-bucket'
    gcp_service_account_path = None      # e.g., './service-account-key.json'
    
    # =============================================================================
    # Benchmarking
    # =============================================================================
    n_mc_paths = 100000        # Monte Carlo paths for benchmarking
    mc_seed = 42
