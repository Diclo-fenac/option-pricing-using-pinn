"""
Training pipeline for FNO-based option pricing — Production Edition

Key upgrades:
  1. AD-based PDE residuals (no finite differences)
  2. NTK-based adaptive loss weighting (no fixed λ)
  3. Curriculum training (easy → hard samples)
  4. Greeks validation (Delta, Gamma accuracy tracking)
  5. Hard-constrained boundaries (no BC loss needed)
"""
import os
import io
import gzip
import time
import math
import threading
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fno_model import FNOOptionPricer, compute_pde_residual_autograd, compute_greeks_autograd

# Optional cloud logging
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# Checkpoint key sets for slim vs full uploads
RESUME_KEYS = ['epoch', 'iteration', 'model', 'optimizer', 'scheduler']
FULL_KEYS = RESUME_KEYS + ['history', 'config', 'run_name']


# Adaptive Loss Weighting (NTK-based)

class AdaptiveLossWeights:
    """
    NTK-inspired adaptive loss weighting.

    Balances gradient magnitudes across loss components so that no
    single term dominates the optimization landscape.

    References:
      - Wang et al. "Understanding and mitigating gradient pathologies in PINNs" (2021)
      - SoftAdapt: "Dynamic loss weighting for physics-informed neural networks"
    """

    def __init__(self, n_terms, alpha=0.9, min_weight=0.01, max_weight=1e6):
        """
        Parameters
        ----------
        n_terms : int — number of loss components (e.g., 2: data + PDE)
        alpha : float — EMA smoothing factor
        min_weight, max_weight : clamp bounds
        """
        self.n_terms = n_terms
        self.alpha = alpha
        self.min_w = min_weight
        self.max_w = max_weight
        # Track moving average of gradient norms
        self.grad_norms_ema = None
        # Track moving average of loss values
        self.loss_ema = None

    def update(self, model, loss_terms):
        """
        Compute adaptive weights based on gradient norms.

        Parameters
        ----------
        model : nn.Module
        loss_terms : list of scalar tensors — [L_data, L_pde, ...]

        Returns
        -------
        weights : (n_terms,) — adaptive weights for each loss term
        """
        # Compute gradient norm for each loss term separately
        grad_norms = []
        for i, loss in enumerate(loss_terms):
            model.zero_grad()
            loss.backward(retain_graph=(i < len(loss_terms) - 1))

            norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    norm_sq += p.grad.data.norm(2).item() ** 2
            grad_norms.append(math.sqrt(norm_sq) + 1e-8)

        grad_norms = torch.tensor(grad_norms, device=loss_terms[0].device)

        if self.grad_norms_ema is None:
            self.grad_norms_ema = torch.ones(self.n_terms, device=loss_terms[0].device)

        # EMA update
        self.grad_norms_ema = (
            self.alpha * self.grad_norms_ema + (1 - self.alpha) * grad_norms
        )

        # NTK weighting: balance to reference (first term = data loss)
        ref = self.grad_norms_ema[0]
        weights = ref / self.grad_norms_ema

        # Clamp
        weights = weights.clamp(self.min_w, self.max_w)

        # Detach so weights don't pollute gradients
        return weights.detach()

    def update_cheap(self, loss_terms):
        """
        Cheaper loss-ratio based weighting (no extra backward passes).
        Uses moving average of loss magnitudes instead of gradient norms.

        Parameters
        ----------
        loss_terms : list of scalar tensors

        Returns
        -------
        weights : (n_terms,)
        """
        losses = torch.tensor([l.item() + 1e-12 for l in loss_terms], device=loss_terms[0].device)

        if self.loss_ema is None:
            self.loss_ema = losses.clone()

        # EMA of loss magnitudes
        self.loss_ema = self.alpha * self.loss_ema + (1 - self.alpha) * losses

        # Scale all terms to the data-loss reference. This keeps extreme PDE
        # loss spikes from dominating the weighted objective.
        ref = self.loss_ema[0]
        weights = ref / self.loss_ema.clamp(min=1e-12)
        weights = weights.clamp(self.min_w, self.max_w)

        return weights.detach().to(loss_terms[0].device)


# Curriculum Scheduler

class CurriculumScheduler:
    """
    Curriculum training: progressively increase sample difficulty.

    Phase 1 (epochs 0–20):   Easy — long maturity, ATM strikes, moderate vol
    Phase 2 (epochs 20–60):  Medium — add shorter maturities, wider strikes
    Phase 3 (epochs 60+):    Full — entire parameter distribution
    """

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def get_active_mask(self, params, epoch):
        """
        Return boolean mask of samples to include at this epoch.

        Parameters
        ----------
        params : (N, 4) — [σ, r, K_raw, T]  (K is raw strike, NOT normalized)
        epoch : int

        Returns
        -------
        mask : (N,) bool
        """
        sigma = params[:, 0]
        # r = params[:, 1]  # not used in curriculum
        K = params[:, 2]       # raw strike (e.g., 100), NOT K_norm
        T = params[:, 3]       # raw time to expiry

        # Curriculum phases
        if epoch < 20:
            # Phase 1: easy — T > 0.5, K ∈ [60, 140], σ ∈ [0.15, 0.5]
            mask = (T > 0.5) & (K > 60) & (K < 140) & (sigma > 0.15) & (sigma < 0.5)
        elif epoch < 60:
            # Phase 2: medium — T > 0.2, K ∈ [40, 180], σ ∈ [0.08, 0.65]
            mask = (T > 0.2) & (K > 40) & (K < 180) & (sigma > 0.08) & (sigma < 0.65)
        else:
            # Phase 3: all samples
            mask = torch.ones(len(params), dtype=torch.bool)

        # Ensure minimum batch size
        if mask.sum() < 8:
            # Fall back to easiest samples
            difficulty = T + (K - 100).abs() / 100 + sigma.abs()
            _, easiest_idx = torch.topk(difficulty, min(32, len(params)), largest=False)
            mask = torch.zeros(len(params), dtype=torch.bool)
            mask[easiest_idx] = True

        return mask


# Trainer

class FNOTrainer:
    """
    Production training loop with:
      - AD-based PDE residuals
      - Adaptive loss weighting
      - Curriculum training
      - Greeks tracking
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if hasattr(config, 'device') else 'cuda')

        # Model
        self.model = FNOOptionPricer(config).to(self.device)

        # Grids
        self._init_grids(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 1e-4
        )

        lr = config.learning_rate if hasattr(config, 'learning_rate') else 1e-3
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=lr * 0.01
        )

        # Adaptive weighting: 2 terms (data + PDE)
        self.weighting = AdaptiveLossWeights(n_terms=2, alpha=0.9)

        # Curriculum
        self.curriculum = CurriculumScheduler(
            config.n_epochs if hasattr(config, 'n_epochs') else 200
        )

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_data_loss': [], 'train_pde_loss': [],
            'val_rmse': [], 'val_delta_rmse': [], 'val_gamma_rmse': [],
            'lr': [], 'weights_data': [], 'weights_pde': []
        }

        # Directories
        ckpt_dir = config.checkpoint_dir if hasattr(config, 'checkpoint_dir') else './checkpoints'
        res_dir = config.results_dir if hasattr(config, 'results_dir') else './results'
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        # GCP Upload Queue (persistent worker, avoids thread pile-up)
        self._upload_queue = None
        self._upload_worker_thread = None
        self._last_gcp_upload_epoch = -999  # Track frequency control for best.pt
        self._init_upload_queue()

    def _init_upload_queue(self):
        """Initialize persistent upload worker thread with queue."""
        if not (hasattr(self.config, 'gcp_bucket_name') and self.config.gcp_bucket_name and
                hasattr(self.config, 'gcp_service_account_path') and self.config.gcp_service_account_path):
            return

        maxsize = getattr(self.config, 'gcp_upload_queue_size', 3)
        self._upload_queue = Queue(maxsize=maxsize)

        def worker():
            while True:
                task = self._upload_queue.get()
                if task is None:  # Poison pill
                    break
                local_path, dest_blob = task
                try:
                    from utils import upload_to_gcp_bucket
                    upload_to_gcp_bucket(
                        local_path,
                        self.config.gcp_bucket_name,
                        dest_blob,
                        self.config.gcp_service_account_path,
                        storage_class=getattr(self.config, 'gcp_storage_class', 'STANDARD')
                    )
                    print(f"[GCP Upload] ✓ {os.path.basename(local_path)} → gs://{self.config.gcp_bucket_name}/{dest_blob}")
                except Exception as e:
                    print(f"[GCP Upload Warning] Failed to upload {local_path}: {e}")
                finally:
                    self._upload_queue.task_done()

        self._upload_worker_thread = threading.Thread(target=worker, daemon=True)
        self._upload_worker_thread.start()

    def _init_grids(self, config):
        """Initialize S and t grids."""
        # S grid: log-spaced from S_min to S_max to match decoder log-S normalization
        S_min = config.S_min if hasattr(config, 'S_min') else 1e-3
        S_max = config.S_max if hasattr(config, 'S_max') else 600.0
        n_S = config.S_grid_size if hasattr(config, 'S_grid_size') else 256
        n_t = config.t_grid_size if hasattr(config, 't_grid_size') else 64
        t_power = config.t_sampling_power if hasattr(config, 't_sampling_power') else 2.0
        T_max = config.T_max if hasattr(config, 'T_max') else 2.0

        S_log_min = math.log(S_min)
        S_log_max = math.log(S_max)
        S_grid_log = torch.linspace(S_log_min, S_log_max, n_S, dtype=torch.float32, device=self.device)
        self.S_grid = torch.exp(S_grid_log)
        t_u = torch.linspace(0, 1, n_t, dtype=torch.float32, device=self.device) ** t_power
        self.t_grid = t_u * T_max

        # Interior points for PDE residual evaluation
        # Avoid boundaries where derivatives are noisy
        n_interior_S = max(32, n_S // 4)
        n_interior_t = max(16, n_t // 2)
        S_interior_log = torch.linspace(
            math.log(S_min * 1.1), math.log(S_max * 0.9),
            n_interior_S, dtype=torch.float32, device=self.device
        )
        self.S_interior = torch.exp(S_interior_log)
        self.t_interior = torch.linspace(T_max * 0.05, T_max * 0.95, n_interior_t,
                                         dtype=torch.float32, device=self.device)

        print(f"Grid: S∈[{S_min:.2f},{S_max:.1f}] ({n_S}, log-spaced), t∈[0,{T_max}] ({n_t})")
        print(f"PDE interior: {n_interior_S} × {n_interior_t} points")

    def _compute_loss(self, batch, compute_pde=True):
        """
        Composite loss: L = w_data · L_data + w_pde · L_PDE

        No boundary loss needed — enforced by hard-constrained ansatz.
        """
        # Extract batch
        sigma = batch['sigma'].to(self.device)
        r = batch['r'].to(self.device)
        K = batch['K'].to(self.device)
        T = batch['T'].to(self.device)
        V_true = batch['V'].to(self.device)

        # Normalize
        K_norm = K / 100.0
        T_norm = T / 2.0

        # Forward
        V_pred = self.model(sigma, r, K_norm, T_norm, self.S_grid, self.t_grid)

        # Data loss (MSE)
        data_loss = torch.mean((V_pred - V_true) ** 2)

        # PDE residual loss (AD-based)
        pde_loss = torch.tensor(0.0, device=self.device)
        if compute_pde:
            # Only compute PDE on a subset for efficiency (every 3rd batch)
            residual, dV_dS, d2V_dS2, dV_dt, V_pde = compute_pde_residual_autograd(
                self.model, sigma, r, K_norm, T_norm,
                self.S_grid, self.t_grid,
                self.S_interior, self.t_interior,
                return_value=True
            )
            # Normalize PDE residual by option-price scale per sample.
            V_scale = V_pde.detach().abs().mean(dim=(1, 2), keepdim=True).clamp(min=1.0)
            pde_loss = torch.mean((residual / V_scale) ** 2)

            # Sobolev Loss: Match AD derivatives with true Delta and Gamma on the interior points
            # DISABLED: The current AD setup computes global gradients (summed over batch/time),
            # which doesn't match the per-batch-element structure needed for proper Sobolev training.
            # The PDE residual already enforces correct derivative relationships.
            # if 'Delta' in batch and 'Gamma' in batch:
            #     Delta_true = batch['Delta'].to(self.device)
            #     Gamma_true = batch['Gamma'].to(self.device)
            #
            #     # Interpolate true Delta and Gamma to the interior points
            #     Delta_interp = self.model._bilinear_interpolate(Delta_true, self.S_grid, self.t_grid, self.S_interior, self.t_interior)
            #     Gamma_interp = self.model._bilinear_interpolate(Gamma_true, self.S_grid, self.t_grid, self.S_interior, self.t_interior)
            #
            #     # Scale Sobolev terms (weight them as recommended: 0.5 for Delta, 0.1 for Gamma)
            #     delta_mse = torch.mean((dV_dS - Delta_interp)**2)
            #     gamma_mse = torch.mean((d2V_dS2 - Gamma_interp)**2)
            #     sobolev_loss = 0.5 * delta_mse + 0.1 * gamma_mse
            #     data_loss = data_loss + sobolev_loss

        return data_loss, pde_loss, V_pred

    def train_epoch(self, dataloader, epoch, use_curriculum=True, use_pde=True):
        """Single training epoch with curriculum and adaptive weighting.

        Parameters
        ----------
        dataloader : DataLoader
        epoch : int
        use_curriculum : bool
        use_pde : bool — if False, skip PDE loss (data-only ablation)
        """
        self.model.train()
        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_pde_loss = 0.0
        n_batches = 0
        weights = torch.tensor([1.0, 0.0], device=self.device)
        if epoch == 5 and use_pde:
            self.weighting.loss_ema = None

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)):
            # Curriculum filtering
            if use_curriculum and 'params' in batch:
                params = batch['params']  # keep on CPU for filtering
                mask = self.curriculum.get_active_mask(params, epoch)
                if mask.sum() < 8:
                    continue
                batch = {k: v[mask] for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Compute losses (skip PDE if use_pde=False)
            data_loss, pde_loss, _ = self._compute_loss(batch, compute_pde=use_pde and (batch_idx % 3 == 0))

            # Adaptive weighting
            if epoch < 5 or not use_pde:
                # Warm-up or data-only: data loss only
                weights = torch.tensor([1.0, 0.0], device=self.device)
            elif use_pde and (batch_idx % 3 == 0):
                weights = self.weighting.update_cheap([data_loss, pde_loss])
            # else: keep previous weights

            weighted_loss = weights[0] * data_loss
            if weights[1].item() > 0:
                weighted_loss = weighted_loss + weights[1] * pde_loss

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_total_loss += weighted_loss.item()
            epoch_data_loss += data_loss.item()
            epoch_pde_loss += pde_loss.item() if weights[1].item() > 0 else 0.0
            n_batches += 1

            # Record weights
            if batch_idx == 0:
                self.history['weights_data'].append(weights[0].item())
                self.history['weights_pde'].append(weights[1].item())

        n_batches = max(n_batches, 1)
        return {
            'total_loss': epoch_total_loss / n_batches,
            'data_loss': epoch_data_loss / n_batches,
            'pde_loss': epoch_pde_loss / n_batches
        }

    def validate(self, dataloader):
        """Validation: RMSE + Greeks accuracy."""
        self.model.eval()
        total_rmse = 0.0
        total_rel_l2 = 0.0
        total_delta_rmse = 0.0
        total_gamma_rmse = 0.0
        total_data_loss = 0.0
        
        # Subset tracking
        rel_l2_itm, rel_l2_atm, rel_l2_otm = 0.0, 0.0, 0.0
        n_itm, n_atm, n_otm = 0, 0, 0
        
        n_batches = 0
        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                sigma = batch['sigma'].to(self.device)
                r = batch['r'].to(self.device)
                K = batch['K'].to(self.device)
                T = batch['T'].to(self.device)
                V_true = batch['V'].to(self.device)

                K_norm = K / 100.0
                T_norm = T / 2.0

                # Compute data loss only on validation (skip expensive PDE residual)
                data_loss, _, V_pred = self._compute_loss(batch, compute_pde=False)

                rmse = torch.sqrt(torch.mean((V_pred - V_true) ** 2))
                total_rmse += rmse.item() * len(sigma)
                
                # Relative L2 Error
                batch_rel_l2 = torch.sqrt(torch.mean((V_pred - V_true) ** 2, dim=(1, 2))) / (
                    torch.sqrt(torch.mean(V_true ** 2, dim=(1, 2))) + 1e-12
                )
                total_rel_l2 += torch.sum(batch_rel_l2).item()
                
                # Subset categorization
                # Use S=100 as reference for ATM/ITM/OTM splits based on strike K
                moneyness = 100.0 / K
                
                # OTM: S/K < 0.95 (call) -> moneyness < 0.95
                otm_mask = moneyness < 0.95
                if otm_mask.sum() > 0:
                    rel_l2_otm += torch.sum(batch_rel_l2[otm_mask]).item()
                    n_otm += otm_mask.sum().item()
                    
                # ATM: 0.95 <= S/K <= 1.05
                atm_mask = (moneyness >= 0.95) & (moneyness <= 1.05)
                if atm_mask.sum() > 0:
                    rel_l2_atm += torch.sum(batch_rel_l2[atm_mask]).item()
                    n_atm += atm_mask.sum().item()
                    
                # ITM: S/K > 1.05
                itm_mask = moneyness > 1.05
                if itm_mask.sum() > 0:
                    rel_l2_itm += torch.sum(batch_rel_l2[itm_mask]).item()
                    n_itm += itm_mask.sum().item()

                total_data_loss += data_loss.item() * len(sigma)

                # Greeks accuracy (if available in batch)
                if 'Delta' in batch and 'Gamma' in batch:
                    Delta_true = batch['Delta'].to(self.device)
                    Gamma_true = batch['Gamma'].to(self.device)

                    with torch.enable_grad():
                        # Compute Greeks via AD at each sample's ATM point and middle time.
                        n_S = len(self.S_grid)
                        n_t = len(self.t_grid)
                        t_ref = self.t_grid[n_t // 2:n_t // 2 + 1]
                        
                        batch_size = sigma.shape[0]
                        S_atm_idx = torch.argmin(
                            (self.S_grid.unsqueeze(0) - K.unsqueeze(1)).abs(), dim=1
                        )
                        S_ref = self.S_grid[S_atm_idx].unsqueeze(1)
                        S_q = S_ref.detach().clone().requires_grad_(True)
                        t_q = t_ref.unsqueeze(0).expand(batch_size, -1).detach().clone().requires_grad_(True)
    
                        V_q = self.model.query(sigma, r, K_norm, T_norm, S_q, t_q,
                                               self.S_grid, self.t_grid)
    
                        # AD derivatives
                        dV_dS = torch.autograd.grad(V_q.sum(), S_q, create_graph=True, retain_graph=True)[0]
                        grad2 = torch.autograd.grad(
                            dV_dS[:, 0].sum(), S_q, create_graph=False, retain_graph=False
                        )[0]

                    # Extract at reference point (middle of S grid, middle of time grid)
                    Delta_pred = dV_dS  # (batch, 1)
                    Gamma_pred = grad2  # (batch, 1)

                    # True Greeks at same ATM S index and middle t.
                    batch_indices = torch.arange(batch_size, device=self.device)
                    Delta_true_ref = Delta_true[batch_indices, S_atm_idx, n_t // 2]
                    Gamma_true_ref = Gamma_true[batch_indices, S_atm_idx, n_t // 2]

                    delta_rmse = torch.sqrt(torch.mean(
                        (Delta_pred.view(-1) - Delta_true_ref) ** 2
                    ))
                    gamma_rmse = torch.sqrt(torch.mean(
                        (Gamma_pred.view(-1) - Gamma_true_ref) ** 2
                    ))

                    total_delta_rmse += delta_rmse.item() * len(sigma)
                    total_gamma_rmse += gamma_rmse.item() * len(sigma)

                n_batches += 1
                n_samples += len(sigma)

        total_rmse /= max(n_samples, 1)
        total_rel_l2 /= max(n_samples, 1)
        rel_l2_itm = rel_l2_itm / max(n_itm, 1) if n_itm > 0 else 0.0
        rel_l2_atm = rel_l2_atm / max(n_atm, 1) if n_atm > 0 else 0.0
        rel_l2_otm = rel_l2_otm / max(n_otm, 1) if n_otm > 0 else 0.0
        
        total_delta_rmse /= max(n_samples, 1)
        total_gamma_rmse /= max(n_samples, 1)

        # Average data loss across samples
        avg_data_loss = total_data_loss / max(n_samples, 1)

        return {
            'rmse': total_rmse,
            'rel_l2': total_rel_l2,
            'rel_l2_itm': rel_l2_itm,
            'rel_l2_atm': rel_l2_atm,
            'rel_l2_otm': rel_l2_otm,
            'delta_rmse': total_delta_rmse,
            'gamma_rmse': total_gamma_rmse,
            'val_data_loss': avg_data_loss,
        }

    def train(self, train_dataset, val_dataset, use_pde=True):
        """Full training loop.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
        val_dataset : torch.utils.data.Dataset
        use_pde : bool — if False, skip PDE loss entirely (data-only ablation)

        Returns
        -------
        history : dict — training history with all metrics
        """
        # DataLoader
        dl_kwargs = {
            'batch_size': self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            'num_workers': getattr(self.config, 'num_workers', 0),
            'pin_memory': getattr(self.config, 'pin_memory', False),
            'persistent_workers': getattr(self.config, 'persistent_workers', False),
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

        n_params = sum(p.numel() for p in self.model.parameters())
        patience = self.config.patience if hasattr(self.config, 'patience') else 25
        mode_label = "Physics-Informed (PDE)" if use_pde else "Data-Only"

        print(f"\n{'='*60}")
        print(f"Training FNO Option Pricer — {mode_label}")
        print(f"Device: {self.device}")
        print(f"Parameters: {n_params:,}")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"Patience: {patience}")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')
        patience_ctr = 0
        t0 = time.time()
        start_epoch = 0
        iteration = 0  # Track total batches processed

        n_epochs = self.config.n_epochs if hasattr(self.config, 'n_epochs') else 200

        # ============================================================
        # Resume from checkpoint if specified
        # ============================================================
        resume_path = getattr(self.config, 'resume_from_checkpoint', None)
        if resume_path and os.path.exists(resume_path):
            start_epoch, iteration = self.load_checkpoint(resume_path)
            best_val_loss = min(self.history.get('val_rmse', [float('inf')]))
            if best_val_loss == float('inf'):
                best_val_loss = float('inf')
            patience_ctr = 0

        # ============================================================
        # WandB initialization with resume support
        # ============================================================
        if hasattr(self.config, 'use_wandb') and self.config.use_wandb and _HAS_WANDB:
            wandb_run_id = getattr(self.config, 'wandb_run_id', None)
            wandb_resume = getattr(self.config, 'wandb_resume', 'allow')

            # If resuming and we have a run_id from checkpoint, use it
            if start_epoch > 0 and 'wandb_run_id' in self.history:
                wandb_run_id = self.history['wandb_run_id']

            # Initialize or resume WandB run
            if wandb_run_id and wandb_resume in ['allow', 'must']:
                run = wandb.init(
                    project=self.config.wandb_project,
                    id=wandb_run_id,
                    resume=wandb_resume,
                )
            else:
                run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.run_name,
                )
                # Store run ID for resume
                self.config.wandb_run_id = run.id
                self.history['wandb_run_id'] = run.id

            wandb.config.update({
                'n_epochs': n_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'optimizer': type(self.optimizer).__name__,
                'n_train_samples': len(train_dataset),
                'n_val_samples': len(val_dataset),
                'S_grid_size': getattr(self.config, 'S_grid_size', None),
                't_grid_size': getattr(self.config, 't_grid_size', None),
                'lambda_physics': None,
                'fno_modes': getattr(self.config, 'fno_modes', None),
                'fno_layers': getattr(self.config, 'fno_layers', None),
                'fno_width': getattr(self.config, 'fno_width', None),
                'random_seed': torch.initial_seed(),
                'resume_from_epoch': start_epoch,
            })

        for epoch in range(start_epoch, n_epochs):
            t_epoch = time.time()
            
            # LR schedule
            warmup_epochs = 5
            base_lr = self.config.learning_rate if hasattr(self.config, 'learning_rate') else 1e-3
            if epoch < warmup_epochs:
                lr = 1e-4 + (base_lr - 1e-4) * ((epoch + 1) / warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

            # Train (pass use_pde through to epoch)
            train_losses = self.train_epoch(train_loader, epoch, use_curriculum=True, use_pde=use_pde)

            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['rmse']
            epoch_time = time.time() - t_epoch

            # Record
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['val_loss'].append(val_loss)
            self.history['train_data_loss'].append(train_losses['data_loss'])
            self.history['train_pde_loss'].append(train_losses['pde_loss'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_delta_rmse'].append(val_metrics['delta_rmse'])
            self.history['val_gamma_rmse'].append(val_metrics['gamma_rmse'])
            # Record validation component losses
            self.history['val_data_loss'] = self.history.get('val_data_loss', [])
            self.history['val_data_loss'].append(val_metrics.get('val_data_loss', 0.0))

            # Epoch time
            self.history.setdefault('epoch_time_sec', []).append(epoch_time)
            self.history['lr'].append(lr)

            # WandB logging (if enabled)
            if hasattr(self.config, 'use_wandb') and self.config.use_wandb and _HAS_WANDB:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': self.history['train_loss'][-1],
                    'val_loss': val_loss,
                    'train_data_loss': train_losses['data_loss'],
                    'train_physics_loss': train_losses['pde_loss'],
                    'val_data_loss': val_metrics.get('val_data_loss', None),
                    'val_rmse': val_metrics['rmse'],
                    'val_delta_rmse': val_metrics['delta_rmse'],
                    'val_gamma_rmse': val_metrics['gamma_rmse'],
                    'lr': lr,
                    'w_data': self.history['weights_data'][-1] if len(self.history['weights_data'])>0 else None,
                    'w_pde': self.history['weights_pde'][-1] if len(self.history['weights_pde'])>0 else None,
                    'epoch_time_sec': epoch_time,
                })

            # Log
            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Train: {self.history['train_loss'][-1]:.6e} | "
                      f"Val RMSE: {val_metrics['rmse']:.6e} | "
                      f"Δ RMSE: {val_metrics['delta_rmse']:.6e} | "
                      f"Γ RMSE: {val_metrics['gamma_rmse']:.6e} | "
                      f"LR: {lr:.2e} | {elapsed:.0f}s")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr = 0
                self.save('best', epoch=epoch, iteration=iteration, is_best=True)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"\nEarly stop at epoch {epoch+1}")
                    break

            # Save latest checkpoint (for resume)
            save_latest_n = getattr(self.config, 'save_latest_every_epoch', 1)
            if (epoch + 1) % save_latest_n == 0:
                self.save('checkpoint_latest', epoch=epoch, iteration=iteration)

            # Save epoch checkpoint every N epochs
            save_every_n = getattr(self.config, 'save_every_n_epochs', 10)
            if (epoch + 1) % save_every_n == 0:
                epoch_name = f'checkpoint_epoch_{epoch+1:04d}'
                self.save(epoch_name, epoch=epoch, iteration=iteration)

            # Update iteration counter
            iteration += len(train_loader)

        self.save('final')
        # Final evaluation on validation set (as test proxy)
        final_metrics = self._final_evaluation(val_loader)
        # Save final metrics in history and log to wandb
        self.history['final_metrics'] = final_metrics
        if hasattr(self.config, 'use_wandb') and self.config.use_wandb and _HAS_WANDB:
            wandb.log(final_metrics)
        self._save_history()
        print(f"\nDone! Best val RMSE: {best_val_loss:.6e}")

        return self.history

    def save(self, name, epoch=None, iteration=None, is_best=False):
        """Save checkpoint with structured naming, compression, and optimized GCP upload.

        Parameters
        ----------
        name : str — 'latest', 'best', or 'epoch_{N}'
        epoch : int — current epoch number (for checkpoint metadata)
        iteration : int — current iteration (for checkpoint metadata)
        is_best : bool — whether this is the best model so far (triggers GCP upload)
        """
        d = self.config.checkpoint_dir if hasattr(self.config, 'checkpoint_dir') else './checkpoints'
        run_name = self.config.run_name if hasattr(self.config, 'run_name') else 'model'
        full_name = f'{run_name}_{name}.pt'
        path = os.path.join(d, full_name)

        # Determine checkpoint scope (slim vs full)
        # latest checkpoints: slim (resume only, no history)
        # best/periodic checkpoints: full (complete metadata)
        is_slim = (name == 'checkpoint_latest')
        keys_to_save = RESUME_KEYS if is_slim else FULL_KEYS

        checkpoint = {
            'epoch': epoch if epoch is not None else -1,
            'iteration': iteration if iteration is not None else -1,
            'run_name': run_name,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        # Add history and config only for full checkpoints
        if not is_slim:
            checkpoint['history'] = self.history
            checkpoint['config'] = {
                'S_grid_size': getattr(self.config, 'S_grid_size', None),
                't_grid_size': getattr(self.config, 't_grid_size', None),
                'fno_modes': getattr(self.config, 'fno_modes', None),
                'fno_layers': getattr(self.config, 'fno_layers', None),
                'fno_width': getattr(self.config, 'fno_width', None),
            }
        else:
            # Slim checkpoints: minimal config needed for resume
            checkpoint['config'] = {
                'S_grid_size': getattr(self.config, 'S_grid_size', None),
                't_grid_size': getattr(self.config, 't_grid_size', None),
                'fno_modes': getattr(self.config, 'fno_modes', None),
                'fno_layers': getattr(self.config, 'fno_layers', None),
                'fno_width': getattr(self.config, 'fno_width', None),
            }

        torch.save(checkpoint, path)

        # Upload to WandB as a file for easy access
        if hasattr(self.config, 'use_wandb') and self.config.use_wandb and _HAS_WANDB:
            try:
                wandb.save(path)
            except Exception:
                pass

        # GCP Upload with frequency control and compression
        gcp_configured = (hasattr(self.config, 'gcp_bucket_name') and self.config.gcp_bucket_name and
                         hasattr(self.config, 'gcp_service_account_path') and self.config.gcp_service_account_path)

        if gcp_configured and self._upload_queue is not None:
            upload_frequency = getattr(self.config, 'upload_latest_every_n_epochs', 15)
            upload_best = getattr(self.config, 'upload_best_always', True)
            use_compression = getattr(self.config, 'compress_checkpoints', True)

            should_upload = False
            gcp_path = full_name

            # Upload logic (all controlled by upload_frequency):
            # 1. best.pt: only upload if enough epochs passed since last upload
            # 2. checkpoint_latest.pt: upload every N epochs
            # 3. epoch checkpoints: upload when they're created
            last_upload = getattr(self, '_last_gcp_upload_epoch', -upload_frequency)
            epochs_since = (epoch + 1) - last_upload

            if is_best and upload_best and epochs_since >= upload_frequency:
                should_upload = True
            elif name == 'checkpoint_latest' and epoch is not None and (epoch + 1) % upload_frequency == 0:
                should_upload = True
            elif name.startswith('checkpoint_epoch_'):
                should_upload = True

            if should_upload:
                # Track last upload epoch (so best.pt doesn't upload every improvement)
                if is_best:
                    self._last_gcp_upload_epoch = epoch + 1
                # Compression for GCP upload (doesn't affect local file)
                if use_compression:
                    compressed_path = path + '.gz'
                    self._save_compressed_checkpoint(path, compressed_path)
                    upload_path = compressed_path
                    gcp_path = full_name + '.gz'
                else:
                    upload_path = path
                    gcp_path = full_name

                # Queue upload (non-blocking, backpressure via queue size)
                gcp_prefix = getattr(self.config, 'gcp_prefix', 'pinns')
                dest_blob = f"{gcp_prefix}/{run_name}/{gcp_path}"

                try:
                    self._upload_queue.put_nowait((upload_path, dest_blob))
                except Exception as e:
                    print(f"[GCP Upload Warning] Queue full, skipping upload: {e}")

    def _save_compressed_checkpoint(self, src_path, dest_path):
        """Save compressed checkpoint with gzip (level=1 for speed)."""
        try:
            with open(src_path, 'rb') as f:
                data = f.read()
            with gzip.open(dest_path, 'wb', compresslevel=1) as gz:
                gz.write(data)
        except Exception as e:
            # Fallback: copy original if compression fails
            import shutil
            shutil.copy2(src_path, dest_path.replace('.gz', ''))
            print(f"[Compression Warning] Failed: {e}, using uncompressed")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore model, optimizer, scheduler, and history.
        Supports both compressed (.gz) and uncompressed checkpoints.

        Parameters
        ----------
        checkpoint_path : str — path to checkpoint file

        Returns
        -------
        resume_epoch : int — epoch to resume from
        resume_iter : int — iteration to resume from
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"{'='*60}\n")

        # Handle compressed checkpoints
        if checkpoint_path.endswith('.gz'):
            with gzip.open(checkpoint_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
            checkpoint = torch.load(buffer, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model'])
        print(f"✓ Model state loaded")

        # Load optimizer state
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"✓ Optimizer state loaded")

        # Load scheduler state
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"✓ Scheduler state loaded")

        # Load history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            print(f"✓ Training history loaded ({len(self.history.get('train_loss', []))} epochs)")

        resume_epoch = checkpoint.get('epoch', -1) + 1  # +1 because we start from next epoch
        resume_iter = checkpoint.get('iteration', -1) + 1

        print(f"\nResuming from epoch {resume_epoch} (was at epoch {checkpoint.get('epoch', -1)})")
        print(f"Best val RMSE so far: {min(self.history['val_rmse']):.6e}" if self.history.get('val_rmse') else "No val RMSE history")
        print(f"{'='*60}\n")

        return resume_epoch, resume_iter

    def _final_evaluation(self, dataloader):
        """Compute final evaluation metrics on provided dataloader.

        Returns a dict matching the requested final-eval schema (approximate).
        """
        self.model.eval()
        all_rel_l2 = []
        all_abs_max = []
        preds = []
        trues = []
        times = []

        with torch.no_grad():
            for batch in dataloader:
                sigma = batch['sigma'].to(self.device)
                r = batch['r'].to(self.device)
                K = batch['K'].to(self.device)
                T = batch['T'].to(self.device)
                V_true = batch['V'].to(self.device)

                K_norm = K / 100.0
                T_norm = T / 2.0

                t0 = time.time()
                V_pred = self.model(sigma, r, K_norm, T_norm, self.S_grid, self.t_grid)
                t1 = time.time()

                times.append((t1 - t0) / max(len(sigma), 1))

                # relative L2 per sample
                batch_rel = torch.sqrt(torch.mean((V_pred - V_true) ** 2, dim=(1, 2))) / (
                    torch.sqrt(torch.mean(V_true ** 2, dim=(1, 2))) + 1e-12
                )
                all_rel_l2.extend(batch_rel.cpu().tolist())

                # max pointwise absolute error per sample
                batch_absmax = torch.max(torch.abs(V_pred - V_true), dim=2)[0]
                batch_absmax = torch.max(batch_absmax, dim=1)[0]
                all_abs_max.extend(batch_absmax.cpu().tolist())

                preds.append(V_pred.cpu())
                trues.append(V_true.cpu())

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        # Aggregate metrics
        test_relative_l2 = float(np.mean(all_rel_l2))
        max_pointwise_error = float(np.max(all_abs_max)) if len(all_abs_max) > 0 else 0.0
        fno_inference_time_ms = float(np.mean(times) * 1000.0) if len(times) > 0 else 0.0

        # R^2 score
        trues_flat = trues.reshape(trues.shape[0], -1).numpy()
        preds_flat = preds.reshape(preds.shape[0], -1).numpy()
        ss_res = np.sum((trues_flat - preds_flat) ** 2)
        ss_tot = np.sum((trues_flat - np.mean(trues_flat)) ** 2) + 1e-12
        r2_score = 1.0 - ss_res / ss_tot

        # Coarse sigma-split statistics if params exist in dataset
        # We'll attempt to use dataloader.dataset.params if present
        low_mean = mid_mean = high_mean = None
        try:
            params = dataloader.dataset.params
            sigmas = params[:, 0]
            rels = np.array(all_rel_l2)
            low_mask = (sigmas >= 0.1) & (sigmas < 0.3)
            mid_mask = (sigmas >= 0.3) & (sigmas < 0.6)
            high_mask = (sigmas >= 0.6) & (sigmas <= 0.8)
            if low_mask.sum() > 0:
                low_mean = float(np.mean(rels[low_mask]))
            if mid_mask.sum() > 0:
                mid_mean = float(np.mean(rels[mid_mask]))
            if high_mask.sum() > 0:
                high_mean = float(np.mean(rels[high_mask]))
        except Exception:
            pass

        return {
            'test_relative_l2': test_relative_l2,
            'test_relative_l2_low_sigma': low_mean,
            'test_relative_l2_mid_sigma': mid_mean,
            'test_relative_l2_high_sigma': high_mean,
            'max_pointwise_error': max_pointwise_error,
            'r2_score': r2_score,
            'fno_inference_time_ms': fno_inference_time_ms,
            'fdm_solve_time_ms': None,
            'speedup_factor': None,
        }

    def _save_history(self):
        """Save history + plots."""
        d = self.config.results_dir if hasattr(self.config, 'results_dir') else './results'
        np.save(os.path.join(d, 'history.npy'), self.history)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, self.history['val_rmse'])
        axes[0, 1].set_title('Price RMSE')

        axes[0, 2].plot(epochs, self.history['val_delta_rmse'], label='Delta')
        axes[0, 2].plot(epochs, self.history['val_gamma_rmse'], label='Gamma')
        axes[0, 2].set_title('Greeks RMSE')
        axes[0, 2].legend()

        axes[1, 0].plot(epochs, self.history['train_data_loss'], label='Data')
        axes[1, 0].plot(epochs, self.history['train_pde_loss'], label='PDE')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].legend()

        if len(self.history['weights_data']) > 0:
            w_len = min(len(self.history['weights_data']), len(epochs))
            axes[1, 1].plot(range(1, w_len+1), self.history['weights_data'][:w_len], label='w_data')
            axes[1, 1].plot(range(1, w_len+1), self.history['weights_pde'][:w_len], label='w_pde')
            axes[1, 1].set_title('Adaptive Weights')
            axes[1, 1].legend()

        axes[1, 2].plot(epochs, self.history['lr'])
        axes[1, 2].set_title('Learning Rate')

        plt.tight_layout()
        plt.savefig(os.path.join(d, 'training.png'), dpi=150)
        plt.close()


def load_model(config, path):
    """Load trained model. Supports both compressed (.gz) and uncompressed checkpoints."""
    device = torch.device(config.device if hasattr(config, 'device') else 'cuda')
    model = FNOOptionPricer(config).to(device)

    # Handle compressed checkpoints
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        ckpt = torch.load(buffer, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, device


def train_model(config):
    import h5py

    # Apply thread settings if provided
    if hasattr(config, 'num_threads') and config.num_threads > 0:
        try:
            torch.set_num_threads(config.num_threads)
        except Exception:
            pass
    if hasattr(config, 'num_interop_threads') and config.num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(config.num_interop_threads)
        except Exception:
            pass

    # Load data
    train_path = os.path.join(config.data_dir, 'train.h5')
    val_path = os.path.join(config.data_dir, 'val.h5')

    # Auto-detect data format and return appropriate Dataset class
    def _make_dataset(path):
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
        if 'params' not in keys:
            raise ValueError(
                f"Data file {path} uses the old format (separate sigma/r/K keys). "
                "The trainer requires the large-scale format with 'params (N,4)' key. "
                "Generate data with: python data_generator_large.py --n_samples 10000"
            )
        # Large-scale format (data_generator_large.py)
        class LargeDataset(torch.utils.data.Dataset):
            def __init__(self, p):
                with h5py.File(p, 'r') as fh:
                    self.params = fh['params'][:]
                    self.V = fh['V'][:]
                    self.Delta = fh['Delta'][:] if 'Delta' in fh else None
                    self.Gamma = fh['Gamma'][:] if 'Gamma' in fh else None
            def __len__(self):
                return len(self.params)
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
        return LargeDataset(path)

    train_ds = _make_dataset(train_path)
    val_ds = _make_dataset(val_path)

    # Initialize WandB if requested
    if hasattr(config, 'use_wandb') and config.use_wandb:
        if not _HAS_WANDB:
            raise RuntimeError('wandb package not available; install with `pip install wandb`')
        # Initialize run with a minimal config
        run = wandb.init(project=config.wandb_project if hasattr(config, 'wandb_project') else 'fno-option-pricer', config={
            'batch_size': config.batch_size,
            'lr': config.learning_rate,
            'epochs': config.n_epochs,
            'fno_modes': config.fno_modes,
            'fno_layers': config.fno_layers,
            'fno_width': config.fno_width,
        })
        # Override run_name with WandB run name
        config.run_name = run.name

    # Make DataLoader options explicit and configurable
    trainer = FNOTrainer(config)
    history = trainer.train(train_ds, val_ds)

    # Final artifact upload
    if hasattr(config, 'use_wandb') and config.use_wandb and _HAS_WANDB:
        try:
            wandb.save(os.path.join(config.results_dir, 'training.png'))
            wandb.save(os.path.join(config.results_dir, 'history.npy'))
            wandb.finish()
        except Exception:
            pass

    # GCP Upload of the models after training completes
    if hasattr(config, 'gcp_bucket_name') and config.gcp_bucket_name and \
       hasattr(config, 'gcp_service_account_path') and config.gcp_service_account_path:
        from utils import upload_to_gcp_bucket
        run_name = config.run_name if hasattr(config, 'run_name') else 'model'
        d = config.checkpoint_dir if hasattr(config, 'checkpoint_dir') else './checkpoints'
        gcp_prefix = getattr(config, 'gcp_prefix', 'pinns')
        use_compression = getattr(config, 'compress_checkpoints', True)

        for name in ['best', 'final', 'checkpoint_latest']:
            full_name = f'{run_name}_{name}.pt'
            path = os.path.join(d, full_name)
            if os.path.exists(path):
                # Compress if enabled
                if use_compression:
                    compressed_path = path + '.gz'
                    try:
                        import gzip
                        with open(path, 'rb') as f:
                            data = f.read()
                        with gzip.open(compressed_path, 'wb', compresslevel=1) as gz:
                            gz.write(data)
                        upload_path = compressed_path
                        dest_blob = f"{gcp_prefix}/{run_name}/{full_name}.gz"
                    except Exception as e:
                        print(f"[Compression Warning] {e}, uploading uncompressed")
                        upload_path = path
                        dest_blob = f"{gcp_prefix}/{run_name}/{full_name}"
                else:
                    upload_path = path
                    dest_blob = f"{gcp_prefix}/{run_name}/{full_name}"

                upload_to_gcp_bucket(
                    upload_path,
                    config.gcp_bucket_name,
                    dest_blob,
                    config.gcp_service_account_path,
                    storage_class=getattr(config, 'gcp_storage_class', 'STANDARD')
                )

    # Optional: Run full benchmark suite immediately after training
    if hasattr(config, 'run_benchmark') and config.run_benchmark:
        print(f"\n{'='*60}")
        print("STARTING AUTOMATIC BENCHMARK")
        print(f"{'='*60}")
        # Pass the newly created run_name to the benchmark
        run_full_benchmark(config)

    return history


# =============================================================================
# Standalone Checkpoint Utilities
# =============================================================================

def load_checkpoint_for_resume(checkpoint_path, device=None):
    """
    Load a checkpoint and return its metadata for inspection or manual resume.
    Supports both compressed (.gz) and uncompressed checkpoints.

    Parameters
    ----------
    checkpoint_path : str — path to checkpoint file
    device : torch.device — device to load tensors on

    Returns
    -------
    checkpoint : dict — full checkpoint data
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Handle compressed checkpoints
    if checkpoint_path.endswith('.gz'):
        import gzip
        with gzip.open(checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        checkpoint = torch.load(buffer, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"\n{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"Run name:     {checkpoint.get('run_name', 'N/A')}")
    print(f"Epoch:        {checkpoint.get('epoch', -1)}")
    print(f"Iteration:    {checkpoint.get('iteration', -1)}")
    print(f"Best val RMSE: {min(checkpoint.get('history', {}).get('val_rmse', [float('inf')])):.6e}")
    print(f"Config:       {checkpoint.get('config', {})}")
    print(f"{'='*60}\n")

    return checkpoint


def inspect_checkpoint(checkpoint_path, device=None):
    """
    Quick utility to inspect a checkpoint without loading full state.

    Parameters
    ----------
    checkpoint_path : str
    device : torch.device
    """
    cp = load_checkpoint_for_resume(checkpoint_path, device)
    return {
        'run_name': cp.get('run_name'),
        'epoch': cp.get('epoch'),
        'iteration': cp.get('iteration'),
        'history_length': len(cp.get('history', {}).get('train_loss', [])),
        'config': cp.get('config', {}),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    from config import Config
    train_model(Config())
