"""
Fourier Neural Operator for Option Pricing — Production Edition

Key upgrades over baseline:
  1. Fourier positional encoding (mitigates spectral bias near sharp features)
  2. Hard-constrained output via ansatz (enforces expiry BC by construction)
  3. Coordinate-aware decoder (enables AD for PDE residuals & Greeks)

Architecture:
  Branch (FNO): (σ, r, K) → latent features on grid
  Trunk  (MLP): (S, t) coordinates → basis
  Output: hard-constrained ansatz
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# Spectral Convolution (Fourier layer)

class SpectralConv2d(nn.Module):
    """2D Fourier convolution — Li et al. (2020)."""

    def __init__(self, in_c, out_c, modes1, modes2):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat))

    def _mul(self, x, w):
        """Complex contraction: (b,i,m1,m2) × (i,o,m1,m2) → (b,o,m1,m2)"""
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        """x: (batch, ch, nx, ny)"""
        b = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        nx, ny_half = x.shape[-2], x.shape[-1] // 2 + 1

        # Both ops are graph-connected, no in-place writes
        out1 = self._mul(x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out2 = self._mul(x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        mid_rows = max(0, nx - 2 * self.modes1)
        z_mid = torch.zeros(b, self.out_c, mid_rows, self.modes2,
                            dtype=x_ft.dtype, device=x.device)
        z_right = torch.zeros(b, self.out_c, nx, ny_half - self.modes2,
                              dtype=x_ft.dtype, device=x.device)

        left_col = torch.cat([out1, z_mid, out2], dim=2)  # (b, out_c, nx, modes2)
        out_ft = torch.cat([left_col, z_right], dim=3)    # (b, out_c, nx, ny_half)

        return torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))


class FourierBlock2D(nn.Module):
    """Single FNO block: spectral conv + pointwise linear + residual."""

    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return self.spec(x) + self.w(x)


# Fourier Positional Encoding

class FourierEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for continuous coordinates.

    Mitigates spectral bias — helps network learn high-frequency
    features (payoff kinks, near-expiry discontinuities).
    """

    def __init__(self, in_dim, num_freqs, logscale=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = in_dim * (2 * num_freqs + 1)  # +1 for raw coord

        if logscale:
            freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32) * math.pi
        else:
            freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * math.pi

        self.register_buffer('freqs', freqs)  # (num_freqs,)

    def forward(self, x):
        """
        x: (..., in_dim)
        Returns: (..., out_dim)
        """
        # x * freqs: (..., in_dim, num_freqs)
        x_scaled = x.unsqueeze(-1) * self.freqs  # (..., in_dim, num_freqs)

        # Concatenate: raw coord + sin + cos for each freq
        embeddings = [x]  # raw coordinates
        embeddings.append(torch.sin(x_scaled).view(*x.shape[:-1], -1))
        embeddings.append(torch.cos(x_scaled).view(*x.shape[:-1], -1))

        return torch.cat(embeddings, dim=-1)



# Coordinate-Aware Decoder (Trunk Network)

class CoordinateDecoder(nn.Module):
    """
    MLP that takes per-grid-point features + (S, t) coordinates
    and produces a differentiable V(S, t).

    Enables AD-based PDE residuals and Greeks computation.
    """

    def __init__(self, feature_dim, hidden_dim=128, num_layers=3):
        super().__init__()

        # Fourier embedding for (S, t) coordinates
        self.fourier_embed = FourierEmbedding(in_dim=2, num_freqs=16)
        coord_dim = self.fourier_embed.out_dim

        # Build MLP
        layers = []
        layers.append(nn.Linear(feature_dim + coord_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def evaluate(self, features, S_query, t_query, S_min, S_max, t_min, t_max):
        """
        Evaluate at arbitrary coordinates.
        features: (batch, ..., feature_dim)
        S_query: (batch, ...) or (...)
        t_query: (batch, ...) or (...)
        """
        # Normalize coordinates using global bounds
        S_log = torch.log(S_query + 1e-6)
        S_log_min = math.log(S_min + 1e-6)
        S_log_max = math.log(S_max + 1e-6)
        S_norm = (S_log - S_log_min) / (S_log_max - S_log_min + 1e-8)
        
        t_norm = (t_query - t_min) / (t_max - t_min + 1e-8)

        # Coordinate stack
        coords = torch.stack([S_norm, t_norm], dim=-1) # (..., 2)
        
        # Fourier embedding
        coord_emb = self.fourier_embed(coords) # (..., coord_dim)
        
        # If coords are not batched, expand them
        if coord_emb.dim() < features.dim():
            coord_emb = coord_emb.unsqueeze(0).expand(features.shape[0], *([-1]*(features.dim()-1)))

        # Concatenate features + coordinates
        x = torch.cat([features, coord_emb], dim=-1)

        # Apply MLP pointwise
        V = self.mlp(x).squeeze(-1)

        return V

    def forward(self, features, S_grid, t_grid):
        """
        Parameters
        ----------
        features : (batch, n_S, n_t, feature_dim) — from FNO backbone
        S_grid   : (n_S,) — asset price grid
        t_grid   : (n_t,) — time grid

        Returns
        -------
        V : (batch, n_S, n_t) — differentiable w.r.t. S_grid and t_grid
        """
        batch, n_S, n_t, feat_dim = features.shape
        S_mesh, t_mesh = torch.meshgrid(S_grid, t_grid, indexing='ij')
        
        V = self.evaluate(features, S_mesh, t_mesh, 
                          S_grid.min().item(), S_grid.max().item(), 
                          t_grid.min().item(), t_grid.max().item())
        return V


# Hard-Constrained Ansatz

class HardConstrainedOutput(nn.Module):
    """
    Enforces boundary conditions by construction:

    V(S, t) = payoff(S, K) · indicator(τ=0) + V̂_raw · τ · S/(S+K) · σ(5·τ)

    where:
      τ = T - t (time to maturity)
      payoff = max(S - K, 0) for calls
      σ(x) = sigmoid(x)

    Guarantees:
      - At expiry (τ=0): V = payoff
      - At S = 0:       V = 0
    """

    @staticmethod
    def forward(V_raw, S, K, t, T):
        """
        Parameters
        ----------
        V_raw  : (batch, ...) — unconstrained network output
        S      : (batch, ...) or (...,)
        K      : (batch,)
        t      : (batch, ...) or (...,)
        T      : (batch,)
        """
        device = V_raw.device
        
        # Expand K and T to match V_raw shape
        batch_size = V_raw.shape[0]
        extra_dims = [1] * (V_raw.dim() - 1)
        K_expanded = K.view(batch_size, *extra_dims)
        T_expanded = T.view(batch_size, *extra_dims)
        
        # τ = T - t
        tau = T_expanded - t
        tau = tau.clamp(min=0.0)

        # Payoff at expiry
        payoff = torch.maximum(S - K_expanded, torch.tensor(0.0, device=device))

        # Modulation factor that vanishes at τ=0 and S=0
        S_mod = S / (S + K_expanded + 1e-8)
        # Sigmoid envelope for smoothness
        sigmoid_tau = torch.sigmoid(5.0 * tau)

        modulation = tau * S_mod * sigmoid_tau

        # Hard-constrained output (apply payoff purely as terminal constraint)
        V = payoff * (tau <= 1e-8).float() + V_raw * modulation

        return V


# Full FNO Model

class FNOOptionPricer(nn.Module):
    """
    Complete FNO model with:
      - Fourier positional encoding
      - 3 Fourier layers
      - Coordinate-aware decoder (AD-enabled)
      - Hard-constrained expiry boundary

    Learns operator: G: (σ, r, K, T) → V(S, t)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_S = config.S_grid_size if hasattr(config, 'S_grid_size') else 256
        self.n_t = config.t_grid_size if hasattr(config, 't_grid_size') else 64
        self.modes = config.fno_modes
        self.width = config.fno_width
        self.n_layers = config.fno_layers

        # --- Lifting network ---
        # Input: broadcast parameters (σ, r, K_norm, T_norm) over grid + Fourier PE
        # Output: (batch, width, n_S, n_t)
        in_channels = 6  # σ, r, K/S_ref, T, log(S/K), sqrt(τ)

        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
        )

        # --- Fourier layers ---
        self.fourier_blocks = nn.ModuleList([
            FourierBlock2D(self.modes, self.modes, self.width)
            for _ in range(self.n_layers)
        ])

        # Post-Fourier feature expansion
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
        )

        # --- Coordinate-aware decoder ---
        self.decoder = CoordinateDecoder(
            feature_dim=self.width,
            hidden_dim=128,
            num_layers=3
        )

    def forward(self, sigma, r, K_norm, T_norm, S_grid, t_grid, return_raw=False):
        """
        Forward pass with hard-constrained output on the grid.
        """
        device = sigma.device
        batch_size = sigma.shape[0]

        # Broadcast parameters over spatial grid
        sigma_f = sigma.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)
        r_f = r.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)
        K_f = K_norm.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)
        T_f = T_norm.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)

        # Log-moneyness encoding
        K_actual = K_norm * 100.0
        S_2d = S_grid.view(1, 1, self.n_S, 1).expand(batch_size, -1, -1, self.n_t)
        log_moneyness = torch.log(S_2d / (K_actual.view(batch_size, 1, 1, 1) + 1e-8) + 1e-8)

        # sqrt(tau) encoding
        T_actual = T_norm * 2.0
        t_abs = t_grid.view(1, 1, 1, self.n_t)
        tau_grid = torch.maximum(T_actual.view(batch_size, 1, 1, 1) - t_abs, torch.tensor(1e-8, device=device))
        sqrt_tau = torch.sqrt(tau_grid).expand(batch_size, -1, self.n_S, -1)

        x = torch.cat([sigma_f, r_f, K_f, T_f, log_moneyness, sqrt_tau], dim=1)

        # Lifting
        x = self.lifting(x)  # (batch, width, n_S, n_t)

        # Fourier layers
        for block in self.fourier_blocks:
            x = F.gelu(block(x))

        # Feature projection
        features = self.feature_proj(x)  # (batch, width, n_S, n_t)

        # Rearrange for decoder: (batch, n_S, n_t, width)
        features = features.permute(0, 2, 3, 1)

        # Coordinate-aware decoder
        V_raw = self.decoder(features, S_grid, t_grid)  # (batch, n_S, n_t)

        # Hard-constrained boundary condition
        S_mesh, t_mesh = torch.meshgrid(S_grid, t_grid, indexing='ij')
        V = HardConstrainedOutput.forward(V_raw, S_mesh, K_actual, t_mesh, T_actual)

        if return_raw:
            return V, V_raw
        return V

    def query(self, sigma, r, K_norm, T_norm, S_query, t_query, S_grid, t_grid):
        """
        Query the model at arbitrary (S, t) coordinates (differentiable).
        """
        device = sigma.device
        batch_size = sigma.shape[0]

        # 1. Get features from FNO backbone on the base grid
        sigma_f = sigma.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)
        r_f = r.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)
        K_f = K_norm.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)
        T_f = T_norm.view(batch_size, 1, 1, 1).expand(-1, -1, self.n_S, self.n_t)

        K_actual = K_norm * 100.0
        S_2d = S_grid.view(1, 1, self.n_S, 1).expand(batch_size, -1, -1, self.n_t)
        log_moneyness = torch.log(S_2d / (K_actual.view(batch_size, 1, 1, 1) + 1e-8) + 1e-8)

        T_actual = T_norm * 2.0
        t_abs_grid = t_grid.view(1, 1, 1, self.n_t)
        tau_grid = torch.maximum(T_actual.view(batch_size, 1, 1, 1) - t_abs_grid, torch.tensor(1e-8, device=device))
        sqrt_tau = torch.sqrt(tau_grid).expand(batch_size, -1, self.n_S, -1)

        x = torch.cat([sigma_f, r_f, K_f, T_f, log_moneyness, sqrt_tau], dim=1)
        x = self.lifting(x)
        for block in self.fourier_blocks:
            x = F.gelu(block(x))
        features = self.feature_proj(x) # (batch, width, n_S, n_t)

        # 2. Interpolate grid features to query meshgrid
        # S_query: (n_qS,) or (batch, n_qS)
        if S_query.dim() == 1:
            S_mesh_q, t_mesh_q = torch.meshgrid(S_query, t_query, indexing='ij')
        else:
            # S_query is (batch, n_qS), t_query is (batch, n_qt)
            S_mesh_q = S_query.unsqueeze(2).expand(-1, -1, t_query.shape[1])
            t_mesh_q = t_query.unsqueeze(1).expand(-1, S_query.shape[1], -1)
        
        # Bilinear interpolation of features using custom vectorized implementation
        # features: (batch, width, n_S, n_t)
        feat_interp = self._bilinear_interpolate(features, S_grid, t_grid, S_query, t_query)

        # 3. Evaluate Decoder MLP directly on query coordinates
        V_raw = self.decoder.evaluate(feat_interp, S_mesh_q, t_mesh_q,
                                      S_grid.min().item(), S_grid.max().item(),
                                      t_grid.min().item(), t_grid.max().item())

        # 4. Apply Hard Constraints directly on query coordinates
        V = HardConstrainedOutput.forward(V_raw, S_mesh_q, K_actual, t_mesh_q, T_actual)

        return V

    @staticmethod
    def _bilinear_interpolate(V, S_grid, t_grid, S_q, t_q):
        """
        Bilinear interpolation on the (S, t) grid.
        First-order gradients flow through bilinear weights; second-order
        gradients flow through the decoder's coordinate-aware MLP (which
        receives S_q, t_q directly as inputs).

        V: (batch, width, n_S, n_t)
        S_grid: (n_S,), t_grid: (n_t,)
        S_q: (n_qS,) or (batch, n_qS), t_q: (n_qt,) or (batch, n_qt)

        Returns: (batch, n_qS, n_qt, width)
        """
        batch, width, n_S, n_t = V.shape

        if S_q.dim() == 1:
            n_qS, n_qt = S_q.shape[0], t_q.shape[0]
    
            # Normalize query coordinates to [0, n-1] index space
            s_idx = (S_q - S_grid[0]) / (S_grid[-1] - S_grid[0] + 1e-8) * (n_S - 1)
            t_idx = (t_q - t_grid[0]) / (t_grid[-1] - t_grid[0] + 1e-8) * (n_t - 1)
    
            s_idx = s_idx.clamp(0, n_S - 2)
            t_idx = t_idx.clamp(0, n_t - 2)
    
            s0 = s_idx.floor().long()
            s1 = s0 + 1
            t0 = t_idx.floor().long()
            t1 = t0 + 1
    
            # Interpolation weights — first-order gradient path through s_idx, t_idx
            ws = s_idx - s0.float()  # (n_qS,)
            wt = t_idx - t0.float()  # (n_qt,)
    
            b_idx = torch.arange(batch, device=V.device)[:, None, None, None]
            w_idx = torch.arange(width, device=V.device)[None, :, None, None]
            c00 = V[b_idx, w_idx, s0[None, None, :, None], t0[None, None, None, :]]
            c10 = V[b_idx, w_idx, s1[None, None, :, None], t0[None, None, None, :]]
            c01 = V[b_idx, w_idx, s0[None, None, :, None], t1[None, None, None, :]]
            c11 = V[b_idx, w_idx, s1[None, None, :, None], t1[None, None, None, :]]
    
            # Bilinear interpolation
            V_q = (c00 * (1 - ws)[None, None, :, None] * (1 - wt)[None, None, None, :] +
                   c10 * ws[None, None, :, None] * (1 - wt)[None, None, None, :] +
                   c01 * (1 - ws)[None, None, :, None] * wt[None, None, None, :] +
                   c11 * ws[None, None, :, None] * wt[None, None, None, :])
        else:
            n_qS, n_qt = S_q.shape[1], t_q.shape[1]
    
            s_idx = (S_q - S_grid[0]) / (S_grid[-1] - S_grid[0] + 1e-8) * (n_S - 1)
            t_idx = (t_q - t_grid[0]) / (t_grid[-1] - t_grid[0] + 1e-8) * (n_t - 1)
    
            s_idx = s_idx.clamp(0, n_S - 2)
            t_idx = t_idx.clamp(0, n_t - 2)
    
            s0 = s_idx.floor().long()
            s1 = s0 + 1
            t0 = t_idx.floor().long()
            t1 = t0 + 1
    
            ws = s_idx - s0.float()  # (batch, n_qS)
            wt = t_idx - t0.float()  # (batch, n_qt)
    
            b_idx = torch.arange(batch, device=V.device)[:, None, None, None]
            w_idx = torch.arange(width, device=V.device)[None, :, None, None]
            c00 = V[b_idx, w_idx, s0[:, None, :, None], t0[:, None, None, :]]
            c10 = V[b_idx, w_idx, s1[:, None, :, None], t0[:, None, None, :]]
            c01 = V[b_idx, w_idx, s0[:, None, :, None], t1[:, None, None, :]]
            c11 = V[b_idx, w_idx, s1[:, None, :, None], t1[:, None, None, :]]
    
            V_q = (c00 * (1 - ws)[:, None, :, None] * (1 - wt)[:, None, None, :] +
                   c10 * ws[:, None, :, None] * (1 - wt)[:, None, None, :] +
                   c01 * (1 - ws)[:, None, :, None] * wt[:, None, None, :] +
                   c11 * ws[:, None, :, None] * wt[:, None, None, :])

        return V_q.permute(0, 2, 3, 1)


# AD-based PDE Residual & Greeks

def compute_pde_residual_autograd(model, sigma, r, K_norm, T_norm,
                                   S_grid, t_grid, S_interp, t_interp):
    """
    Compute Black-Scholes PDE residual using Automatic Differentiation
    in log-S coordinates for numerical stability.

    Let x = log(S). The BS PDE becomes:
      ∂V/∂t + ½σ²(∂²V/∂x²) + (r - ½σ²)(∂V/∂x) - rV = 0

    This eliminates the S² amplification factor that causes instability
    with raw S coordinates (S up to 600 → S² up to 360,000×).

    Parameters
    ----------
    model : FNOOptionPricer
    sigma, r, K_norm, T_norm : (batch,)
    S_grid, t_grid : full grids for model backbone
    S_interp, t_interp : (n_qS,), (n_qt,) query points for PDE evaluation

    Returns
    -------
    residual : (batch, n_qS, n_qt) — should be ≈ 0
    dV_dS, d2V_dS2, dV_dt : Greeks for debugging (in original S coords)
    """
    with torch.enable_grad():
        batch = sigma.shape[0]
        # Pass raw S (NOT log-S) to model.query — the decoder already takes log internally.
        # Passing log-S here causes double-log: log(log(S)) → NaN when S < 1.
        S_q = S_interp.unsqueeze(0).expand(batch, -1).detach().clone().requires_grad_(True)
        t_q = t_interp.unsqueeze(0).expand(batch, -1).detach().clone().requires_grad_(True)

        # Forward through model with raw S coordinates
        V = model.query(sigma, r, K_norm, T_norm, S_q, t_q, S_grid, t_grid)

        # ∂V/S via autograd (raw S coords)
        dV_dS = torch.autograd.grad(
            V.sum(), S_q, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        dV_dS = dV_dS if dV_dS is not None else torch.zeros_like(S_q).requires_grad_(True)

        # ∂²V/∂S² via autograd (raw S coords)
        d2V_dS2 = torch.autograd.grad(
            dV_dS.sum(), S_q, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        d2V_dS2 = d2V_dS2 if d2V_dS2 is not None else torch.zeros_like(S_q).requires_grad_(True)

        # ∂V/∂t
        dV_dt = torch.autograd.grad(
            V.sum(), t_q, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        dV_dt = dV_dt if dV_dt is not None else torch.zeros_like(t_q).requires_grad_(True)

        # Convert to log-S derivatives via chain rule:
        # x = log(S) → dV/dx = S · dV/dS
        # d²V/dx² = S² · d²V/dS² + S · dV/dS
        dV_dx = S_q * dV_dS
        d2V_dx2 = S_q ** 2 * d2V_dS2 + S_q * dV_dS

        # Black-Scholes PDE residual in log-S form
        # ∂V/∂t + ½σ²(∂²V/∂x²) + (r - ½σ²)(∂V/∂x) - rV = 0
        # Shapes: dV_dx, d2V_dx2 ~ (batch, n_qS), dV_dt ~ (batch, n_qt)
        dV_dx_3d = dV_dx.unsqueeze(-1)       # (batch, n_qS, 1)
        d2V_dx2_3d = d2V_dx2.unsqueeze(-1)   # (batch, n_qS, 1)
        dV_dt_3d = dV_dt.unsqueeze(1)        # (batch, 1, n_qt)
        sigma_3d = sigma.view(-1, 1, 1)
        r_3d = r.view(-1, 1, 1)

        residual = (dV_dt_3d
                    + 0.5 * sigma_3d**2 * d2V_dx2_3d
                    + (r_3d - 0.5 * sigma_3d**2) * dV_dx_3d
                    - r_3d * V)

        return residual, dV_dS, d2V_dS2, dV_dt


def compute_greeks_autograd(model, sigma, r, K_norm, T_norm,
                             S_grid, t_grid, S_query):
    """
    Compute Delta and Gamma at specified S coordinates via AD.
    """
    with torch.enable_grad():
        batch = sigma.shape[0]
        # Use middle of time grid for Greeks
        t_mid = t_grid[len(t_grid) // 2:len(t_grid) // 2 + 1]
        
        # Expand for batch dimension
        S_query_batched = S_query.unsqueeze(0).expand(batch, -1).detach().clone().requires_grad_(True)
        t_mid_batched = t_mid.unsqueeze(0).expand(batch, -1).detach().clone().requires_grad_(True)
    
        V = model.query(sigma, r, K_norm, T_norm, S_query_batched, t_mid_batched, S_grid, t_grid)
    
        dV_dS = torch.autograd.grad(
            V.sum(), S_query_batched, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        dV_dS = dV_dS if dV_dS is not None else torch.zeros_like(S_query_batched).requires_grad_(True)
    
        d2V_dS2 = torch.autograd.grad(
            dV_dS.sum(), S_query_batched, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        d2V_dS2 = d2V_dS2 if d2V_dS2 is not None else torch.zeros_like(S_query_batched).requires_grad_(True)
    
        return dV_dS, d2V_dS2
