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
        out_ft = torch.zeros_like(x_ft)

        # Low-frequency block (top-left)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self._mul(x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        # Low-frequency block (top-right in complex-conjugate symmetric sense)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._mul(x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

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

        # Normalize coordinates to [0, 1] for stable Fourier embedding
        # Use log-price transform for S to better resolve OTM regions
        S_log = torch.log(S_grid + 1e-6)
        S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min() + 1e-8)
        
        t_norm = (t_grid - t_grid.min()) / (t_grid.max() - t_grid.min() + 1e-8)

        # Coordinate meshgrid
        S_mesh, t_mesh = torch.meshgrid(S_norm, t_norm, indexing='ij')  # (n_S, n_t)
        coords = torch.stack([S_mesh, t_mesh], dim=-1)  # (n_S, n_t, 2)

        # Fourier embedding of coordinates
        coord_emb = self.fourier_embed(coords)  # (n_S, n_t, coord_dim)

        # Broadcast coords across batch
        coord_emb = coord_emb.unsqueeze(0).expand(batch, -1, -1, -1)  # (batch, n_S, n_t, coord_dim)

        # Concatenate features + coordinates
        x = torch.cat([features, coord_emb], dim=-1)  # (batch, n_S, n_t, feat_dim + coord_dim)

        # Apply MLP pointwise
        V = self.mlp(x).squeeze(-1)  # (batch, n_S, n_t)

        return V


# Hard-Constrained Ansatz

class HardConstrainedOutput(nn.Module):
    """
    Enforces boundary conditions by construction:

    V(S, t) = payoff(S, K) · exp(-τ) + V̂_raw · τ · S/(S+K) · σ(5·τ)

    where:
      τ = T - t (time to maturity)
      payoff = max(S - K, 0) for calls
      σ(x) = sigmoid(x)

    Guarantees:
      - At expiry (τ=0): V = payoff
      - At S = 0:       V = 0
    """

    @staticmethod
    def forward(V_raw, S_grid, K, t_grid, T):
        """
        Parameters
        ----------
        V_raw  : (batch, n_S, n_t) — unconstrained network output
        S_grid : (n_S,)
        K      : (batch,)
        t_grid : (n_t,)
        T      : (batch,)

        Returns
        -------
        V : (batch, n_S, n_t)
        """
        device = V_raw.device
        n_S, n_t = V_raw.shape[1], V_raw.shape[2]

        # τ = T - t  →  (batch, n_t)
        tau = T[:, None] - t_grid[None, :]  # (batch, n_t)
        tau = tau.clamp(min=0.0)

        # Payoff at expiry: (batch, n_S, 1)
        S_2d = S_grid.view(1, n_S, 1)        # (1, n_S, 1)
        K_3d = K.view(-1, 1, 1)               # (batch, 1, 1)
        payoff = torch.maximum(S_2d - K_3d, torch.tensor(0.0, device=device))

        # Modulation factor that vanishes at τ=0 and S=0
        # τ: (batch, 1, n_t)
        tau_3d = tau[:, None, :]
        # S/(S+K): (1, n_S, 1)
        S_mod = S_2d / (S_2d + K_3d + 1e-8)
        # Sigmoid envelope for smoothness
        sigmoid_tau = torch.sigmoid(5.0 * tau_3d)

        modulation = tau_3d * S_mod * sigmoid_tau  # (batch, n_S, n_t)

        # Hard-constrained output
        V = payoff + V_raw * modulation

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
        in_channels = 4  # σ, r, K/S_ref, T

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
        Forward pass with hard-constrained output.

        Parameters
        ----------
        sigma    : (batch,) volatility
        r        : (batch,) risk-free rate
        K_norm   : (batch,) normalized strike (K / S_ref)
        T_norm   : (batch,) normalized maturity (T / T_max)
        S_grid   : (n_S,) asset price grid
        t_grid   : (n_t,) time grid
        return_raw : bool — if True, also return unconstrained V̂_raw

        Returns
        -------
        V : (batch, n_S, n_t) — hard-constrained pricing surface
        V_raw : (batch, n_S, n_t) — only if return_raw=True
        """
        device = sigma.device
        batch_size = sigma.shape[0]

        # Broadcast parameters over spatial grid
        # (batch, 4, n_S, n_t)
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
        tau = torch.maximum(T_actual.view(batch_size, 1, 1, 1) - t_abs, torch.tensor(1e-8, device=device))
        sqrt_tau = torch.sqrt(tau).expand(batch_size, -1, self.n_S, -1)

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
        K_actual = K_norm * 100.0  # Assuming S_ref = 100
        T_actual = T_norm * 2.0    # Assuming T_max = 2.0
        V = HardConstrainedOutput.forward(V_raw, S_grid, K_actual, t_grid, T_actual)

        if return_raw:
            return V, V_raw
        return V

    def query(self, sigma, r, K_norm, T_norm, S_query, t_query, S_grid, t_grid):
        """
        Query the model at arbitrary (S, t) coordinates (differentiable).

        This enables AD-based PDE residual and Greek computation at
        any point, not just grid nodes.

        Parameters
        ----------
        sigma, r, K_norm, T_norm : (batch,)
        S_query : (n_qS,) query S coordinates
        t_query : (n_qt,) query t coordinates
        S_grid, t_grid : full grids for FNO backbone

        Returns
        -------
        V : (batch, n_qS, n_qt)
        """
        # Get full-surface prediction
        V_full = self.forward(sigma, r, K_norm, T_norm, S_grid, t_grid)

        # Bilinear interpolation to query points
        V = self._bilinear_interpolate(V_full, S_grid, t_grid, S_query, t_query)
        return V

    @staticmethod
    def _bilinear_interpolate(V, S_grid, t_grid, S_q, t_q):
        """
        Bilinear interpolation on the (S, t) grid.
        V: (batch, n_S, n_t)
        S_grid: (n_S,), t_grid: (n_t,)
        S_q: (n_qS,), t_q: (n_qt,)

        Returns: (batch, n_qS, n_qt)
        """
        batch, n_S, n_t = V.shape
        n_qS, n_qt = len(S_q), len(t_q)

        # Normalize query coordinates to [0, n-1] index space
        s_idx = (S_q - S_grid[0]) / (S_grid[-1] - S_grid[0] + 1e-8) * (n_S - 1)
        t_idx = (t_q - t_grid[0]) / (t_grid[-1] - t_grid[0] + 1e-8) * (n_t - 1)

        s_idx = s_idx.clamp(0, n_S - 2)
        t_idx = t_idx.clamp(0, n_t - 2)

        s0 = s_idx.floor().long()
        s1 = s0 + 1
        t0 = t_idx.floor().long()
        t1 = t0 + 1

        # Interpolation weights
        ws = s_idx - s0.float()  # (n_qS,)
        wt = t_idx - t0.float()  # (n_qt,)

        # Gather corner values: V[batch, s, t]
        # Shape: (batch, n_qS, n_qt)
        c00 = V[torch.arange(batch)[:, None, None], s0[:, None], t0[None, :]]
        c10 = V[torch.arange(batch)[:, None, None], s1[:, None], t0[None, :]]
        c01 = V[torch.arange(batch)[:, None, None], s0[:, None], t1[None, :]]
        c11 = V[torch.arange(batch)[:, None, None], s1[:, None], t1[None, :]]

        # Bilinear interpolation
        V_q = (c00 * (1 - ws)[:, None] * (1 - wt)[None, :] +
               c10 * ws[:, None] * (1 - wt)[None, :] +
               c01 * (1 - ws)[:, None] * wt[None, :] +
               c11 * ws[:, None] * wt[None, :])

        return V_q


# AD-based PDE Residual & Greeks

def compute_pde_residual_autograd(model, sigma, r, K_norm, T_norm,
                                   S_grid, t_grid, S_interp, t_interp):
    """
    Compute Black-Scholes PDE residual using Automatic Differentiation.

    PDE: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

    Parameters
    ----------
    model : FNOOptionPricer
    sigma, r, K_norm, T_norm : (batch,)
    S_grid, t_grid : full grids for model backbone
    S_interp, t_interp : (n_qS,), (n_qt,) query points for PDE evaluation

    Returns
    -------
    residual : (batch, n_qS, n_qt) — should be ≈ 0
    dV_dS, d2V_dS2, dV_dt : Greeks for debugging
    """
    # Enable gradients for query coordinates
    S_q = S_interp.detach().clone().requires_grad_(True)
    t_q = t_interp.detach().clone().requires_grad_(True)

    # Forward through model
    V = model.query(sigma, r, K_norm, T_norm, S_q, t_q, S_grid, t_grid)

    # ∂V/∂S
    dV_dS = torch.autograd.grad(
        V.sum(), S_q, create_graph=True, retain_graph=True
    )[0]

    # ∂²V/∂S²
    d2V_dS2 = torch.autograd.grad(
        dV_dS.sum(), S_q, create_graph=True, retain_graph=True
    )[0]

    # ∂V/∂t
    dV_dt = torch.autograd.grad(
        V.sum(), t_q, create_graph=True, retain_graph=True
    )[0]

    # Black-Scholes PDE residual
    S_2d = S_q.view(1, -1, 1)      # (1, n_qS, 1)
    sigma_3d = sigma.view(-1, 1, 1)
    r_3d = r.view(-1, 1, 1)

    residual = dV_dt + 0.5 * sigma_3d**2 * S_2d**2 * d2V_dS2 + \
               r_3d * S_2d * dV_dS - r_3d * V

    return residual, dV_dS, d2V_dS2, dV_dt


def compute_greeks_autograd(model, sigma, r, K_norm, T_norm,
                             S_grid, t_grid, S_query):
    """
    Compute Delta and Gamma at specified S coordinates via AD.

    Parameters
    ----------
    model : FNOOptionPricer
    sigma, r, K_norm, T_norm : (batch,)
    S_grid, t_grid : full grids
    S_query : (n_qS,) S coordinates for Greek computation

    Returns
    -------
    delta : (batch, n_qS)
    gamma : (batch, n_qS)
    """
    batch = sigma.shape[0]
    n_qS = len(S_query)
    # Use single time point (middle of grid) for Greeks
    t_mid = t_grid[len(t_grid) // 2:len(t_grid) // 2 + 1]

    S_q = S_query.detach().clone().requires_grad_(True)
    t_q = t_mid.detach().clone().requires_grad_(True)

    V = model.query(sigma, r, K_norm, T_norm, S_q, t_q, S_grid, t_grid)

    dV_dS = torch.autograd.grad(
        V.sum(), S_q, create_graph=True, retain_graph=True
    )[0]

    d2V_dS2 = torch.autograd.grad(
        dV_dS.sum(), S_q, create_graph=True, retain_graph=True
    )[0]

    return dV_dS, d2V_dS2
