"""
Black-Scholes analytical formulas, Greeks, and utility functions
"""
import numpy as np
import torch
from scipy.stats import norm


def black_scholes_price(S, K, T, sigma, r, option_type='call'):
    """
    Black-Scholes analytical price for European options.
    
    Parameters
    ----------
    S : float or np.ndarray - Asset price
    K : float or np.ndarray - Strike price
    T : float or np.ndarray - Time to maturity (years)
    sigma : float or np.ndarray - Volatility
    r : float or np.ndarray - Risk-free rate
    option_type : str - 'call' or 'put'
    
    Returns
    -------
    price : float or np.ndarray
    """
    # Handle T=0 case (at expiry)
    if np.isscalar(T):
        if T <= 0:
            if option_type == 'call':
                return np.maximum(S - K, 0.0)
            else:
                return np.maximum(K - S, 0.0)
    else:
        # Array case: handle element-wise
        price = np.where(
            T <= 0,
            np.maximum(S - K, 0.0) if option_type == 'call' else np.maximum(K - S, 0.0),
            _bs_price(S, K, T, sigma, r, option_type)
        )
        return price
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def _bs_price(S, K, T, sigma, r, option_type):
    """Internal helper for BS price (assumes T > 0)"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_greeks(S, K, T, sigma, r, option_type='call'):
    """
    Compute Greeks for European options.
    
    Returns
    -------
    greeks : dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    if T <= 0:
        # At expiry: Greeks are not well-defined (discontinuous)
        delta = np.where(S > K, 1.0, 0.0) if option_type == 'call' else np.where(S > K, 0.0, -1.0)
        return {
            'delta': delta,
            'gamma': np.zeros_like(S),
            'vega': np.zeros_like(S),
            'theta': np.zeros_like(S),
            'rho': np.zeros_like(S)
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
    
    # Gamma (same for call and put)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    
    # Vega (same for call and put)
    vega = S * pdf_d1 * sqrt_T / 100  # Divided by 100 for 1% change
    
    # Theta
    common_theta = -S * pdf_d1 * sigma / (2 * sqrt_T)
    if option_type == 'call':
        theta = common_theta - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = common_theta + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = theta / 365  # Per-day theta
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def monte_carlo_price(S, K, T, sigma, r, n_paths=100000, seed=42, option_type='call'):
    """
    Monte Carlo simulation for European option pricing.
    
    Parameters
    ----------
    S : float - Current asset price
    K : float - Strike price
    T : float - Time to maturity
    sigma : float - Volatility
    r : float - Risk-free rate
    n_paths : int - Number of Monte Carlo paths
    seed : int - Random seed
    option_type : str - 'call' or 'put'
    
    Returns
    -------
    price : float - MC estimated price
    std_err : float - Standard error of estimate
    """
    rng = np.random.default_rng(seed)
    
    # Simulate terminal prices using geometric Brownian motion
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Compute payoffs
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    
    # Discount and average
    discount = np.exp(-r * T)
    prices = discount * payoffs
    
    return np.mean(prices), np.std(prices) / np.sqrt(n_paths)


# =============================================================================
# PyTorch versions for GPU computation
# =============================================================================

def black_scholes_price_torch(S, K, T, sigma, r, option_type='call'):
    """
    PyTorch version of Black-Scholes pricing (supports batching on GPU).
    """
    eps = 1e-10
    sqrt_T = torch.sqrt(T.clamp(min=eps))
    
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if option_type == 'call':
        price = S * torch.normal_cdf(d1) - K * torch.exp(-r * T) * torch.normal_cdf(d2)
    else:
        price = K * torch.exp(-r * T) * torch.normal_cdf(-d2) - S * torch.normal_cdf(-d1)
    
    # Handle T ≈ 0 case
    at_expiry = T <= eps
    if option_type == 'call':
        payoff_at_expiry = torch.maximum(S - K, torch.tensor(0.0, device=S.device))
    else:
        payoff_at_expiry = torch.maximum(K - S, torch.tensor(0.0, device=S.device))
    
    price = torch.where(at_expiry, payoff_at_expiry, price)
    return price


def compute_rmse(pred, true):
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - true)**2))


def compute_mape(pred, true, eps=1e-8):
    """Mean Absolute Percentage Error"""
    mask = true.abs() > eps
    return torch.mean(torch.abs((pred[mask] - true[mask]) / (true[mask] + eps))) * 100


def compute_max_error(pred, true):
    """Maximum Absolute Error"""
    return torch.max(torch.abs(pred - true))
