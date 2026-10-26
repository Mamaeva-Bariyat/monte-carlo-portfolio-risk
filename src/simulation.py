"""
Monte Carlo simulation engine for multi-asset portfolios.
Uses Geometric Brownian Motion (GBM) under the risk-neutral / historical measure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm


def run_monte_carlo_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
    initial_value: float = 100_000.0,
    n_simulations: int = 10_000,
    n_days: int = 252,
    dt: float = 1.0 / 252,
    seed: Optional[int] = 42,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate portfolio value paths using correlated Geometric Brownian Motion.

    The multivariate normal shocks are generated from the Cholesky decomposition
    of the covariance matrix so that asset correlations are preserved.

    Parameters
    ----------
    mean_returns : array-like, shape (n_assets,)
        Expected daily log-returns (or simple returns if consistent with model).
    cov_matrix : array-like, shape (n_assets, n_assets)
        Covariance matrix of daily returns.
    weights : array-like, shape (n_assets,)
        Portfolio weights (should sum to 1).
    initial_value : float
        Starting portfolio value.
    n_simulations : int
        Number of Monte Carlo paths.
    n_days : int
        Number of trading days to simulate (default 252 ≈ 1 year).
    dt : float
        Time step in years (default 1/252).
    seed : int or None
        Random seed for reproducibility.
    show_progress : bool
        Whether to display a progress bar.

    Returns
    -------
    portfolio_paths : np.ndarray, shape (n_simulations, n_days + 1)
        Simulated portfolio value paths (including t=0).
    final_values : np.ndarray, shape (n_simulations,)
        Terminal portfolio values.
    """
    mean_returns = np.asarray(mean_returns, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_assets = len(mean_returns)
    assert cov_matrix.shape == (n_assets, n_assets)
    assert len(weights) == n_assets
    assert np.isclose(weights.sum(), 1.0, atol=1e-6), "Weights must sum to 1"

    if seed is not None:
        np.random.seed(seed)

    # Cholesky for correlated Brownian motions
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Add small ridge if matrix is not positive-definite due to numerical issues
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
        L = np.linalg.cholesky(cov_matrix)

    # Pre-allocate
    portfolio_paths = np.zeros((n_simulations, n_days + 1))
    portfolio_paths[:, 0] = initial_value

    # Drift and diffusion under GBM (using historical mean)
    # For each asset: dS = mu * S * dt + sigma * S * dW
    # We work with portfolio return approximation for efficiency and clarity.

    iterator = range(n_simulations)
    if show_progress:
        iterator = tqdm(iterator, desc="Симуляции Монте-Карло")

    for i in iterator:
        # Generate correlated standard normal shocks: (n_days, n_assets)
        Z = np.random.standard_normal((n_days, n_assets))
        correlated_Z = Z @ L.T  # shape (n_days, n_assets)

        # Daily portfolio returns under the multivariate normal model
        # r_p,t = w^T * mu + w^T * correlated_shock
        daily_port_returns = mean_returns @ weights + (correlated_Z @ weights)

        # Compound the returns
        cumulative = np.cumprod(1.0 + daily_port_returns)
        portfolio_paths[i, 1:] = initial_value * cumulative

    final_values = portfolio_paths[:, -1]
    return portfolio_paths, final_values


def run_monte_carlo_gbm_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    n_simulations: int = 5000,
    n_steps: int = 252,
    dt: float = 1 / 252,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Full multi-asset GBM path simulation (asset-level then portfolio aggregation).
    Slightly more expensive but produces individual asset paths if needed.
    """
    n_assets = len(S0)
    if seed is not None:
        np.random.seed(seed)

    L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-10)

    # Shape: (n_simulations, n_steps + 1, n_assets)
    paths = np.zeros((n_simulations, n_steps + 1, n_assets))
    paths[:, 0, :] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal((n_simulations, n_assets))
        dW = Z @ L.T * np.sqrt(dt)
        # Euler-Maruyama / exact log-Euler for GBM
        paths[:, t, :] = paths[:, t - 1, :] * np.exp(
            (mu - 0.5 * np.diag(cov)) * dt + dW
        )

    # Portfolio value
    port_paths = np.sum(paths * weights, axis=2)
    return port_paths
