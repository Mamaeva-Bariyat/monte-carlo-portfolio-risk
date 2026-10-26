"""
Data loading and preprocessing utilities for portfolio analysis.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def load_portfolio_data(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers using yfinance.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols (e.g. ['AAPL', 'MSFT', 'GOOGL']).
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str or None
        End date. If None, downloads up to the most recent available date.
    auto_adjust : bool
        Whether to use adjusted close prices (recommended).

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices indexed by date, columns = tickers.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance returns MultiIndex when multiple tickers
        if "Close" in data.columns.levels[0]:
            prices = data["Close"]
        elif "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"]
        else:
            prices = data.xs("Close", axis=1, level=0, drop_level=True)
    else:
        prices = data[["Close"]].copy() if "Close" in data.columns else data

    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()  # forward-fill minor gaps then drop remaining NaNs

    # Ensure column order matches requested tickers where possible
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]

    if prices.empty:
        raise ValueError("No price data downloaded. Check tickers and date range.")

    return prices


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute daily returns from price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame (dates x tickers).
    method : {'log', 'simple'}
        Type of returns to compute.

    Returns
    -------
    pd.DataFrame
        Daily returns.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

    return returns.dropna()


def get_portfolio_params(
    returns: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate mean vector and covariance matrix of returns.

    Returns
    -------
    mean_returns : np.ndarray  shape (n_assets,)
    cov_matrix   : np.ndarray  shape (n_assets, n_assets)
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    return mean_returns, cov_matrix
