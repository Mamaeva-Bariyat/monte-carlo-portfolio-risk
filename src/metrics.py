"""
Risk metrics computed from Monte Carlo simulation results.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def calculate_risk_metrics(
    final_values: np.ndarray,
    initial_value: float = 100_000.0,
    confidence_levels: tuple = (0.95, 0.99),
) -> Dict[str, float]:
    """
    Compute key risk and performance metrics from terminal portfolio values.

    Metrics
    -------
    - Expected final value & mean return
    - Volatility of final values
    - Value at Risk (VaR) at given confidence levels
    - Conditional VaR / Expected Shortfall (CVaR)
    - Probability of loss
    - Probability of doubling capital (example)
    - Percentiles
    """
    final_values = np.asarray(final_values)
    returns = (final_values - initial_value) / initial_value

    metrics = {
        "initial_value": float(initial_value),
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "std_final_value": float(np.std(final_values)),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "volatility_of_terminal": float(np.std(returns)),
        "min_final_value": float(np.min(final_values)),
        "max_final_value": float(np.max(final_values)),
        "prob_loss": float(np.mean(final_values < initial_value)),
        "prob_gain_20pct": float(np.mean(returns > 0.20)),
        "prob_double": float(np.mean(final_values >= 2 * initial_value)),
    }

    for cl in confidence_levels:
        # VaR is the loss relative to initial (positive number = loss)
        var_threshold = np.percentile(final_values, (1 - cl) * 100)
        var = initial_value - var_threshold
        cvar = initial_value - np.mean(final_values[final_values <= var_threshold])

        metrics[f"VaR_{int(cl*100)}"] = float(var)
        metrics[f"CVaR_{int(cl*100)}"] = float(cvar)
        metrics[f"VaR_{int(cl*100)}_pct"] = float(var / initial_value)
        metrics[f"CVaR_{int(cl*100)}_pct"] = float(cvar / initial_value)

    # Additional percentiles
    for p in [1, 5, 10, 25, 75, 90, 95, 99]:
        metrics[f"percentile_{p}"] = float(np.percentile(final_values, p))

    return metrics


def summarize_simulations(
    final_values: np.ndarray,
    initial_value: float = 100_000.0,
) -> str:
    """Возвращает читаемую сводку результатов симуляции на русском языке."""
    m = calculate_risk_metrics(final_values, initial_value)

    summary = f"""
=== Сводка результатов Монте-Карло анализа портфеля ===
Начальный капитал              : ${m['initial_value']:,.0f}
Количество симуляций           : {len(final_values):,}

Ожидаемая конечная стоимость   : ${m['mean_final_value']:,.0f}
Медианная конечная стоимость   : ${m['median_final_value']:,.0f}
Стд. отклонение конечной ст-ти : ${m['std_final_value']:,.0f}

Средняя доходность             : {m['mean_return']*100:.1f}%
Медианная доходность           : {m['median_return']*100:.1f}%

Вероятность убытка             : {m['prob_loss']*100:.1f}%
Вероятность роста +20% и более : {m['prob_gain_20pct']*100:.1f}%
Вероятность удвоения капитала  : {m['prob_double']*100:.1f}%

VaR 95% (горизонт 1 год)       : ${m['VaR_95']:,.0f}  ({m['VaR_95_pct']*100:.1f}%)
CVaR 95% (Expected Shortfall)  : ${m['CVaR_95']:,.0f}  ({m['CVaR_95_pct']*100:.1f}%)
VaR 99%                        : ${m['VaR_99']:,.0f}  ({m['VaR_99_pct']*100:.1f}%)
CVaR 99%                       : ${m['CVaR_99']:,.0f}  ({m['CVaR_99_pct']*100:.1f}%)

5-й перцентиль                 : ${m['percentile_5']:,.0f}
95-й перцентиль                : ${m['percentile_95']:,.0f}
"""
    return summary.strip()
