"""
Visualization helpers for Monte Carlo portfolio analysis.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_simulation_paths(
    paths: np.ndarray,
    initial_value: float = 100_000.0,
    n_show: int = 200,
    title: str = "Траектории портфеля (Монте-Карло)",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Рисует выборку смоделированных траекторий стоимости портфеля.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Подвыборка траекторий для наглядности
    idx = np.random.choice(paths.shape[0], size=min(n_show, paths.shape[0]), replace=False)
    for i in idx:
        ax.plot(paths[i], color="steelblue", alpha=0.15, linewidth=0.7)

    # Средняя траектория
    mean_path = paths.mean(axis=0)
    ax.plot(mean_path, color="darkblue", linewidth=2.5, label="Средняя траектория")

    # Полоса 5–95 перцентилей
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    ax.fill_between(
        range(paths.shape[1]), p5, p95, color="skyblue", alpha=0.3, label="Полоса 5–95%"
    )

    ax.axhline(initial_value, color="red", linestyle="--", linewidth=1.5, label="Начальная стоимость")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Торговые дни")
    ax.set_ylabel("Стоимость портфеля ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    sns.despine()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_final_distribution(
    final_values: np.ndarray,
    initial_value: float = 100_000.0,
    title: str = "Распределение конечной стоимости портфеля",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Гистограмма + KDE конечных значений портфеля с линиями VaR.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(final_values, bins=80, kde=True, color="steelblue", ax=ax, stat="density")

    mean_val = np.mean(final_values)
    median_val = np.median(final_values)
    var95 = np.percentile(final_values, 5)

    ax.axvline(initial_value, color="red", linestyle="--", linewidth=2, label=f"Начальная ${initial_value:,.0f}")
    ax.axvline(mean_val, color="green", linestyle="-", linewidth=2, label=f"Среднее ${mean_val:,.0f}")
    ax.axvline(median_val, color="orange", linestyle="-.", linewidth=2, label=f"Медиана ${median_val:,.0f}")
    ax.axvline(var95, color="purple", linestyle=":", linewidth=2, label=f"5-й перц. (VaR 95%) ${var95:,.0f}")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Конечная стоимость портфеля ($)")
    ax.set_ylabel("Плотность")
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_risk_metrics(
    metrics: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Столбчатые диаграммы ключевых процентных метрик.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Слева: вероятности
    probs = {
        "P(убыток)": metrics["prob_loss"] * 100,
        "P(+20%)": metrics["prob_gain_20pct"] * 100,
        "P(удвоение)": metrics["prob_double"] * 100,
    }
    axes[0].bar(probs.keys(), probs.values(), color=["#e74c3c", "#2ecc71", "#3498db"])
    axes[0].set_ylabel("Вероятность (%)")
    axes[0].set_title("Ключевые вероятности")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Справа: VaR / CVaR в % от капитала
    risk = {
        "VaR 95%": metrics["VaR_95_pct"] * 100,
        "CVaR 95%": metrics["CVaR_95_pct"] * 100,
        "VaR 99%": metrics["VaR_99_pct"] * 100,
        "CVaR 99%": metrics["CVaR_99_pct"] * 100,
    }
    axes[1].bar(risk.keys(), risk.values(), color=["#9b59b6", "#8e44ad", "#c0392b", "#e74c3c"])
    axes[1].set_ylabel("% от начального капитала")
    axes[1].set_title("Метрики риска (горизонт 1 год)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
