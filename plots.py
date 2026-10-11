"""Построение графиков (matplotlib, объектный интерфейс - без pyplot, безопасно для Streamlit)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from matplotlib.figure import Figure

from model import N_X, X_SHORT, ZETA_NAMES, Result

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
    "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#ad494a", "#637939", "#a55194",
]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def fig_dynamics(res: Result, selected: Sequence[int], show_target: bool = True) -> Figure:
    """X_i(t) для выбранных показателей (индексы с 0)."""
    fig = Figure(figsize=(10, 5.2), dpi=100)
    ax = fig.add_subplot(111)
    for i in selected:
        ax.plot(res.t, res.X[:, i], color=_color(i), lw=1.8, label=X_SHORT[i])
        if show_target:
            ax.axhline(res.cfg["indicators"][i]["xstar"], color=_color(i), lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("Время t")
    ax.set_ylabel("Значение показателя X_i(t)")
    ax.set_title("Динамика показателей (пунктир - рекомендованные X*)", fontsize=11)
    ax.grid(alpha=0.3)
    if selected:
        ax.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def fig_small_multiples(res: Result) -> Figure:
    """Все 14 показателей в сетке малых графиков."""
    fig = Figure(figsize=(11, 8), dpi=100)
    for i in range(N_X):
        ax = fig.add_subplot(4, 4, i + 1)
        ind = res.cfg["indicators"][i]
        ax.plot(res.t, res.X[:, i], color=_color(i), lw=1.6)
        ax.axhline(ind["xstar"], color="k", lw=0.7, ls=":")
        ax.set_ylim(ind["xmin"] - 0.03, ind["xmax"] + 0.03)
        ax.set_title(X_SHORT[i], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(4, 4, 15)
    ax.plot(res.t, res.Z, color="k", lw=1.6)
    ax.set_title("Целевая функция Z(t)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_zeta(res: Result) -> Figure:
    """Профили возмущений ζ_m(t)."""
    fig = Figure(figsize=(10, 4), dpi=100)
    ax = fig.add_subplot(111)
    for m in range(res.zeta.shape[1]):
        ax.plot(res.t, res.zeta[:, m], lw=1.8, label=ZETA_NAMES[m])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Время t")
    ax.set_ylabel("ζ(t)")
    ax.set_title("Возмущения ζ_m(t)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def fig_phase(res: Result, i: int, j: int) -> Figure:
    """Зависимость показателя X_i от показателя X_j (траектория по времени)."""
    fig = Figure(figsize=(6.5, 5), dpi=100)
    ax = fig.add_subplot(111)
    sc = ax.scatter(res.X[:, j], res.X[:, i], c=res.t, s=10, cmap="viridis")
    ax.plot(res.X[:, j], res.X[:, i], color="gray", lw=0.6, alpha=0.6)
    ax.scatter(res.X[0, j], res.X[0, i], marker="o", s=60, facecolor="none", edgecolor="k", label="t = 0")
    ax.scatter(res.X[-1, j], res.X[-1, i], marker="s", s=60, facecolor="none", edgecolor="r", label="t = T")
    fig.colorbar(sc, ax=ax, label="Время t")
    ax.set_xlabel(X_SHORT[j])
    ax.set_ylabel(X_SHORT[i])
    ax.set_title(f"Зависимость {X_SHORT[i].split()[0]} от {X_SHORT[j].split()[0]}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_sensitivity(sens: Dict[str, np.ndarray], zeta_index: int, selected: Sequence[int]) -> Figure:
    """Зависимость X_i(T) и Z(T) от уровня возмущения ζ_m."""
    fig = Figure(figsize=(11, 4.6), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    lv = sens["levels"]
    for i in selected:
        ax1.plot(lv, sens["X_final"][:, i], color=_color(i), lw=1.6, label=X_SHORT[i])
    ax1.set_xlabel(f"Уровень {ZETA_NAMES[zeta_index - 1].split()[0]}")
    ax1.set_ylabel("X_i(T)")
    ax1.set_title("Конечные значения показателей")
    ax1.grid(alpha=0.3)
    if selected:
        ax1.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax2.plot(lv, sens["Z_final"], color="k", lw=2)
    ax2.set_xlabel(f"Уровень {ZETA_NAMES[zeta_index - 1].split()[0]}")
    ax2.set_ylabel("Z(T)")
    ax2.set_title("Целевая функция")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_radar_grid(res: Result, fractions: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> Figure:
    """Сечения: лепестковые диаграммы в моменты t = frac * T (сетка 2 x 3)."""
    ind = res.cfg["indicators"]
    lo = np.array([d["xmin"] for d in ind], dtype=float)
    hi = np.array([d["xmax"] for d in ind], dtype=float)
    rng = np.where(np.abs(hi - lo) > 1e-12, hi - lo, 1.0)
    star = np.clip((np.array([d["xstar"] for d in ind], dtype=float) - lo) / rng, 0.0, 1.0)
    ang = np.linspace(0, 2 * np.pi, N_X, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = Figure(figsize=(12, 8.5), dpi=100)
    for n, fr in enumerate(fractions):
        idx = int(np.argmin(np.abs(res.t - fr * res.t[-1])))
        v = np.clip((res.X[idx] - lo) / rng, 0.0, 1.0)
        ax = fig.add_subplot(2, 3, n + 1, projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(ang_c, np.concatenate([star, star[:1]]), color="#2ca02c", lw=1.0, ls="--")
        ax.plot(ang_c, np.concatenate([v, v[:1]]), color="#d62728", lw=1.8)
        ax.fill(ang_c, np.concatenate([v, v[:1]]), color="#d62728", alpha=0.18)
        ax.set_xticks(ang)
        ax.set_xticklabels([f"X{i + 1}" for i in range(N_X)], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1"], fontsize=6, color="gray")
        ax.set_title(f"t = {res.t[idx]:.2f}", fontsize=10, pad=16)
    fig.suptitle("Сечения: лепестковые диаграммы (красный - расчёт, зелёный пунктир - X*)", fontsize=11)
    fig.tight_layout(h_pad=3.5)
    return fig


def fig_radar(res: Result, t_index: int, show_initial: bool = True, show_target: bool = True) -> Figure:
    """Лепестковая диаграмма: X_i в момент t, начальные значения и рекомендованные X_i*.

    Значения нормируются по диапазону [Xmin, Xmax] каждого показателя -> шкала 0..1."""
    ind = res.cfg["indicators"]
    lo = np.array([d["xmin"] for d in ind], dtype=float)
    hi = np.array([d["xmax"] for d in ind], dtype=float)
    rng = np.where(np.abs(hi - lo) > 1e-12, hi - lo, 1.0)

    def norm(v):
        return np.clip((np.asarray(v, dtype=float) - lo) / rng, 0.0, 1.0)

    ang = np.linspace(0, 2 * np.pi, N_X, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    def closed(v):
        return np.concatenate([v, v[:1]])

    fig = Figure(figsize=(8.2, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    if show_target:
        v = norm([d["xstar"] for d in ind])
        ax.plot(ang_c, closed(v), color="#2ca02c", lw=1.6, ls="--", label="Рекомендованные X*")
    if show_initial:
        v = norm(res.X[0])
        ax.plot(ang_c, closed(v), color="#7f7f7f", lw=1.4, label="Начальные (t = 0)")
    v = norm(res.X[t_index])
    ax.plot(ang_c, closed(v), color="#d62728", lw=2.2, label=f"Расчёт при t = {res.t[t_index]:.2f}")
    ax.fill(ang_c, closed(v), color="#d62728", alpha=0.18)

    ax.set_xticks(ang)
    ax.set_xticklabels([s.split()[0] for s in X_SHORT], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax.set_title("Лепестковая диаграмма показателей\n(шкала: (X - Xmin)/(Xmax - Xmin))", fontsize=11, pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9)
    fig.tight_layout()
    return fig
