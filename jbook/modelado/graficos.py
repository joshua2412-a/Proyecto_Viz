"""Gráficos reutilizables del capítulo de modelado (matplotlib)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import config

TINTA = "#1f2933"
TINTA_2 = "#52606d"
REJILLA = "#e4e7eb"


def estilo():
    """Estilo común: fondo blanco, rejilla tenue, sin bordes superior/derecho."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#9aa5b1", "axes.labelcolor": TINTA,
        "axes.titlecolor": TINTA, "axes.titleweight": "bold", "axes.titlesize": 11.5,
        "axes.labelsize": 10, "axes.grid": True, "grid.color": REJILLA,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "font.size": 10,
        "legend.frameon": False, "figure.dpi": 100,
    })


def diagrama_cd(rangos, cd: float, ax=None, titulo: str | None = None):
    """Diagrama de diferencia crítica (Demšar, 2006).

    Coloca cada tratamiento en su rango promedio (1 = mejor, a la izquierda) y
    une con una barra horizontal los grupos cuya diferencia de rangos es menor
    que ``cd`` (estadísticamente indistinguibles según Nemenyi).

    Parameters
    ----------
    rangos : pandas.Series
        Rango promedio por tratamiento (índice = nombre legible).
    cd : float
        Diferencia crítica.
    """
    rangos = rangos.sort_values()
    k = len(rangos)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.2 + 0.32 * k))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    lo, hi = 1, k
    ax.set_xlim(lo - 0.1, hi + 0.1)
    ax.xaxis.tick_top()
    ax.set_xticks(range(1, k + 1))
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color(TINTA_2)
    mitad = int(np.ceil(k / 2))
    n_izq = mitad
    alto_texto = 0.6
    for i, (nombre, r) in enumerate(rangos.items()):
        izquierda = i < n_izq
        nivel = (i if izquierda else k - 1 - i) + 1
        y = -nivel * alto_texto - 0.4
        x_txt = lo - 0.15 if izquierda else hi + 0.15
        ax.plot([r, r, x_txt], [0, y, y], color=TINTA_2, lw=0.9)
        ax.text(x_txt + (-0.05 if izquierda else 0.05), y, f"{nombre} ({r:.2f})",
                ha="right" if izquierda else "left", va="center", fontsize=9, color=TINTA)
    # Grupos no significativos: secuencias maximales con rango_max - rango_min < cd
    valores = rangos.to_numpy()
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and valores[j + 1] - valores[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for g, (a, b) in enumerate(grupos):
        y = -0.15 - 0.12 * g
        ax.plot([valores[a] - 0.03, valores[b] + 0.03], [y, y], color=TINTA, lw=3,
                solid_capstyle="round")
    # Barra de referencia con la CD
    ax.plot([lo, lo + cd], [0.55, 0.55], color=config.COLOR_MODELO["logistica"], lw=2)
    ax.text(lo + cd / 2, 0.68, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9,
            color=TINTA)
    ax.set_ylim(-(mitad + 1) * alto_texto - 0.2, 1.0)
    if titulo:
        ax.set_title(titulo, pad=28)
    return ax


def rangos_con_cd(rangos, cd: float, colores=None, ax=None, titulo=None):
    """Variante para muchos tratamientos: un punto por tratamiento en su rango
    promedio, ordenados, con la banda ``[mejor, mejor + CD]``.

    Los tratamientos dentro de la banda no se distinguen del mejor.
    """
    rangos = rangos.sort_values()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.2 * len(rangos) + 1))
    y = np.arange(len(rangos))[::-1]
    mejor = rangos.iloc[0]
    ax.axvspan(mejor, mejor + cd, color="#d9e8f5", zorder=0,
               label=f"No distinguible del mejor (CD = {cd:.1f})")
    c = [colores.get(n, TINTA_2) if colores else TINTA_2 for n in rangos.index]
    ax.scatter(rangos.to_numpy(), y, s=36, c=c, zorder=3, edgecolor="white", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rangos.index, fontsize=7.5)
    ax.set_xlabel("Rango promedio en los pliegues externos (1 = mejor)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    if titulo:
        ax.set_title(titulo)
    return ax


def mejor_acumulado(historial) -> np.ndarray:
    """Máximo acumulado de una secuencia de evaluaciones (curva *anytime*)."""
    return np.maximum.accumulate(np.asarray(historial, dtype=float))


def confiabilidad(ax, curva, color, etiqueta):
    """Dibuja una curva del diagrama de confiabilidad sobre ``ax``."""
    ax.plot(curva["predicha"], curva["observada"], "-o", color=color, lw=2, ms=6,
            label=etiqueta, markeredgecolor="white", markeredgewidth=1)


def beeswarm(valores_shap, X, nombres, ax=None, max_vars: int = 15, titulo=None):
    """Gráfico resumen de SHAP (*beeswarm*).

    Una fila por variable, ordenadas por ``mean(|SHAP|)``; cada punto es un
    paciente, situado según su contribución y coloreado por el valor de la
    variable (azul = bajo / ausente, rojo = alto / mutado).
    """
    valores_shap = np.asarray(valores_shap)
    X = np.asarray(X, dtype=float)
    orden = np.argsort(np.abs(valores_shap).mean(0))[::-1][:max_vars]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.38 * len(orden) + 1.2))
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for fila, j in enumerate(orden[::-1]):
        v = valores_shap[:, j]
        x = X[:, j]
        xn = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
        # dispersión vertical proporcional a la densidad local
        hist, bordes = np.histogram(v, bins=40)
        dens = hist[np.clip(np.digitize(v, bordes[1:-1]), 0, 39)]
        jitter = rng.uniform(-1, 1, len(v)) * 0.38 * dens / max(dens.max(), 1)
        ax.scatter(v, fila + jitter, c=cmap(xn), s=9, alpha=0.85, linewidth=0)
    ax.axvline(0, color=TINTA_2, lw=0.8)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([nombres[j] for j in orden[::-1]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Valor SHAP (contribución al log-odds de GBM)")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["bajo / 0", "alto / 1"])
    cb.outline.set_visible(False)
    if titulo:
        ax.set_title(titulo)
    return ax


def waterfall(base, contribuciones, valores, nombres, ax=None, max_vars: int = 9,
              titulo=None):
    """Explicación local tipo *waterfall*.

    Parte del valor base (log-odds medio) y suma, de mayor a menor magnitud,
    la contribución de cada variable hasta llegar a la predicción del
    paciente; las variables restantes se agrupan en "otras".
    """
    contribuciones = np.asarray(contribuciones, dtype=float)
    orden = np.argsort(np.abs(contribuciones))[::-1]
    top, resto = orden[:max_vars], orden[max_vars:]
    etiquetas = [f"{nombres[j]} = {_fmt(valores[j])}" for j in top]
    pasos = list(contribuciones[top])
    if len(resto):
        etiquetas.append(f"otras {len(resto)} variables")
        pasos.append(contribuciones[resto].sum())
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.42 * len(pasos) + 1.4))
    posiciones = base + np.concatenate([[0], np.cumsum(pasos)])
    lo, hi = posiciones.min(), posiciones.max()
    margen = 0.18 * (hi - lo if hi > lo else 1)
    ax.set_xlim(lo - margen, hi + margen)
    acumulado = base
    for fila, (et, d) in enumerate(zip(etiquetas, pasos)):
        y = len(pasos) - 1 - fila
        color = config.COLOR_CLASE[1] if d > 0 else config.COLOR_CLASE[0]
        ax.barh(y, d, left=acumulado, color=color, height=0.62, edgecolor="white")
        # la etiqueta va en el extremo exterior de la barra
        x_txt = max(acumulado, acumulado + d) + margen * 0.08 if d > 0 \
            else min(acumulado, acumulado + d) - margen * 0.08
        ax.text(x_txt, y, f"{d:+.2f}", va="center", ha="left" if d > 0 else "right",
                fontsize=8.5, color=TINTA)
        acumulado += d
    ax.set_yticks(range(len(pasos)))
    ax.set_yticklabels(etiquetas[::-1], fontsize=9)
    ax.axvline(base, color=TINTA_2, lw=0.8, ls="--")
    ax.axvline(acumulado, color=TINTA, lw=1.2)
    p = 1 / (1 + np.exp(-acumulado))
    ax.set_xlabel(f"log-odds de GBM   (base {base:+.2f}  →  f(x) = {acumulado:+.2f}, "
                  f"P(GBM) = {p:.2f})")
    ax.grid(axis="y", visible=False)
    if titulo:
        ax.set_title(titulo)
    return ax


def _fmt(v):
    v = float(v)
    return f"{v:.0f}" if v.is_integer() else f"{v:.1f}"
