"""Calibración, pruebas estadísticas y tamaños de efecto.

Todo está implementado con numpy/scipy para no depender de librerías
adicionales; cada función indica la referencia del método.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------
def ece(y, proba, n_bins: int = 10) -> float:
    """Error de calibración esperado (Naeini et al., 2015).

    Promedio, ponderado por el número de pacientes de cada *bin* de igual
    ancho, de ``|frecuencia observada - probabilidad media predicha|``.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    bordes = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bordes[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - proba[m].mean())
    return float(total)


def curva_confiabilidad(y, proba, n_bins: int = 10) -> pd.DataFrame:
    """Puntos del diagrama de confiabilidad (bins de igual ancho)."""
    y, proba = np.asarray(y), np.asarray(proba)
    bordes = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bordes[1:-1]), 0, n_bins - 1)
    filas = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            filas.append({"bin": b, "predicha": proba[m].mean(),
                          "observada": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# DeLong (AUC correlacionadas)
# ---------------------------------------------------------------------------
def _componentes_delong(pos, neg):
    """Componentes estructurales V10 y V01 de DeLong (forma O(n log n))."""
    m, n = len(pos), len(neg)
    todos = np.concatenate([pos, neg])
    r_todos = stats.rankdata(todos)
    r_pos = stats.rankdata(pos)
    r_neg = stats.rankdata(neg)
    v10 = (r_todos[:m] - r_pos) / n
    v01 = 1.0 - (r_todos[m:] - r_neg) / m
    return v10, v01


def delong(y, proba_a, proba_b) -> dict:
    """Prueba de DeLong para dos AUC medidas sobre los mismos pacientes.

    DeLong, DeLong y Clarke-Pearson (1988), con el algoritmo rápido de Sun y
    Xu (2014). Bajo H0 las dos AUC son iguales; el estadístico usa la
    covarianza entre ambas, que es alta porque se evalúan sobre los mismos
    pacientes.

    Returns
    -------
    dict
        ``auc_a``, ``auc_b``, ``diferencia`` (a - b), ``z`` y ``p``
        (bilateral).
    """
    y = np.asarray(y).astype(bool)
    aucs, v10s, v01s = [], [], []
    for p in (np.asarray(proba_a), np.asarray(proba_b)):
        v10, v01 = _componentes_delong(p[y], p[~y])
        aucs.append(v10.mean())
        v10s.append(v10)
        v01s.append(v01)
    s10 = np.cov(np.vstack(v10s))
    s01 = np.cov(np.vstack(v01s))
    cov = s10 / y.sum() + s01 / (~y).sum()
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    dif = aucs[0] - aucs[1]
    z = dif / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": aucs[0], "auc_b": aucs[1], "diferencia": dif,
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def holm(pvalores) -> np.ndarray:
    """Ajuste de Holm-Bonferroni (controla el error de familia)."""
    p = np.asarray(pvalores, dtype=float)
    orden = np.argsort(p)
    k = len(p)
    ajust = np.empty(k)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, (k - rango) * p[i])
        ajust[i] = min(acumulado, 1.0)
    return ajust


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def ic_bootstrap(estadistico, *muestras, n: int = 2000, semilla: int = 42,
                 metodo: str = "BCa", nivel: float = 0.95):
    """Intervalo de confianza bootstrap (BCa por defecto) remuestreando pacientes.

    Usa ``scipy.stats.bootstrap`` con ``paired=True``: todas las muestras se
    remuestrean con los mismos índices, como exige comparar dos modelos sobre
    los mismos pacientes. Si una réplica deja una sola clase, el estadístico
    devuelve ``nan`` y BCa no es calculable; en ese caso se cae al método de
    percentiles.
    """
    rng = np.random.default_rng(semilla)

    def _est(*args):
        try:
            return estadistico(*args)
        except ValueError:
            return np.nan

    try:
        r = stats.bootstrap(muestras, _est, paired=True, vectorized=False,
                            n_resamples=n, method=metodo, confidence_level=nivel,
                            random_state=rng)
        bajo, alto = r.confidence_interval
        if np.isnan(bajo) or np.isnan(alto):
            raise ValueError
    except ValueError:
        r = stats.bootstrap(muestras, _est, paired=True, vectorized=False,
                            n_resamples=n, method="percentile",
                            confidence_level=nivel, random_state=rng)
        bajo, alto = r.confidence_interval
    return float(bajo), float(alto)


# ---------------------------------------------------------------------------
# Friedman, Nemenyi y diferencia crítica
# ---------------------------------------------------------------------------
def friedman(matriz: pd.DataFrame) -> dict:
    """Prueba de Friedman sobre una matriz bloques × tratamientos.

    Parameters
    ----------
    matriz : pandas.DataFrame
        Filas = bloques (pliegues externos), columnas = modelos; valores = la
        métrica (más alto es mejor).

    Returns
    -------
    dict
        Estadístico χ²_F, grados de libertad, p-valor, la corrección de Iman y
        Davenport (F_F, menos conservadora) y los rangos promedio (1 = mejor).
    """
    rangos = matriz.rank(axis=1, ascending=False)
    N, k = matriz.shape
    chi2, p = stats.friedmanchisquare(*[matriz[c].to_numpy() for c in matriz.columns])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    return {"chi2": float(chi2), "gl": k - 1, "p": float(p), "F_ID": float(ff),
            "p_ID": float(p_ff), "N": N, "k": k,
            "rangos": rangos.mean().sort_values()}


def diferencia_critica(k: int, N: int, alfa: float = 0.05) -> float:
    """Diferencia crítica de Nemenyi (Demšar, 2006).

    ``CD = q_α · sqrt(k (k + 1) / (6 N))`` con ``q_α`` el cuantil del rango
    estudentizado con infinitos grados de libertad dividido entre ``√2``.
    """
    q = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * N)))


def nemenyi(rangos: pd.Series, N: int) -> pd.DataFrame:
    """p-valores de Nemenyi para todos los pares a partir de los rangos promedio."""
    k = len(rangos)
    se = np.sqrt(k * (k + 1) / (6 * N))
    nombres = list(rangos.index)
    p = pd.DataFrame(1.0, index=nombres, columns=nombres)
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(rangos.iloc[i] - rangos.iloc[j]) / se * np.sqrt(2)
            p.iloc[i, j] = p.iloc[j, i] = float(stats.studentized_range.sf(q, k, np.inf))
    return p


# ---------------------------------------------------------------------------
# Tamaños de efecto
# ---------------------------------------------------------------------------
def delta_cliff(a, b) -> float:
    """Delta de Cliff: P(a > b) - P(a < b). Rango [-1, 1]."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def magnitud_cliff(d: float) -> str:
    """Umbrales de Romano et al. (2006): 0.147, 0.33 y 0.474."""
    d = abs(d)
    if d < 0.147:
        return "despreciable"
    if d < 0.33:
        return "pequeño"
    if d < 0.474:
        return "mediano"
    return "grande"
