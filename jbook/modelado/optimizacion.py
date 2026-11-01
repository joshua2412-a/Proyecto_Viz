"""Optimización de hiperparámetros y validación cruzada anidada.

Los cuatro métodos comparten la misma interfaz (:func:`buscar`), el mismo
espacio de búsqueda (``modelos.ESPACIOS``), el mismo presupuesto de
evaluaciones y el mismo particionador interno. Cada evaluación es la AUC media
de una validación cruzada interna de ``K_INTERNO`` pliegues sobre un pipeline
completo, así que ninguna búsqueda ve los datos de validación al preprocesar.
"""

from __future__ import annotations

import json
import random
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)

from . import config, modelos


@dataclass
class ResultadoBusqueda:
    """Lo que devuelve cualquier método de búsqueda.

    Attributes
    ----------
    mejores_params : dict
        Hiperparámetros ganadores (sin prefijo ``modelo__``).
    mejor_score : float
        AUC media de validación interna de la mejor configuración.
    historial : list of float
        AUC interna de cada evaluación, en el orden en que se hizo. Su máximo
        acumulado es la curva de desempeño en cualquier momento (*anytime*).
    tiempo : float
        Tiempo de reloj (s) de toda la búsqueda.
    extra : dict
        Información propia del método (p. ej. diversidad genética).
    """

    mejores_params: dict
    mejor_score: float
    historial: list
    tiempo: float
    extra: dict = field(default_factory=dict)

    @property
    def n_eval(self) -> int:
        return len(self.historial)


def _sin_prefijo(params: dict) -> dict:
    return {k.replace("modelo__", ""): _nativo(v) for k, v in params.items()}


def _nativo(v):
    """Convierte escalares numpy a tipos de Python (para JSON)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _cv_interno(semilla: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.K_INTERNO, shuffle=True, random_state=semilla)


def _evaluar(modelo, balanceo, params, X, y, cv, semilla) -> float:
    """AUC media de validación interna para una configuración."""
    pipe = modelos.construir_pipeline(modelo, balanceo, semilla, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(cross_val_score(pipe, X, y, cv=cv,
                                             scoring=config.METRICA_OBJETIVO)))


# ---------------------------------------------------------------------------
# Los cuatro métodos
# ---------------------------------------------------------------------------
def buscar_grid(modelo, balanceo, X, y, semilla=config.SEED, presupuesto=None):
    """Grid Search exhaustivo sobre ``modelos.rejilla(modelo)`` (24 puntos)."""
    t0 = time.perf_counter()
    gs = GridSearchCV(modelos.construir_pipeline(modelo, balanceo, semilla),
                      modelos.rejilla(modelo), scoring=config.METRICA_OBJETIVO,
                      cv=_cv_interno(semilla), refit=False, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    hist = [float(s) for s in gs.cv_results_["mean_test_score"]]
    return ResultadoBusqueda(_sin_prefijo(gs.best_params_), float(gs.best_score_),
                             hist, time.perf_counter() - t0)


def buscar_random(modelo, balanceo, X, y, semilla=config.SEED,
                  presupuesto=config.PRESUPUESTO):
    """Random Search con distribuciones continuas (log-uniformes donde aplica)."""
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(modelos.construir_pipeline(modelo, balanceo, semilla),
                            modelos.distribuciones_random(modelo), n_iter=presupuesto,
                            scoring=config.METRICA_OBJETIVO, cv=_cv_interno(semilla),
                            refit=False, random_state=semilla, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rs.fit(X, y)
    hist = [float(s) for s in rs.cv_results_["mean_test_score"]]
    return ResultadoBusqueda(_sin_prefijo(rs.best_params_), float(rs.best_score_),
                             hist, time.perf_counter() - t0)


def buscar_bayesiano(modelo, balanceo, X, y, semilla=config.SEED,
                     presupuesto=config.PRESUPUESTO, n_iniciales=8):
    """Optimización bayesiana con Optuna (TPE).

    * **Modelo sustituto:** *Tree-structured Parzen Estimator*. En lugar de
      modelar ``p(AUC | λ)`` como un proceso gaussiano, TPE parte las
      evaluaciones en "buenas" (cuantil superior γ) y "malas" y ajusta una
      densidad de Parzen a cada grupo, ``l(λ)`` y ``g(λ)``.
    * **Función de adquisición:** Mejora Esperada (EI). Bergstra et al. (2011)
      muestran que, con ese sustituto, maximizar EI equivale a maximizar
      ``l(λ) / g(λ)``; TPE muestrea candidatos de ``l`` y se queda con el de
      mayor cociente. Las primeras ``n_iniciales`` evaluaciones son aleatorias
      (exploración pura) y a partir de ahí el cociente favorece la
      explotación de regiones buenas sin anular la exploración, porque los
      candidatos siguen saliendo de una densidad, no de un óptimo puntual.

    El pipeline completo se construye dentro de la función objetivo, así cada
    *trial* preprocesa con los datos de entrenamiento de cada pliegue interno.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = _cv_interno(semilla)
    t0 = time.perf_counter()

    def objetivo(trial):
        return _evaluar(modelo, balanceo, modelos.sugerir_optuna(trial, modelo),
                        X, y, cv, semilla)

    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=n_iniciales),
    )
    estudio.optimize(objetivo, n_trials=presupuesto)
    hist = [float(t.value) for t in estudio.trials]
    return ResultadoBusqueda({k: _nativo(v) for k, v in estudio.best_params.items()},
                             float(estudio.best_value), hist, time.perf_counter() - t0)


def buscar_genetico(modelo, balanceo, X, y, semilla=config.SEED,
                    presupuesto=config.PRESUPUESTO, mu=6, prob_cruce=0.7,
                    prob_mutacion=0.3, sigma=0.2, torneo=3):
    """Algoritmo genético (μ + λ) con DEAP.

    * **Representación:** un gen real en ``[0, 1]`` por hiperparámetro,
      decodificado por :func:`modelos.decodificar_genoma`. Así el mismo
      operador sirve para parámetros continuos, enteros y categóricos.
    * **Selección:** torneo de tamaño ``torneo`` (3). Mantiene presión
      selectiva moderada y, a diferencia de la ruleta, no depende de la escala
      de la AUC (todas las AUC están entre 0.85 y 0.93, la ruleta casi no
      distinguiría).
    * **Cruce:** *blend* (BLX-α, α = 0.5) con probabilidad ``prob_cruce``.
    * **Mutación:** gaussiana (σ = ``sigma``, probabilidad por gen 0.3) con
      probabilidad ``prob_mutacion`` por individuo.
    * **Elitismo:** el esquema (μ + λ) elige la siguiente generación entre
      padres e hijos, de modo que el mejor individuo nunca se pierde; un
      ``HallOfFame`` lo guarda explícitamente.
    * **Presupuesto:** μ = λ = 6 → 6 iniciales + 3 generaciones × 6 hijos =
      24 evaluaciones. Los genomas que se decodifican a la misma
      configuración se cachean y no consumen presupuesto.
    * **Diversidad:** distancia euclídea media entre pares de genomas de la
      población, normalizada por ``sqrt(d)``. Si cae a cero antes de tiempo
      hay convergencia prematura.
    """
    from deap import base, creator, tools

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individuo"):
        creator.create("Individuo", list, fitness=creator.FitnessMax)

    rng = random.Random(semilla)
    d = len(modelos.ESPACIOS[modelo])
    cv = _cv_interno(semilla)
    caja = base.Toolbox()
    caja.register("individuo", tools.initIterate, creator.Individuo,
                  lambda: [rng.random() for _ in range(d)])
    caja.register("poblacion", tools.initRepeat, list, caja.individuo)

    # DEAP usa el módulo ``random`` global en sus operadores.
    random.seed(semilla)
    t0 = time.perf_counter()
    cache, hist = {}, []

    def evaluar(ind):
        params = modelos.decodificar_genoma(ind, modelo)
        clave = json.dumps(params, sort_keys=True, default=str)
        if clave not in cache:
            cache[clave] = _evaluar(modelo, balanceo, params, X, y, cv, semilla)
            hist.append(cache[clave])
        return (cache[clave],)

    def acotar(ind):
        for i, g in enumerate(ind):
            ind[i] = min(max(g, 0.0), 1.0)
        return ind

    def diversidad(pob):
        G = np.array(pob, dtype=float)
        dif = G[:, None, :] - G[None, :, :]
        dist = np.sqrt((dif ** 2).sum(-1))[np.triu_indices(len(G), 1)]
        return float(dist.mean() / np.sqrt(d))

    pob = caja.poblacion(n=mu)
    for ind in pob:
        ind.fitness.values = evaluar(ind)
    salon = tools.HallOfFame(1)
    salon.update(pob)
    registro = [{"generacion": 0, "evaluaciones": len(hist),
                 "mejor": max(i.fitness.values[0] for i in pob),
                 "media": float(np.mean([i.fitness.values[0] for i in pob])),
                 "diversidad": diversidad(pob)}]

    gen = 0
    while len(hist) < presupuesto and gen < 50:
        gen += 1
        hijos = []
        while len(hijos) < mu:
            p1, p2 = (caja.clone(x) for x in tools.selTournament(pob, 2, tournsize=torneo))
            if rng.random() < prob_cruce:
                tools.cxBlend(p1, p2, alpha=0.5)
            for h in (p1, p2):
                if rng.random() < prob_mutacion:
                    tools.mutGaussian(h, mu=0.0, sigma=sigma, indpb=0.3)
                acotar(h)
                del h.fitness.values
                hijos.append(h)
        hijos = hijos[:mu]
        for h in hijos:
            if len(hist) >= presupuesto:
                h.fitness.values = (-np.inf,)
                continue
            h.fitness.values = evaluar(h)
        pob = tools.selBest(pob + hijos, mu)          # (mu + lambda): elitista
        salon.update(pob)
        registro.append({"generacion": gen, "evaluaciones": len(hist),
                         "mejor": salon[0].fitness.values[0],
                         "media": float(np.mean([i.fitness.values[0] for i in pob])),
                         "diversidad": diversidad(pob)})

    return ResultadoBusqueda(modelos.decodificar_genoma(salon[0], modelo),
                             float(salon[0].fitness.values[0]), hist,
                             time.perf_counter() - t0, {"generaciones": registro})


METODOS = {
    "grid": buscar_grid,
    "random": buscar_random,
    "bayesiano": buscar_bayesiano,
    "genetico": buscar_genetico,
}


def buscar(metodo: str, modelo: str, balanceo: str, X, y, semilla: int = config.SEED,
           presupuesto: int = config.PRESUPUESTO) -> ResultadoBusqueda:
    """Despacha al método de búsqueda pedido (``'grid'``, ``'random'``,
    ``'bayesiano'`` o ``'genetico'``)."""
    return METODOS[metodo](modelo, balanceo, X, y, semilla=semilla, presupuesto=presupuesto)


# ---------------------------------------------------------------------------
# Métricas y validación anidada
# ---------------------------------------------------------------------------
def metricas(y, proba, umbral: float = 0.5) -> dict:
    """Métricas de clasificación con GBM (clase 1) como clase positiva."""
    pred = (np.asarray(proba) >= umbral).astype(int)
    return {
        "auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "brier": brier_score_loss(y, proba),
    }


def _pliegue_externo(i, idx_tr, idx_va, metodo, modelo, balanceo, X, y, semilla):
    """Un pliegue del bucle externo: busca en ``idx_tr`` y evalúa en ``idx_va``."""
    X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr]
    X_va, y_va = X.iloc[idx_va], y.iloc[idx_va]
    res = buscar(metodo, modelo, balanceo, X_tr, y_tr, semilla=semilla)
    t0 = time.perf_counter()
    pipe = modelos.construir_pipeline(modelo, balanceo, semilla, **res.mejores_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
    t_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = pipe.predict_proba(X_va)[:, 1]
    t_pred = time.perf_counter() - t0
    fila = {"pliegue": i, **metricas(y_va, proba),
            "auc_interna": res.mejor_score, "n_eval": res.n_eval,
            "t_busqueda": res.tiempo, "t_por_eval": res.tiempo / max(res.n_eval, 1),
            "t_ajuste_final": t_ajuste, "t_prediccion": t_pred,
            "params": json.dumps(res.mejores_params, default=str)}
    return fila, res.historial, res.extra, idx_va, proba


def validacion_anidada(metodo: str, modelo: str, balanceo: str, X, y,
                       semilla: int = config.SEED, n_jobs: int = 2):
    """Validación cruzada anidada ``K_EXTERNO`` × ``K_INTERNO``.

    El bucle externo (estratificado, ``K_EXTERNO`` pliegues) solo mide; el
    interno vive dentro de :func:`buscar`. La métrica que se reporta es la
    del pliegue externo, que el optimizador nunca vio.

    Returns
    -------
    filas : list of dict
        Una fila por pliegue externo (métricas, tiempos, hiperparámetros).
    historiales : list of list
        Curva de búsqueda de cada pliegue.
    extras : list of dict
        Información propia del método por pliegue.
    oof : numpy.ndarray
        Probabilidades fuera de pliegue para todo ``X``.
    """
    externo = StratifiedKFold(n_splits=config.K_EXTERNO, shuffle=True, random_state=semilla)
    salida = Parallel(n_jobs=n_jobs)(
        delayed(_pliegue_externo)(i, tr, va, metodo, modelo, balanceo, X, y, semilla)
        for i, (tr, va) in enumerate(externo.split(X, y))
    )
    oof = np.full(len(y), np.nan)
    filas, historiales, extras = [], [], []
    for fila, hist, extra, idx_va, proba in salida:
        filas.append(fila)
        historiales.append(hist)
        extras.append(extra)
        oof[idx_va] = proba
    return filas, historiales, extras, oof
