"""Definición de modelos, técnicas de balanceo, pipelines y espacios de búsqueda.

Cada modelo tiene **un solo** espacio de búsqueda (:class:`Param`) del que se
derivan las cuatro representaciones que piden los optimizadores:

* la rejilla discreta de Grid Search (``Param.rejilla``),
* las distribuciones de Random Search (:func:`distribuciones_random`),
* las sugerencias de Optuna (:func:`sugerir_optuna`),
* la decodificación del genoma en ``[0, 1]^d`` del algoritmo genético
  (:func:`decodificar_genoma`).

Así los cuatro métodos exploran exactamente el mismo dominio y sus
diferencias se deben al método, no al espacio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTEN, SMOTENC
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config


# ---------------------------------------------------------------------------
# XGBoost con pesos balanceados
# ---------------------------------------------------------------------------
class XGBBalanceado(XGBClassifier):
    """``XGBClassifier`` que fija ``scale_pos_weight = n_neg / n_pos`` al ajustar.

    Es el equivalente de ``class_weight='balanced'`` para XGBoost: el cociente
    se calcula con las etiquetas del pliegue de entrenamiento que recibe
    ``fit``, de modo que nunca usa información del pliegue de validación.
    """

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        return super().fit(X, y, **kwargs)


# ---------------------------------------------------------------------------
# Espacios de búsqueda
# ---------------------------------------------------------------------------
@dataclass
class Param:
    """Un hiperparámetro del espacio de búsqueda.

    Attributes
    ----------
    nombre : str
        Nombre del argumento del estimador (sin el prefijo ``modelo__``).
    tipo : {'float', 'int', 'cat'}
        Naturaleza del parámetro.
    bajo, alto : float, optional
        Límites para ``float`` e ``int`` (ambos incluidos).
    log : bool
        Si es ``True`` el parámetro se muestrea en escala logarítmica.
    opciones : list, optional
        Valores posibles para ``cat``.
    rejilla : list
        Valores que usa Grid Search. Es un subconjunto del dominio continuo.
    """

    nombre: str
    tipo: str
    bajo: float | None = None
    alto: float | None = None
    log: bool = False
    opciones: list = field(default_factory=list)
    rejilla: list = field(default_factory=list)


ESPACIOS: dict[str, list[Param]] = {
    # 12 valores de C x 2 penalizaciones = 24 configuraciones en la rejilla.
    "logistica": [
        Param("penalty", "cat", opciones=["l1", "l2"], rejilla=["l1", "l2"]),
        Param("C", "float", 1e-3, 1e2, log=True,
              rejilla=list(np.round(np.logspace(-3, 2, 12), 5))),
    ],
    # 4 x 3 x 2 = 24. max_features en fracción de las 23 predictoras
    # (0.2 ~ sqrt(23)/23, el valor por defecto de sklearn).
    "random_forest": [
        Param("max_depth", "int", 2, 16, rejilla=[3, 5, 8, 16]),
        Param("min_samples_leaf", "int", 1, 20, rejilla=[1, 4, 10]),
        Param("max_features", "float", 0.1, 1.0, rejilla=[0.2, 0.5]),
    ],
    # 3 x 2 x 2 x 2 = 24. Los tres últimos solo los exploran los métodos
    # continuos; Grid los deja en su valor por defecto (limitación típica de
    # la rejilla: no puede cubrir siete dimensiones con 24 puntos).
    "xgboost": [
        Param("max_depth", "int", 2, 8, rejilla=[2, 3, 5]),
        Param("learning_rate", "float", 0.01, 0.3, log=True, rejilla=[0.03, 0.1]),
        Param("n_estimators", "int", 50, 400, rejilla=[100, 300]),
        Param("min_child_weight", "float", 1.0, 10.0, log=True, rejilla=[1.0, 5.0]),
        Param("subsample", "float", 0.6, 1.0),
        Param("colsample_bytree", "float", 0.5, 1.0),
        Param("reg_lambda", "float", 1e-2, 10.0, log=True),
    ],
}


def rejilla(modelo: str) -> dict:
    """Rejilla de Grid Search con el prefijo ``modelo__`` del pipeline."""
    return {f"modelo__{p.nombre}": list(p.rejilla) for p in ESPACIOS[modelo] if p.rejilla}


def distribuciones_random(modelo: str) -> dict:
    """Distribuciones de Random Search (log-uniforme donde ``Param.log``)."""
    dist = {}
    for p in ESPACIOS[modelo]:
        clave = f"modelo__{p.nombre}"
        if p.tipo == "cat":
            dist[clave] = list(p.opciones)
        elif p.tipo == "int":
            dist[clave] = randint(int(p.bajo), int(p.alto) + 1)
        elif p.log:
            dist[clave] = loguniform(p.bajo, p.alto)
        else:
            dist[clave] = uniform(p.bajo, p.alto - p.bajo)
    return dist


def sugerir_optuna(trial, modelo: str) -> dict:
    """Pide a un ``optuna.Trial`` un valor para cada parámetro del espacio."""
    params = {}
    for p in ESPACIOS[modelo]:
        if p.tipo == "cat":
            params[p.nombre] = trial.suggest_categorical(p.nombre, p.opciones)
        elif p.tipo == "int":
            params[p.nombre] = trial.suggest_int(p.nombre, int(p.bajo), int(p.alto))
        else:
            params[p.nombre] = trial.suggest_float(p.nombre, p.bajo, p.alto, log=p.log)
    return params


def decodificar_genoma(genoma, modelo: str) -> dict:
    """Traduce un genoma en ``[0, 1]^d`` a hiperparámetros.

    Cada gen se recorta a ``[0, 1]`` y se mapea al dominio del parámetro:
    lineal o logarítmicamente para ``float``, redondeando para ``int`` y por
    intervalos iguales para ``cat``.
    """
    params = {}
    for g, p in zip(genoma, ESPACIOS[modelo]):
        g = float(np.clip(g, 0.0, 1.0))
        if p.tipo == "cat":
            params[p.nombre] = p.opciones[min(int(g * len(p.opciones)), len(p.opciones) - 1)]
        elif p.log:
            v = float(np.exp(np.log(p.bajo) + g * (np.log(p.alto) - np.log(p.bajo))))
            params[p.nombre] = int(round(v)) if p.tipo == "int" else v
        else:
            v = p.bajo + g * (p.alto - p.bajo)
            params[p.nombre] = int(round(v)) if p.tipo == "int" else float(v)
    return params


# ---------------------------------------------------------------------------
# Estimadores y pipeline
# ---------------------------------------------------------------------------
def estimador(modelo: str, balanceo: str, semilla: int = config.SEED, **params):
    """Instancia el clasificador base.

    Parameters
    ----------
    modelo : {'logistica', 'random_forest', 'xgboost'}
    balanceo : str
        Solo importa si es ``'class_weight'``: activa los pesos balanceados
        nativos (o ``scale_pos_weight`` en XGBoost).
    semilla : int
        ``random_state`` del modelo.
    **params
        Hiperparámetros que sobrescriben los valores por defecto.
    """
    pesos = "balanced" if balanceo == "class_weight" else None
    if modelo == "logistica":
        base = dict(solver="liblinear", max_iter=5000, class_weight=pesos,
                    random_state=semilla)
        return LogisticRegression(**{**base, **params})
    if modelo == "random_forest":
        base = dict(n_estimators=200, class_weight=pesos, n_jobs=1,
                    random_state=semilla)
        return RandomForestClassifier(**{**base, **params})
    if modelo == "xgboost":
        base = dict(tree_method="hist", n_estimators=200, learning_rate=0.1,
                    max_depth=3, eval_metric="logloss", n_jobs=1,
                    random_state=semilla, verbosity=0)
        clase = XGBBalanceado if balanceo == "class_weight" else XGBClassifier
        return clase(**{**base, **params})
    raise ValueError(f"Modelo desconocido: {modelo}")


def muestreador(balanceo: str, semilla: int = config.SEED, n_columnas: int | None = None,
                con_edad: bool = True):
    """Técnica de sobremuestreo que va dentro del pipeline (o ``'passthrough'``).

    * ``smote``: SMOTE-NC, la variante de SMOTE para predictoras mixtas. 22 de
      las 23 predictoras son binarias; el SMOTE clásico generaría pacientes
      con "media mutación" (p. ej. ``IDH1 = 0.37``), mientras que SMOTE-NC
      interpola solo la edad y asigna a cada binaria el valor más frecuente
      entre los vecinos.
    * ``adasyn``: ADASYN no tiene variante nominal, así que sí produce valores
      intermedios en las binarias. Se mantiene porque la guía lo exige y su
      efecto se discute en el libro.

    ``n_columnas`` y ``con_edad`` describen la matriz que llega al
    muestreador (la edad, si está, va en la primera columna). Sin variables
    continuas SMOTE-NC no es aplicable y se usa SMOTE-N, su versión
    puramente nominal.
    """
    n_columnas = n_columnas or len(config.PREDICTORAS)
    if balanceo == "smote":
        if not con_edad:
            return SMOTEN(k_neighbors=5, random_state=semilla)
        return SMOTENC(categorical_features=list(range(1, n_columnas)),
                       k_neighbors=5, random_state=semilla)
    if balanceo == "adasyn":
        return ADASYN(n_neighbors=5, random_state=semilla)
    return "passthrough"


def construir_pipeline(modelo: str, balanceo: str, semilla: int = config.SEED,
                       columnas: list | None = None, **params) -> Pipeline:
    """Pipeline completo: escalado -> sobremuestreo -> clasificador.

    Todo el preprocesamiento vive dentro del pipeline, de modo que en cada
    pliegue (interno o externo) se ajusta solo con los datos de entrenamiento
    de ese pliegue: el escalador aprende media y desviación de la edad sin ver
    la validación, y el sobremuestreo nunca crea pacientes sintéticos a partir
    de pacientes de validación (``imblearn.Pipeline`` solo remuestrea en
    ``fit``, nunca en ``predict``).

    Solo se escala la edad: las binarias ya están en ``{0, 1}``. Los árboles
    no necesitan escalado, pero mantener el mismo pipeline para los tres
    modelos hace que las distancias de SMOTE/ADASYN sean comparables.

    ``columnas`` permite entrenar con un subconjunto de predictoras (análisis
    del panel reducido); por defecto se usan las 23 de ``config.PREDICTORAS``.
    """
    columnas = list(columnas or config.PREDICTORAS)
    num = [c for c in config.NUMERICAS if c in columnas]
    binarias = [c for c in config.BINARIAS if c in columnas]
    partes = ([("edad", StandardScaler(), num)] if num else []) + \
             ([("binarias", "passthrough", binarias)] if binarias else [])
    escala = ColumnTransformer(partes, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("escala", escala),
        ("balanceo", muestreador(balanceo, semilla, len(num) + len(binarias), bool(num))),
        ("modelo", estimador(modelo, balanceo, semilla, **params)),
    ])
