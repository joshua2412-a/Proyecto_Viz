"""Carga de datos y partición entrenamiento/prueba.

La partición reproduce exactamente la del EDA (``01_EDA.ipynb``): mismo
dataset, misma proporción, misma estratificación y misma semilla, de modo que
los 168 pacientes de prueba son los mismos que el EDA dejó sin mirar.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def cargar_dataset(ruta=None) -> pd.DataFrame:
    """Lee ``TCGA_InfoWithGrade.csv``.

    Parameters
    ----------
    ruta : str or Path, optional
        Ruta al CSV. Por defecto ``config.RUTA_DATASET``.

    Returns
    -------
    pandas.DataFrame
        Las 839 filas y 24 columnas originales, sin transformar.
    """
    return pd.read_csv(ruta or config.RUTA_DATASET)


def preparar_predictoras(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de predictoras que usan los modelos.

    Agrupa ``Race`` en ``Race_binary`` (0 = White, 1 = otros grupos), la misma
    agrupación que el EDA usó para la prueba de independencia, porque dos de
    las cuatro categorías originales tienen 14 y 1 pacientes y no permiten
    estimar un efecto propio.

    Parameters
    ----------
    df : pandas.DataFrame
        Datos con las columnas originales (con o sin ``Grade``).

    Returns
    -------
    pandas.DataFrame
        Columnas ``config.PREDICTORAS`` en ese orden, todas numéricas
        (``float`` la edad, ``int`` las binarias).
    """
    X = df.copy()
    X["Race_binary"] = (X["Race"].astype(int) != 0).astype(int)
    X = X[config.PREDICTORAS]
    X[config.BINARIAS] = X[config.BINARIAS].astype(int)
    X["Age_at_diagnosis"] = X["Age_at_diagnosis"].astype(float)
    return X


def particion(df: pd.DataFrame | None = None, semilla: int = config.SEED):
    """Partición estratificada 80/20 idéntica a la del EDA.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Dataset original. Si es ``None`` se carga con :func:`cargar_dataset`.
    semilla : int
        Semilla de la partición (42, como en el EDA).

    Returns
    -------
    X_train, X_test : pandas.DataFrame
        Predictoras ya preparadas con :func:`preparar_predictoras`.
    y_train, y_test : pandas.Series
        ``Grade`` como entero (0 = LGG, 1 = GBM).
    """
    if df is None:
        df = cargar_dataset()
    x = df.drop(columns=[config.OBJETIVO])
    y = df[config.OBJETIVO].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=semilla, stratify=y
    )
    return (preparar_predictoras(X_train), preparar_predictoras(X_test),
            y_train, y_test)
