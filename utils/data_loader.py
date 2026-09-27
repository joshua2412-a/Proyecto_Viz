"""
Carga de artefactos (dataset, modelo y métricas).

Este módulo es la ÚNICA puerta de entrada a los datos y al modelo entrenado.
Separa con claridad "entrenar" (model/train_model.py) de "cargar y usar"
(este archivo), tal y como exige la arquitectura del proyecto.

Las funciones usan `lru_cache` para que el CSV y el .pkl se lean una sola vez
por proceso, aunque diez pestañas los pidan.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from utils.config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Devuelve el dataset de rotación laboral como DataFrame.

    Añade la columna auxiliar `abandono_label` ("Abandona"/"Permanece") para
    que los gráficos puedan agrupar por texto legible en vez de 0/1.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero: python data/generate_data.py"
        )

    df = pd.read_csv(DATA_PATH)
    df["abandono_label"] = df[TARGET].map({1: "Abandona", 0: "Permanece"})
    return df


def get_dataframe() -> pd.DataFrame:
    """Copia defensiva del dataset (evita que una pestaña mute la caché)."""
    return load_data().copy()


def tasa_abandono_global() -> float:
    """Tasa de abandono del dataset completo (proporción entre 0 y 1)."""
    return float(load_data()[TARGET].mean())


def tasa_por_departamento() -> pd.DataFrame:
    """Tabla de abandono por departamento, ordenada de mayor a menor tasa."""
    df = load_data()
    resumen = (
        df.groupby("departamento")
        .agg(
            empleados=(TARGET, "size"),
            abandonos=(TARGET, "sum"),
            tasa=(TARGET, "mean"),
            salario_promedio=("salario", "mean"),
            satisfaccion_promedio=("satisfaccion", "mean"),
        )
        .reset_index()
        .sort_values("tasa", ascending=False)
    )
    return resumen


def matriz_correlacion() -> pd.DataFrame:
    """Correlaciones de Pearson entre variables numéricas y el objetivo."""
    columnas = NUMERIC_FEATURES + [TARGET]
    return load_data()[columnas].corr(numeric_only=True)


# --------------------------------------------------------------------------- #
# Modelo y métricas
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_model():
    """Carga el pipeline entrenado (preprocesamiento + regresión logística)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {MODEL_PATH}. Ejecuta primero: python model/train_model.py"
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    """Carga las métricas calculadas durante el entrenamiento."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {METRICS_PATH}. Ejecuta primero: python model/train_model.py"
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def predecir_abandono(registro: dict) -> float:
    """Probabilidad de abandono (0-1) para un empleado descrito en un dict.

    El pipeline se encarga del escalado y del one-hot encoding, por lo que
    basta con pasar los valores en crudo con los nombres de columna originales.
    """
    modelo = load_model()
    entrada = pd.DataFrame([registro])
    return float(modelo.predict_proba(entrada)[0][1])


def clear_caches() -> None:
    """Invalida las cachés (útil tras regenerar datos o reentrenar el modelo)."""
    load_data.cache_clear()
    load_model.cache_clear()
    load_metrics.cache_clear()
