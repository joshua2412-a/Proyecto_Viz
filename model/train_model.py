"""
Entrenamiento del clasificador de grado tumoral (LGG vs GBM).

Responsabilidad ÚNICA de este archivo: entrenar y persistir.
    - Lee dataset/TCGA_InfoWithGrade.csv
    - Construye el mismo Pipeline del notebook jbook/02_Baseline.ipynb
    - Evalúa sobre el conjunto de prueba estratificado (80/20, seed 42)
    - Guarda el modelo en model/model.pkl y las métricas en model/metrics.json

La carga del modelo para inferencia vive en utils/data_loader.py, de modo que
el dashboard nunca reentrena nada: solo consume artefactos ya construidos.

El preprocesamiento es idéntico al del libro:
    Age_at_diagnosis -> StandardScaler   (la regresión regularizada es sensible
                                          a la escala)
    Gender, Race     -> OneHotEncoder    (drop="if_binary")
    20 mutaciones    -> passthrough      (ya son indicadores 0/1)

Uso:
    python model/train_model.py              # usa los hiperparámetros del libro
    python model/train_model.py --buscar     # repite la búsqueda con GridSearchCV
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Permite ejecutar el archivo directamente (python model/train_model.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    FEATURES,
    GENE_FEATURES,
    MEJORES_PARAMETROS,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REJILLA_BUSQUEDA,
    TARGET,
    TEST_SIZE,
)


# --------------------------------------------------------------------------- #
# Construcción del pipeline
# --------------------------------------------------------------------------- #
def construir_pipeline(**parametros_modelo) -> Pipeline:
    """Devuelve el pipeline completo de preprocesamiento + modelo.

    Encapsular el preprocesamiento dentro del pipeline es clave: el .pkl
    resultante acepta datos en crudo (como los que envía el formulario de la
    pestaña de predicción) sin que haya que replicar el escalado a mano.
    """
    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), NUMERIC_FEATURES),
            (
                "categoricas",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("geneticas", "passthrough", GENE_FEATURES),
        ],
        remainder="drop",
    )

    parametros = {**MEJORES_PARAMETROS, **parametros_modelo}
    modelo = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        **parametros,
    )

    return Pipeline([("preprocesamiento", preprocesador), ("clasificador", modelo)])


# --------------------------------------------------------------------------- #
# Interpretabilidad
# --------------------------------------------------------------------------- #
def extraer_coeficientes(pipeline: Pipeline) -> list[dict]:
    """Coeficientes del modelo traducidos a razones de odds (odds ratio).

    Un odds ratio > 1 aumenta las probabilidades de GBM; < 1 las reduce (empuja
    hacia LGG). Con penalización L1 muchos coeficientes quedan exactamente en
    cero: esa es la selección automática de variables que hace el modelo.
    """
    preprocesador = pipeline.named_steps["preprocesamiento"]
    nombres = [n.split("__", 1)[-1] for n in preprocesador.get_feature_names_out()]
    coeficientes = pipeline.named_steps["clasificador"].coef_.ravel()

    detalle = [
        {
            "variable": nombre,
            "coeficiente": round(float(coef), 4),
            "odds_ratio": round(float(np.exp(coef)), 4),
        }
        for nombre, coef in zip(nombres, coeficientes)
    ]
    return sorted(detalle, key=lambda fila: abs(fila["coeficiente"]), reverse=True)


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def entrenar(buscar_hiperparametros: bool = False) -> dict:
    """Entrena, evalúa y guarda el modelo. Devuelve el diccionario de métricas."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}.\n"
            "Coloca TCGA_InfoWithGrade.csv en dataset/ (ver dataset/README.md)."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Misma partición que el notebook: 80/20 estratificada con seed 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    validacion = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # --- Búsqueda de hiperparámetros (opcional) -------------------------- #
    parametros = dict(MEJORES_PARAMETROS)
    tiempo_busqueda = 0.0
    if buscar_hiperparametros:
        inicio = time.perf_counter()
        busqueda = GridSearchCV(
            estimator=construir_pipeline(),
            param_grid=REJILLA_BUSQUEDA,
            scoring="roc_auc",
            cv=validacion,
            n_jobs=-1,
            refit=False,
        )
        busqueda.fit(X_train, y_train)
        tiempo_busqueda = time.perf_counter() - inicio
        parametros = {
            clave.replace("clasificador__", ""): valor
            for clave, valor in busqueda.best_params_.items()
        }
        print(f"[busqueda] Mejor configuración: {parametros}")
        print(f"[busqueda] AUC-ROC media en CV: {busqueda.best_score_:.4f}")

    # --- Entrenamiento final -------------------------------------------- #
    pipeline = construir_pipeline(**parametros)
    inicio = time.perf_counter()
    pipeline.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    # --- Evaluación en el conjunto de prueba ----------------------------- #
    inicio = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    tiempo_inferencia = time.perf_counter() - inicio

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Validación cruzada sobre entrenamiento (control de sobreajuste)
    cv_auc = cross_val_score(
        construir_pipeline(**parametros),
        X_train,
        y_train,
        cv=validacion,
        scoring="roc_auc",
    )

    metricas = {
        "modelo": "Regresión logística (baseline del Jupyter Book)",
        "hiperparametros": {
            clave: (valor if not isinstance(valor, float) else round(valor, 6))
            for clave, valor in parametros.items()
        },
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_auc_media": round(float(cv_auc.mean()), 4),
        "cv_auc_desviacion": round(float(cv_auc.std()), 4),
        "matriz_confusion": cm.tolist(),
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
        },
        "coeficientes": extraer_coeficientes(pipeline),
        "intercepto": round(float(pipeline.named_steps["clasificador"].intercept_[0]), 4),
        "n_variables_originales": len(FEATURES),
        # Columnas que ve el modelo tras el one-hot (Race se expande a 4)
        "n_columnas_modelo": int(pipeline.named_steps["clasificador"].coef_.size),
        "n_variables_activas": int(
            (pipeline.named_steps["clasificador"].coef_.ravel() != 0).sum()
        ),
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "prevalencia_gbm": round(float(y.mean()), 4),
        "tiempos_segundos": {
            "busqueda": round(tiempo_busqueda, 3),
            "entrenamiento": round(tiempo_entrenamiento, 3),
            "inferencia": round(tiempo_inferencia, 5),
        },
    }

    # --- Persistencia ---------------------------------------------------- #
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # newline="\n" a propósito: sin esto, Python en Windows escribe CRLF y git
    # marca el archivo entero como modificado en cada reentrenamiento, aunque
    # las métricas no hayan cambiado.
    with open(METRICS_PATH, "w", encoding="utf-8", newline="\n") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False)

    return metricas


def main() -> None:
    """Punto de entrada por línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buscar",
        action="store_true",
        help="repite la búsqueda de hiperparámetros con GridSearchCV",
    )
    argumentos = parser.parse_args()

    m = entrenar(buscar_hiperparametros=argumentos.buscar)

    print("Modelo entrenado y guardado")
    print(f"  Modelo    : {MODEL_PATH}")
    print(f"  Métricas  : {METRICS_PATH}")
    print(f"  Train/Test: {m['n_train']} / {m['n_test']} pacientes")
    print(
        f"  Variables : {m['n_variables_activas']} columnas activas de "
        f"{m['n_columnas_modelo']} ({m['n_variables_originales']} variables originales, "
        "penalización L1)"
    )
    print("\nDesempeño en el conjunto de prueba:")
    for clave in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {clave:<10} {m[clave]:.3f}")
    print(
        f"  AUC validación cruzada ({CV_FOLDS}-fold): "
        f"{m['cv_auc_media']:.3f} ± {m['cv_auc_desviacion']:.3f}"
    )
    print("\nVariables más influyentes (odds ratio):")
    for fila in m["coeficientes"][:6]:
        print(f"  {fila['variable']:<20} OR = {fila['odds_ratio']:.3f}")


if __name__ == "__main__":
    main()
