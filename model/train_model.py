"""
Entrenamiento del modelo de clasificación de abandono.

Responsabilidad ÚNICA de este archivo: entrenar y persistir.
    - Lee el CSV generado por data/generate_data.py
    - Construye un Pipeline (escalado + one-hot + regresión logística)
    - Evalúa sobre un conjunto de prueba estratificado
    - Guarda el modelo en model/model.pkl y las métricas en model/metrics.json

La carga del modelo para inferencia vive en utils/data_loader.py, de modo que
el dashboard nunca reentrena nada: solo consume artefactos ya construidos.

Uso:
    python model/train_model.py
"""

from __future__ import annotations

import json
import sys
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Permite ejecutar el archivo directamente (python model/train_model.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


# --------------------------------------------------------------------------- #
# Construcción del pipeline
# --------------------------------------------------------------------------- #
def construir_pipeline() -> Pipeline:
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
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # compensa que solo ~19% de los casos son abandono
        random_state=RANDOM_STATE,
    )

    return Pipeline([("preprocesamiento", preprocesador), ("clasificador", modelo)])


# --------------------------------------------------------------------------- #
# Interpretabilidad
# --------------------------------------------------------------------------- #
def extraer_importancias(pipeline: Pipeline) -> list[dict]:
    """Coeficientes del modelo traducidos a razones de odds (odds ratio).

    Un odds ratio > 1 aumenta el riesgo de abandono; < 1 lo reduce.
    """
    preprocesador = pipeline.named_steps["preprocesamiento"]
    nombres = list(preprocesador.get_feature_names_out())
    coeficientes = pipeline.named_steps["clasificador"].coef_[0]

    importancias = []
    for nombre, coef in zip(nombres, coeficientes):
        etiqueta = nombre.split("__", 1)[-1]  # quita el prefijo del transformador
        importancias.append(
            {
                "variable": etiqueta,
                "coeficiente": round(float(coef), 4),
                "odds_ratio": round(float(np.exp(coef)), 4),
            }
        )

    # Ordena por magnitud del efecto (mayor influencia primero)
    return sorted(importancias, key=lambda d: abs(d["coeficiente"]), reverse=True)


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def entrenar() -> dict:
    """Entrena, evalúa y guarda el modelo. Devuelve el diccionario de métricas."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero: python data/generate_data.py"
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    # --- Evaluación en el conjunto de prueba ---------------------------- #
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Validación cruzada sobre entrenamiento (control de sobreajuste)
    cv_auc = cross_val_score(
        construir_pipeline(), X_train, y_train, cv=5, scoring="roc_auc"
    )

    metricas = {
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
        "importancias": extraer_importancias(pipeline),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "tasa_abandono": round(float(y.mean()), 4),
        "modelo": "Regresión logística (class_weight='balanced')",
    }

    # --- Persistencia --------------------------------------------------- #
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False)

    return metricas


def main() -> None:
    """Punto de entrada por línea de comandos."""
    m = entrenar()
    print("Modelo entrenado y guardado")
    print(f"  Modelo    : {MODEL_PATH}")
    print(f"  Métricas  : {METRICS_PATH}")
    print(f"  Train/Test: {m['n_train']} / {m['n_test']}")
    print("\nDesempeño en el conjunto de prueba:")
    for clave in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {clave:<10} {m[clave]:.3f}")
    print(f"  AUC validación cruzada (5-fold): {m['cv_auc_media']:.3f} ± {m['cv_auc_desviacion']:.3f}")
    print("\nVariables más influyentes (odds ratio):")
    for imp in m["importancias"][:5]:
        print(f"  {imp['variable']:<28} OR = {imp['odds_ratio']:.3f}")


if __name__ == "__main__":
    main()
