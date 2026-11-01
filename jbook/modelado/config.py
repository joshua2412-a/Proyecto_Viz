"""Configuración global del modelado: semilla, rutas, variables y colores.

Todo lo que deba ser idéntico entre el notebook, el script de experimentos y
el dashboard se define aquí una sola vez.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
#: Semilla global. Es la misma del EDA (``train_test_split(random_state=42)``)
#: y se propaga a numpy, scikit-learn, imbalanced-learn, XGBoost, Optuna y DEAP.
SEED = 42

#: Proporción reservada como conjunto de prueba (idéntica al EDA).
TEST_SIZE = 0.20

#: Pliegues del bucle externo e interno de la validación anidada.
K_EXTERNO = 10
K_INTERNO = 3

#: Evaluaciones (configuraciones de hiperparámetros) por búsqueda. Es igual
#: para los cuatro optimizadores, así la comparación es a presupuesto fijo.
PRESUPUESTO = 24

#: Métrica que guía la búsqueda de hiperparámetros.
METRICA_OBJETIVO = "roc_auc"

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ_LIBRO = Path(__file__).resolve().parents[1]          # .../jbook
RAIZ_PROYECTO = RAIZ_LIBRO.parent                          # .../Proyecto_Viz
RUTA_DATASET = RAIZ_PROYECTO / "dataset" / "TCGA_InfoWithGrade.csv"
DIR_RESULTADOS = RAIZ_LIBRO / "resultados"

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
OBJETIVO = "Grade"
GENES = [
    "IDH1", "TP53", "ATRX", "PTEN", "EGFR", "CIC", "MUC16", "PIK3CA", "NF1",
    "PIK3R1", "FUBP1", "RB1", "NOTCH1", "BCOR", "CSMD3", "SMARCA4", "GRIN2A",
    "IDH2", "FAT4", "PDGFRA",
]
NUMERICAS = ["Age_at_diagnosis"]
#: Binarias en el orden en que entran al modelo (``Race_binary`` sustituye a
#: ``Race``: blanco = 0, otros grupos = 1, la misma agrupación del EDA).
BINARIAS = ["Gender", "Race_binary"] + GENES
PREDICTORAS = NUMERICAS + BINARIAS

# ---------------------------------------------------------------------------
# Nombres legibles y colores
# ---------------------------------------------------------------------------
MODELOS = ["logistica", "random_forest", "xgboost"]
BALANCEOS = ["ninguno", "smote", "adasyn", "class_weight"]
OPTIMIZADORES = ["grid", "random", "bayesiano", "genetico"]

NOMBRES = {
    "logistica": "Regresión Logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "ninguno": "Sin balanceo",
    "smote": "SMOTE-NC",
    "adasyn": "ADASYN",
    "class_weight": "class_weight",
    "grid": "Grid Search",
    "random": "Random Search",
    "bayesiano": "Bayesiano (TPE)",
    "genetico": "Genético (DEAP)",
}

#: Colores de clase, los mismos del EDA (``PALETTE_GRADE``).
COLOR_CLASE = {0: "#2E86AB", 1: "#C1443C"}

#: Un color por modelo. Paleta fría validada para daltonismo (todas las
#: parejas, ΔE OKLab >= 9) sobre fondo claro.
COLOR_MODELO = {
    "logistica": "#1f5fa8",
    "random_forest": "#199e70",
    "xgboost": "#9085e9",
}

#: Un color por optimizador (las curvas siempre llevan etiqueta directa).
COLOR_OPTIMIZADOR = {
    "grid": "#8a94a6",
    "random": "#1f5fa8",
    "bayesiano": "#199e70",
    "genetico": "#9085e9",
}
