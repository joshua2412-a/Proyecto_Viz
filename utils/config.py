"""
Configuración central del proyecto.

Aquí viven las rutas, el esquema del dataset TCGA y las etiquetas legibles.
Todos los módulos (datos, modelo y pestañas) importan desde aquí para que no
haya rutas ni strings "mágicos" repartidos por el código.

El esquema replica exactamente el del Jupyter Book (`jbook/01_EDA.ipynb` y
`jbook/02_Baseline.ipynb`): misma variable objetivo, mismas 23 predictoras,
misma partición y mismos hiperparámetros. Si algo cambia allí, se cambia aquí.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto (siempre absolutas, sin importar desde dónde se ejecute)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
ASSETS_DIR = BASE_DIR / "assets"
LIBRO_DIR = BASE_DIR / "jbook"

# Única copia del dataset en el proyecto: la comparten el dashboard y los
# notebooks del libro, de modo que ambos analizan exactamente los mismos datos.
DATA_PATH = DATA_DIR / "TCGA_InfoWithGrade.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Enlaces del proyecto
# --------------------------------------------------------------------------- #
# URL del servicio en Cloud Run. Si algún día se despliega con otro nombre de
# servicio o en otra región, este valor también aparece en jbook/_config.yml
# (extra_footer) y en jbook/intro.md: cámbialos a la vez.
URL_DASHBOARD = "https://gliomas-dashboard-984926604434.us-central1.run.app"

URL_LIBRO = "https://joshua2412-a.github.io/Proyecto_Viz/"
URL_LIBRO_EDA = f"{URL_LIBRO}01_EDA.html"
URL_LIBRO_MODELO = f"{URL_LIBRO}02_Baseline.html"
URL_REPO_LIBRO = "https://github.com/joshua2412-a/Proyecto_Viz"
URL_DATASET = (
    "https://archive.ics.uci.edu/dataset/759/"
    "glioma+grading+clinical+and+mutation+features+dataset"
)

# --------------------------------------------------------------------------- #
# Parámetros de entrenamiento (idénticos a los del notebook 02_Baseline)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Mejor configuración encontrada por GridSearchCV en el notebook.
# `train_model.py` la usa por defecto y puede volver a buscarla con --buscar.
MEJORES_PARAMETROS = {
    "C": 0.31622776601683794,   # 10 ** -0.5
    "penalty": "l1",
    "solver": "liblinear",
    "class_weight": None,
}
REJILLA_BUSQUEDA = {
    "clasificador__C": [10 ** exponente for exponente in
                        [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]],
    "clasificador__penalty": ["l1", "l2"],
    "clasificador__solver": ["liblinear"],
    "clasificador__class_weight": [None, "balanced"],
}

# --------------------------------------------------------------------------- #
# Esquema del dataset
# --------------------------------------------------------------------------- #
TARGET = "Grade"

NUMERIC_FEATURES = ["Age_at_diagnosis"]
CATEGORICAL_FEATURES = ["Gender", "Race"]

# 20 genes con mayor frecuencia de mutación en TCGA-LGG y TCGA-GBM.
# El orden es el del dataset original y el del notebook: no reordenar.
GENE_FEATURES = [
    "IDH1",
    "TP53",
    "ATRX",
    "PTEN",
    "EGFR",
    "CIC",
    "MUC16",
    "PIK3CA",
    "NF1",
    "PIK3R1",
    "FUBP1",
    "RB1",
    "NOTCH1",
    "BCOR",
    "CSMD3",
    "SMARCA4",
    "GRIN2A",
    "IDH2",
    "FAT4",
    "PDGFRA",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + GENE_FEATURES

# --------------------------------------------------------------------------- #
# Codificación de las variables (tal como viene en TCGA_InfoWithGrade.csv)
# --------------------------------------------------------------------------- #
GRADE_LABELS = {0: "LGG", 1: "GBM"}
GRADE_NOMBRES = {
    0: "Glioma de bajo grado (LGG)",
    1: "Glioblastoma multiforme (GBM)",
}
GENDER_LABELS = {0: "Masculino", 1: "Femenino"}
RACE_LABELS = {
    0: "White",
    1: "Black or African American",
    2: "Asian",
    3: "American Indian or Alaska Native",
}
MUTACION_LABELS = {0: "No mutado (wildtype)", 1: "Mutado"}

# Etiquetas legibles para ejes, tablas y formularios
LABELS = {
    "Grade": "Grado del glioma",
    "Age_at_diagnosis": "Edad al diagnóstico (años)",
    "Gender": "Género",
    "Race": "Grupo racial reportado",
    "grade_label": "Grado",
}
LABELS.update({gen: f"Mutación en {gen}" for gen in GENE_FEATURES})

# Función biológica de cada gen (resumida del marco teórico del libro).
# Se usa en los tooltips del formulario y en la pestaña de marco teórico.
GENE_DESCRIPCION = {
    "IDH1": "Enzima del metabolismo celular. Su mutación es el marcador "
            "característico de los gliomas de bajo grado.",
    "TP53": "Gen supresor tumoral: controla el ciclo celular y la respuesta al "
            "daño del ADN.",
    "ATRX": "Mantiene la estructura de la cromatina y la estabilidad del genoma.",
    "PTEN": "Gen supresor tumoral que regula crecimiento y supervivencia celular.",
    "EGFR": "Receptor de señales de crecimiento y proliferación celular.",
    "CIC": "Regula la expresión de otros genes y el desarrollo celular.",
    "MUC16": "Proteína de gran tamaño asociada a la superficie celular.",
    "PIK3CA": "Vía PI3K/AKT: crecimiento, supervivencia y proliferación.",
    "NF1": "Regulador negativo de señales de crecimiento celular.",
    "PIK3R1": "Subunidad reguladora de la vía PI3K.",
    "FUBP1": "Regula la expresión génica y el crecimiento celular.",
    "RB1": "Gen supresor tumoral: controla la progresión hacia la división celular.",
    "NOTCH1": "Vía Notch: diferenciación, proliferación y supervivencia.",
    "BCOR": "Regula la expresión génica mediante complejos de cromatina.",
    "CSMD3": "Proteína de gran tamaño; muta en distintos tipos de tumor.",
    "SMARCA4": "Remodelación de la cromatina y regulación de la expresión génica.",
    "GRIN2A": "Subunidad de un receptor de glutamato (señalización neuronal).",
    "IDH2": "Enzima metabólica relacionada con IDH1.",
    "FAT4": "Organización, adhesión y crecimiento celular.",
    "PDGFRA": "Receptor de señales de crecimiento y proliferación.",
}

# Genes con asociación significativa al grado según el EDA (|r| >= 0.10,
# p < 0.05): se muestran arriba en el formulario para no perder tiempo con
# los que el modelo deja en cero.
GENES_DESTACADOS = ["IDH1", "ATRX", "CIC", "PTEN", "EGFR", "RB1", "TP53", "IDH2"]

# --------------------------------------------------------------------------- #
# Formulario de predicción
# --------------------------------------------------------------------------- #
# Perfil por defecto: paciente en la edad mediana del dataset, sin mutaciones.
DEFAULT_INPUT = {
    "Age_at_diagnosis": 52.0,
    "Gender": 0,
    "Race": 0,
    **{gen: 0 for gen in GENE_FEATURES},
}

# Rango del slider de edad: cubre el rango real observado (14,42 - 89,29 años)
INPUT_RANGES = {
    "Age_at_diagnosis": (14.0, 90.0, 0.5),
}

# --------------------------------------------------------------------------- #
# Umbrales de lectura de la probabilidad estimada de GBM
# --------------------------------------------------------------------------- #
# El modelo decide en 0.50; estas bandas solo sirven para comunicar cuánta
# confianza hay en esa decisión y cuándo conviene confirmar con secuenciación.
UMBRAL_LGG = 0.35   # por debajo: perfil compatible con LGG
UMBRAL_GBM = 0.65   # por encima: perfil compatible con GBM
