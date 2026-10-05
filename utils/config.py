"""
Configuración central del proyecto.

Aquí viven las rutas, el esquema del dataset TCGA y las etiquetas legibles.
Todos los módulos (datos y pestañas) importan desde aquí para que no haya rutas
ni strings "mágicos" repartidos por el código.

El esquema replica exactamente el del Jupyter Book (`jbook/01_EDA.ipynb`): misma
variable objetivo, mismas 23 predictoras y misma partición. Si algo cambia allí,
se cambia aquí.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto (siempre absolutas, sin importar desde dónde se ejecute)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "dataset"
ASSETS_DIR = BASE_DIR / "assets"
LIBRO_DIR = BASE_DIR / "jbook"

# Única copia del dataset en el proyecto: la comparten el dashboard y los
# notebooks del libro, de modo que ambos analizan exactamente los mismos datos.
DATA_PATH = DATA_DIR / "TCGA_InfoWithGrade.csv"

# --------------------------------------------------------------------------- #
# Enlaces del proyecto
# --------------------------------------------------------------------------- #
# URL del servicio en Cloud Run. Si algún día se despliega con otro nombre de
# servicio o en otra región, este valor también aparece en jbook/_config.yml
# (extra_footer) y en jbook/intro.md: cámbialos a la vez.
URL_DASHBOARD = "https://gliomas-dashboard-984926604434.us-central1.run.app"

URL_LIBRO = "https://joshua2412-a.github.io/Proyecto_Viz/"
URL_LIBRO_EDA = f"{URL_LIBRO}01_EDA.html"
URL_REPO_LIBRO = "https://github.com/joshua2412-a/Proyecto_Viz"

# Autores del proyecto. El orden es el mismo que en la portada del libro
# (jbook/intro.md) y en `author` de jbook/_config.yml: los tres sitios firman
# igual. Un perfil que falte se omite sin dejar un enlace roto, así que basta
# con no poner la clave.
AUTORES = [
    {
        "nombre": "Alejandro Cantillo Escorcia",
        "github": "https://github.com/Alej0126",
        "linkedin": "https://www.linkedin.com/in/alejandro-cantillo-a8a13b3a6",
    },
    {
        "nombre": "Joshua Hincapie LLorente",
        "github": "https://github.com/joshua2412-a",
        "linkedin": (
            "https://www.linkedin.com/in/"
            "joshua-alessandro-hincapie-llorente-74b305441"
        ),
    },
]
URL_DATASET = (
    "https://archive.ics.uci.edu/dataset/759/"
    "glioma+grading+clinical+and+mutation+features+dataset"
)

# --------------------------------------------------------------------------- #
# Partición del dataset
# --------------------------------------------------------------------------- #
# El EDA se describe sobre el conjunto de entrenamiento y reserva el de prueba,
# igual que en el notebook del libro. Estos dos valores reproducen esa misma
# partición 80/20 estratificada, así que las cifras del tablero y las del libro
# coinciden exactamente. No los cambies sin recompilar el libro.
RANDOM_STATE = 42
TEST_SIZE = 0.20

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

# Reagrupación de Race para las pruebas de independencia, exactamente la del
# notebook (jbook/01_EDA.ipynb, celda de `race_agrupada`). Con los cuatro
# niveles originales, Asian y American Indian or Alaska Native dejan casillas
# con frecuencia esperada demasiado baja y el chi-cuadrado pierde validez.
#
# Solo afecta a la PRUEBA. La descripción sigue mostrando las cuatro
# categorías, igual que en el libro.
RACE_AGRUPADA = {
    0: "White",
    1: "Other racial groups",
    2: "Other racial groups",
    3: "Other racial groups",
}

# Etiquetas legibles para ejes, tablas y formularios
LABELS = {
    "Grade": "Grado del glioma",
    "Age_at_diagnosis": "Edad al diagnóstico (años)",
    "Gender": "Género",
    "Race": "Grupo racial reportado",
    "grade_label": "Grado",
}
LABELS.update({gen: f"Mutación en {gen}" for gen in GENE_FEATURES})

# Descripción breve de las tres variables clínicas. Las de los genes están en
# GENE_DESCRIPCION, más abajo. Ambas alimentan la ficha de variable que muestra
# la pestaña Exploración junto al selector.
CLINICA_DESCRIPCION = {
    "Age_at_diagnosis": "Edad del paciente en el momento del diagnóstico. Lleva "
                        "decimales porque recoge los días exactos, no solo el año.",
    "Gender": "Sexo registrado del paciente en la historia clínica de TCGA.",
    "Race": "Grupo racial reportado, en las cuatro categorías que usa TCGA. La "
            "cohorte está muy desbalanceada hacia el grupo White.",
}

# Papel de cada variable en el análisis, para la misma ficha.
VARIABLE_PAPEL = {
    "Age_at_diagnosis": "Predictora clínica · se contrasta con la U de Mann-Whitney",
    "Gender": "Predictora clínica · chi-cuadrado y V de Cramér",
    "Race": "Predictora clínica · chi-cuadrado (categorías muy desbalanceadas)",
}

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
