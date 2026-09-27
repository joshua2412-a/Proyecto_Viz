"""
Pestaña 6 · Metodología
Describe el flujo completo del proyecto: origen de los datos, preprocesamiento,
entrenamiento, validación y métricas de evaluación.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    kpi_row,
    page_header,
    paragraph,
    section_title,
)
from utils.config import N_EMPLEADOS, RANDOM_STATE, TEST_SIZE
from utils.data_loader import get_dataframe, load_metrics
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# Etapas del pipeline metodológico
ETAPAS = [
    (
        "1",
        "Generación de datos",
        "data/generate_data.py",
        "Se simulan las variables del empleado con distribuciones plausibles y "
        "dependencias explícitas entre ellas (la antigüedad está acotada por la "
        "edad, el salario depende del departamento y la experiencia). El abandono "
        "se genera con un proceso logístico conocido más ruido binomial.",
    ),
    (
        "2",
        "Análisis exploratorio",
        "tabs/resultados.py",
        "Se describe la composición de la plantilla, la distribución de cada "
        "variable por grupo de abandono y la estructura de correlaciones, para "
        "verificar que existe señal antes de modelar.",
    ),
    (
        "3",
        "Preprocesamiento",
        "ColumnTransformer",
        "Las seis variables numéricas se estandarizan (media 0, desviación 1) para "
        "que los coeficientes sean comparables; el departamento se codifica con "
        "one-hot encoding descartando una categoría de referencia.",
    ),
    (
        "4",
        "Partición de los datos",
        "train_test_split",
        f"División estratificada {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} "
        "que conserva la proporción de abandonos en ambos conjuntos, con semilla "
        f"fija ({RANDOM_STATE}) para garantizar reproducibilidad.",
    ),
    (
        "5",
        "Entrenamiento",
        "LogisticRegression",
        "Regresión logística con `class_weight='balanced'`, que pondera la clase "
        "minoritaria para que el modelo no se limite a predecir 'permanece' en "
        "todos los casos.",
    ),
    (
        "6",
        "Validación y persistencia",
        "model/model.pkl",
        "Evaluación sobre el conjunto de prueba, validación cruzada de 5 pliegues "
        "sobre entrenamiento y serialización del pipeline completo con joblib.",
    ),
]

# Definición de las métricas de evaluación
METRICAS_DEF = [
    [
        "Accuracy",
        "(VP + VN) / Total",
        "Proporción de aciertos sobre todos los casos.",
        "Engañosa con clases desbalanceadas: predecir siempre 'permanece' ya "
        "acierta ~81%.",
    ],
    [
        "Precisión",
        "VP / (VP + FP)",
        "De los empleados señalados como riesgo, cuántos abandonaron realmente.",
        "Mide el coste de intervenir sobre quien no pensaba irse.",
    ],
    [
        "Recall (sensibilidad)",
        "VP / (VP + FN)",
        "De los que abandonaron, cuántos detectó el modelo.",
        "Métrica prioritaria: no detectar una salida es el error caro.",
    ],
    [
        "F1-score",
        "2·(P·R) / (P + R)",
        "Media armónica entre precisión y recall.",
        "Resume el equilibrio entre ambos errores.",
    ],
    [
        "AUC-ROC",
        "Área bajo la curva ROC",
        "Probabilidad de que el modelo asigne más riesgo a quien sí abandonó.",
        "Independiente del umbral de decisión.",
    ],
]

SUPUESTOS_DATOS = [
    "Los datos son sintéticos: reproducen relaciones plausibles del dominio, no "
    "una empresa real.",
    "No contienen información personal identificable, lo que evita cualquier "
    "problema ético o legal de tratamiento de datos.",
    "El proceso generador es conocido, lo que permite verificar que el modelo "
    "recupera las relaciones esperadas.",
    "La semilla aleatoria está fijada: cualquier persona que ejecute el proyecto "
    "obtiene exactamente el mismo dataset.",
]


def _kpis_metodologia() -> list[dict]:
    """Cifras del diseño experimental."""
    metricas = load_metrics()
    df = get_dataframe()
    return [
        {
            "valor": f"{N_EMPLEADOS:,}",
            "etiqueta": "Registros simulados",
            "detalle": f"{df.shape[1] - 1} variables + objetivo",
            "color": SERIES[0],
        },
        {
            "valor": f"{metricas['n_train']:,}",
            "etiqueta": "Conjunto de entrenamiento",
            "detalle": f"{int((1 - TEST_SIZE) * 100)}% de los datos",
            "color": SERIES[2],
        },
        {
            "valor": f"{metricas['n_test']:,}",
            "etiqueta": "Conjunto de prueba",
            "detalle": f"{int(TEST_SIZE * 100)}% nunca visto por el modelo",
            "color": SERIES[3],
        },
        {
            "valor": f"{metricas['cv_auc_media']:.3f}",
            "etiqueta": "AUC en validación cruzada",
            "detalle": f"5 pliegues · ± {metricas['cv_auc_desviacion']:.3f}",
            "color": SERIES[1],
        },
    ]


def layout() -> html.Div:
    """Layout de la pestaña de metodología."""
    return html.Div(
        [
            page_header(
                "Metodología",
                "Enfoque cuantitativo, de alcance descriptivo y predictivo, con "
                "diseño no experimental de corte transversal.",
                "🧪",
            ),
            kpi_row(_kpis_metodologia()),
            section_title("Flujo de trabajo"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(
                                    [
                                        html.Span(numero, className="etapa-numero"),
                                        html.Code(archivo, className="etapa-archivo"),
                                    ],
                                    className="etapa-head",
                                ),
                                html.Div(titulo, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                            ],
                            color=SERIES[i % len(SERIES)],
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (numero, titulo, archivo, descripcion) in enumerate(ETAPAS)
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "El dataset se construye con un modelo logístico "
                                    "generador: se define un log-odds por empleado a partir "
                                    "de sus características y se extrae la decisión de "
                                    "abandono de una distribución de Bernoulli. Esto "
                                    "garantiza que la relación entre variables y objetivo "
                                    "sea probabilística, no determinista."
                                ),
                                bullet_list(SUPUESTOS_DATOS, color=COLOR_PERMANECE),
                            ],
                            titulo="Sobre los datos",
                            color=COLOR_PERMANECE,
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "El pipeline de scikit-learn encapsula preprocesamiento "
                                    "y modelo en un único objeto. Esa decisión tiene una "
                                    "consecuencia práctica importante: el archivo .pkl "
                                    "acepta datos en crudo, así que el formulario de "
                                    "predicción no necesita replicar el escalado ni la "
                                    "codificación de categorías."
                                ),
                                html.Pre(
                                    "Pipeline([\n"
                                    "    ('preprocesamiento', ColumnTransformer([\n"
                                    "        ('numericas',    StandardScaler(), NUMERIC_FEATURES),\n"
                                    "        ('categoricas',  OneHotEncoder(drop='first'), ['departamento']),\n"
                                    "    ])),\n"
                                    "    ('clasificador', LogisticRegression(\n"
                                    "        max_iter=1000, class_weight='balanced')),\n"
                                    "])",
                                    className="code-block",
                                ),
                                callout(
                                    "El preprocesamiento se ajusta solo con los datos de "
                                    "entrenamiento. Si se ajustara con todo el dataset habría "
                                    "fuga de información y las métricas serían optimistas.",
                                    titulo="Prevención de data leakage",
                                    color=COLOR_ABANDONA,
                                ),
                            ],
                            titulo="Sobre el modelo",
                            color=COLOR_ABANDONA,
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            section_title("Métricas de evaluación"),
            card(
                data_table(
                    ["Métrica", "Fórmula", "Qué mide", "Por qué importa aquí"],
                    METRICAS_DEF,
                ),
                subtitulo="VP: verdaderos positivos · VN: verdaderos negativos · "
                "FP: falsos positivos · FN: falsos negativos.",
                color=SERIES[3],
            ),
        ],
        className="tab-content",
    )
