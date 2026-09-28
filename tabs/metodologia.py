"""
Pestaña 6 · Metodología
Cómo se pasó del CSV al modelo entrenado: partición, preprocesamiento,
búsqueda de hiperparámetros y evaluación. Todo reproducible con una semilla.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    enlace_externo,
    kpi_row,
    page_header,
    paragraph,
    section_title,
)
from utils.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    GENE_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    URL_LIBRO_EDA,
)
from utils.data_loader import load_metrics, proporcion_grado, tamanos_particion
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

ETAPAS = [
    (
        "1",
        "Verificación del dataset",
        "839 registros y 24 variables, sin valores nulos ni faltantes. El único "
        "registro duplicado corresponde a dos pacientes distintos que comparten "
        "fenotipo y perfil genético, así que se conserva. Las variables binarias se "
        "convierten a tipo categórico para el análisis exploratorio.",
    ),
    (
        "2",
        "Partición estratificada 80/20",
        "Se reserva el 20 % de los pacientes para la validación final antes de mirar "
        "cualquier estadístico, de modo que el EDA no filtre información del conjunto "
        "de prueba. La estratificación mantiene la proporción LGG/GBM en ambos lados.",
    ),
    (
        "3",
        "Análisis exploratorio sobre entrenamiento",
        "Univariado (distribuciones, normalidad, valores atípicos), bivariado contra "
        "el grado (Mann-Whitney para la edad, chi-cuadrado y V de Cramér para las "
        "categóricas) y diagnóstico de multicolinealidad entre mutaciones.",
    ),
    (
        "4",
        "Preprocesamiento en pipeline",
        "Un ColumnTransformer aplica a cada tipo de variable su transformación y va "
        "dentro del mismo objeto que el modelo: el .pkl acepta datos en crudo y no "
        "hay riesgo de aplicar un escalado distinto en inferencia.",
    ),
    (
        "5",
        f"Búsqueda de hiperparámetros con validación cruzada de {CV_FOLDS} particiones",
        "GridSearchCV sobre 52 combinaciones (13 valores de C × 2 penalizaciones × 2 "
        "configuraciones de class_weight), optimizando AUC-ROC por ser robusta ante "
        "el desbalance de clases.",
    ),
    (
        "6",
        "Evaluación sobre el conjunto reservado",
        "Métricas de clasificación binaria más los tiempos de búsqueda, entrenamiento "
        "e inferencia, para poder comparar el baseline con modelos más complejos en "
        "las dos dimensiones: error y coste computacional.",
    ),
]


def _kpis() -> list[dict]:
    """Cifras de la partición y de la validación cruzada."""
    tamanos = tamanos_particion()
    metricas = load_metrics()
    proporciones = proporcion_grado("train").set_index("grado")

    return [
        {
            "valor": f"{tamanos['train']}",
            "etiqueta": "Pacientes en entrenamiento",
            "detalle": f"LGG {proporciones.loc['LGG', 'porcentaje']:.1f}% · "
                       f"GBM {proporciones.loc['GBM', 'porcentaje']:.1f}%",
            "color": COLOR_LGG,
        },
        {
            "valor": f"{tamanos['test']}",
            "etiqueta": "Pacientes en prueba",
            "detalle": f"Reservados antes del EDA ({TEST_SIZE:.0%} del total)",
            "color": COLOR_GBM,
        },
        {
            "valor": f"{metricas['cv_auc_media']:.3f}",
            "etiqueta": f"AUC-ROC en CV ({CV_FOLDS}-fold)",
            "detalle": f"Desviación ± {metricas['cv_auc_desviacion']:.3f}",
            "color": SERIES[2],
        },
        {
            "valor": f"seed {RANDOM_STATE}",
            "etiqueta": "Reproducibilidad",
            "detalle": "Misma semilla en libro y dashboard",
            "color": SERIES[3],
        },
    ]


def _tabla_preprocesamiento() -> dbc.Table:
    """Qué transformación recibe cada bloque de variables y por qué."""
    return data_table(
        ["Bloque", "Variables", "Transformación", "Motivo"],
        [
            [
                "Clínica numérica",
                ", ".join(NUMERIC_FEATURES),
                "StandardScaler",
                "La regresión logística regularizada es sensible a la escala: sin "
                "estandarizar, la penalización castigaría a la edad por estar medida "
                "en años.",
            ],
            [
                "Clínicas categóricas",
                ", ".join(CATEGORICAL_FEATURES),
                "OneHotEncoder(drop='if_binary', handle_unknown='ignore')",
                "Race tiene cuatro niveles sin orden natural; Gender es binaria y "
                "basta una columna.",
            ],
            [
                "Mutacionales",
                f"{len(GENE_FEATURES)} genes",
                "passthrough",
                "Ya vienen codificadas como indicadores 0/1: transformarlas no "
                "añadiría nada.",
            ],
        ],
    )


def _tabla_hiperparametros() -> dbc.Table:
    """Configuración seleccionada por la búsqueda y su lectura."""
    hiperparametros = load_metrics()["hiperparametros"]
    lecturas = {
        "C": "Penalización moderadamente fuerte (C < 1): favorece la generalización "
             "dado el tamaño de la muestra.",
        "penalty": "L1 (Lasso): lleva a cero las variables poco informativas, "
                   "haciendo selección automática de variables.",
        "solver": "liblinear: el solver que admite penalización L1 en scikit-learn.",
        "class_weight": "Sin reponderar: el desbalance 58/42 no es lo bastante severo "
                        "como para requerirlo.",
    }
    return data_table(
        ["Hiperparámetro", "Valor", "Lectura"],
        [
            [
                clave,
                f"{valor}" if not isinstance(valor, float) else f"{valor:.4f}",
                lecturas.get(clave, ""),
            ]
            for clave, valor in hiperparametros.items()
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de metodología."""
    metricas = load_metrics()
    tiempos = metricas["tiempos_segundos"]

    return html.Div(
        [
            page_header(
                "Metodología",
                "Del CSV al modelo entrenado, en seis etapas reproducibles con la "
                "misma semilla que usa el Jupyter Book.",
                "🧪",
            ),
            kpi_row(_kpis()),
            section_title("Las seis etapas"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(numero, className="guide-icon"),
                                html.Div(titulo, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                            ],
                            color=SERIES[i % len(SERIES)],
                            className="guide-card",
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (numero, titulo, descripcion) in enumerate(ETAPAS)
                ],
                className="g-3",
            ),
            section_title("Preprocesamiento"),
            card(
                [
                    paragraph(
                        "El preprocesamiento vive dentro del pipeline, no antes de él. Esa "
                        "decisión es la que permite que el formulario de la pestaña de "
                        "predicción envíe la edad en años y las mutaciones como 0/1, sin "
                        "replicar a mano ninguna transformación."
                    ),
                    _tabla_preprocesamiento(),
                    callout(
                        "Se usa el conjunto completo de variables disponibles: la reducción "
                        "del panel no se hace a mano recortando columnas, sino dejando que "
                        "la penalización L1 decida cuáles sobreviven. Así la selección "
                        "queda documentada por el propio modelo.",
                        titulo="Estrategia de selección de variables",
                        color=SERIES[2],
                    ),
                ],
                titulo="Qué se le hace a cada variable",
                color=COLOR_LGG,
            ),
            section_title("Hiperparámetros y coste computacional"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            _tabla_hiperparametros(),
                            titulo="Configuración seleccionada",
                            subtitulo=f"Mejor combinación de las 52 evaluadas, por AUC-ROC en "
                                      f"validación cruzada de {CV_FOLDS} particiones",
                            color=COLOR_GBM,
                        ),
                        lg=8,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                bullet_list(
                                    [
                                        f"Búsqueda de hiperparámetros: {tiempos['busqueda']:.2f} s"
                                        if tiempos["busqueda"]
                                        else "Búsqueda de hiperparámetros: no se repitió "
                                             "(se reutiliza la configuración del libro; "
                                             "ejecuta train_model.py --buscar para repetirla)",
                                        f"Entrenamiento final: {tiempos['entrenamiento']:.3f} s",
                                        f"Inferencia sobre las {metricas['n_test']} "
                                        f"observaciones de prueba: {tiempos['inferencia']:.4f} s",
                                    ],
                                    color=SERIES[3],
                                ),
                                paragraph(
                                    "Los tiempos se separan a propósito: permiten comparar "
                                    "de forma justa este baseline con modelos que no "
                                    "necesitan una búsqueda exhaustiva."
                                ),
                                enlace_externo(
                                    "Ver el EDA completo en el libro", URL_LIBRO_EDA
                                ),
                            ],
                            titulo="Coste computacional",
                            color=SERIES[3],
                        ),
                        lg=4,
                        className="mb-4",
                    ),
                ]
            ),
        ],
        className="tab-content",
    )
