"""
Pestaña 1 · Introducción
Presenta el problema de la clasificación del grado tumoral en gliomas y la
guía de lectura del dashboard.

Cada pestaña expone una única función pública `layout()` y es totalmente
autónoma: no importa nada de las demás pestañas.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    enlace_externo,
    kpi_row,
    page_header,
    paragraph,
    section_title,
)
from utils.config import URL_DATASET, URL_LIBRO
from utils.data_loader import load_metrics, tamanos_particion
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

# Guía de navegación: (icono, pestaña, qué encontrará el usuario)
GUIA = [
    ("📘", "Introducción", "Qué se clasifica aquí y cómo leer este tablero."),
    ("🧬", "Contexto clínico", "Qué son LGG y GBM y por qué importa distinguirlos."),
    ("❗", "Problema", "El coste de la secuenciación completa como cuello de botella."),
    ("🎯", "Objetivos", "Qué se propone resolver el proyecto y con qué criterio."),
    ("📚", "Marco teórico", "Genes del panel, regresión logística y odds ratios."),
    ("🧪", "Metodología", "Partición, preprocesamiento y búsqueda de hiperparámetros."),
    ("📊", "Resultados", "Análisis exploratorio y desempeño del clasificador."),
    ("🔮", "Predicción", "Simulador: perfil clínico-molecular de un paciente."),
    ("⚠️", "Limitaciones", "Qué NO se puede concluir con este trabajo."),
    ("✅", "Conclusiones", "Hallazgos y siguiente paso del proyecto."),
    ("📖", "Documentación", "El Jupyter Book con el análisis completo."),
]


def _kpis() -> list[dict]:
    """Indicadores de portada calculados a partir de los datos y las métricas."""
    metricas = load_metrics()
    tamanos = tamanos_particion()

    return [
        {
            "valor": f"{tamanos['full']}",
            "etiqueta": "Pacientes analizados",
            "detalle": f"{tamanos['train']} entrenamiento · {tamanos['test']} prueba",
            "color": SERIES[0],
        },
        {
            "valor": f"{metricas['prevalencia_gbm']:.1%}",
            "etiqueta": "Casos de GBM",
            "detalle": "El resto son gliomas de bajo grado (LGG)",
            "color": SERIES[1],
        },
        {
            "valor": f"{metricas['roc_auc']:.3f}",
            "etiqueta": "AUC-ROC en prueba",
            "detalle": f"Accuracy {metricas['accuracy']:.1%} · Recall GBM {metricas['recall']:.1%}",
            "color": SERIES[2],
        },
        {
            "valor": f"{metricas['n_variables_activas']} de {metricas['n_columnas_modelo']}",
            "etiqueta": "Columnas activas",
            "detalle": "La penalización L1 deja el resto en cero",
            "color": SERIES[3],
        },
    ]


def layout() -> html.Div:
    """Layout de la pestaña de introducción."""
    return html.Div(
        [
            page_header(
                "Gliomas: distinguir LGG de GBM sin secuenciar de más",
                "Tablero analítico sobre 839 pacientes de los proyectos TCGA-LGG y "
                "TCGA-GBM, con un clasificador de regresión logística.",
                "📘",
            ),
            kpi_row(_kpis()),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Los gliomas son el tumor cerebral primario más común "
                                    "en adultos y se clasifican principalmente en gliomas "
                                    "de bajo grado (LGG) y glioblastoma multiforme (GBM). "
                                    "La diferencia no es de matiz: cambia el pronóstico, la "
                                    "agresividad del tratamiento y el seguimiento."
                                ),
                                paragraph(
                                    "Los criterios histológicos e imagenológicos han sido la "
                                    "base del diagnóstico, pero la caracterización "
                                    "biomolecular se ha vuelto imprescindible para decidir "
                                    "el tratamiento idóneo. El problema es que la "
                                    "secuenciación genética completa tiene un coste elevado "
                                    "para los sistemas de salud y para los pacientes."
                                ),
                                paragraph(
                                    "Este proyecto aborda ese desafío con dos piezas: un "
                                    "análisis exploratorio que identifica qué variables "
                                    "clínicas y qué mutaciones separan realmente a los dos "
                                    "grupos, y un modelo de clasificación que estima la "
                                    "probabilidad de GBM a partir de ese subconjunto "
                                    "reducido de marcadores."
                                ),
                            ],
                            titulo="El problema en una página",
                            color=COLOR_LGG,
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        [
                            card(
                                [
                                    bullet_list(
                                        [
                                            html.Span(
                                                [
                                                    html.B("Nombre: "),
                                                    "Glioma Grading Clinical and Mutation "
                                                    "Features",
                                                ]
                                            ),
                                            html.Span(
                                                [
                                                    html.B("Fuente: "),
                                                    enlace_externo(
                                                        "UCI Machine Learning Repository",
                                                        URL_DATASET,
                                                    ),
                                                ]
                                            ),
                                            html.Span([html.B("Muestra: "), "839 pacientes"]),
                                            html.Span(
                                                [
                                                    html.B("Atributos: "),
                                                    "23 (20 genes con alta frecuencia de "
                                                    "mutación y 3 variables clínicas)",
                                                ]
                                            ),
                                            html.Span(
                                                [
                                                    html.B("Objetivo: "),
                                                    "clasificación binaria LGG vs GBM",
                                                ]
                                            ),
                                            html.Span(
                                                [
                                                    html.B("Naturaleza: "),
                                                    "tabular, multivariada (numérica y "
                                                    "categórica), sin valores faltantes",
                                                ]
                                            ),
                                        ],
                                        color=COLOR_LGG,
                                    ),
                                    callout(
                                        "La variable objetivo es binaria: 1 si el paciente "
                                        "presenta glioblastoma multiforme (GBM), 0 si el "
                                        "glioma es de bajo grado (LGG).",
                                        titulo="Variable a predecir",
                                        color=COLOR_GBM,
                                    ),
                                ],
                                titulo="Ficha técnica del dataset",
                                color=SERIES[3],
                            ),
                            card(
                                [
                                    paragraph(
                                        "Este tablero es la cara interactiva del análisis. "
                                        "El desarrollo estadístico completo —pruebas de "
                                        "hipótesis, supuestos, búsqueda de hiperparámetros— "
                                        "vive en el Jupyter Book del proyecto."
                                    ),
                                    enlace_externo("Abrir el Jupyter Book", URL_LIBRO),
                                ],
                                titulo="Documentación del proyecto",
                                color=SERIES[2],
                                className="mt-3",
                            ),
                        ],
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Cómo recorrer el tablero"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(icono, className="guide-icon"),
                                html.Div(nombre, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                            ],
                            color=SERIES[i % len(SERIES)],
                            className="guide-card",
                        ),
                        lg=3,
                        md=4,
                        sm=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (icono, nombre, descripcion) in enumerate(GUIA)
                ],
                className="g-3",
            ),
        ],
        className="tab-content",
    )
