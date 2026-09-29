"""
Pestaña 1 · Introducción
Presenta el problema de la caracterización del grado tumoral en gliomas y la
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
from utils.data_loader import (
    asociacion_con_grado,
    prevalencia_genes,
    proporcion_grado,
    tamanos_particion,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

# Guía de navegación: (icono, pestaña, qué encontrará el usuario)
GUIA = [
    ("bi-info-circle", "Introducción", "Qué se analiza aquí y cómo leer este tablero."),
    ("bi-heart-pulse", "Contexto clínico", "Qué son LGG y GBM y por qué importa distinguirlos."),
    ("bi-exclamation-circle", "Problema", "El coste de la secuenciación completa como cuello de botella."),
    ("bi-bullseye", "Objetivos", "Qué se propone resolver el proyecto y con qué criterio."),
    ("bi-journal-text", "Marco teórico", "Las variables del panel y las pruebas estadísticas."),
    ("bi-clipboard-data", "Metodología", "Partición, control de calidad y contrastes aplicados."),
    ("bi-bar-chart-line", "Resultados", "El análisis exploratorio, variable a variable."),
    ("bi-exclamation-triangle", "Limitaciones", "Qué NO se puede concluir con este trabajo."),
    ("bi-check2-circle", "Conclusiones", "Hallazgos del EDA y siguiente paso del proyecto."),
    ("bi-book", "Documentación", "El Jupyter Book y el repositorio del proyecto."),
]


def _kpis() -> list[dict]:
    """Indicadores de portada, todos calculados a partir del propio dataset."""
    tamanos = tamanos_particion()
    proporciones = proporcion_grado("full").set_index("grado")
    asociacion = asociacion_con_grado("train")
    idh1 = prevalencia_genes("train").set_index("gen").loc["IDH1"]

    return [
        {
            "valor": f"{tamanos['full']}",
            "etiqueta": "Pacientes analizados",
            "detalle": f"{tamanos['train']} entrenamiento · {tamanos['test']} prueba",
        },
        {
            "valor": f"{proporciones.loc['GBM', 'porcentaje']:.1f}%",
            "etiqueta": "Casos de GBM",
            "detalle": "El resto son gliomas de bajo grado (LGG)",
        },
        {
            "valor": f"{int(asociacion['significativa'].sum())} de {len(asociacion)}",
            "etiqueta": "Variables con señal",
            "detalle": "Asociadas al grado con p < 0,05 y |r| ≥ 0,10",
        },
        {
            "valor": f"{idh1['LGG'] - idh1['GBM']:.0f} pp",
            "etiqueta": "Brecha de IDH1",
            "detalle": f"Mutado en {idh1['LGG']:.1f}% de LGG frente a {idh1['GBM']:.1f}% de GBM",
        },
    ]


def layout() -> html.Div:
    """Layout de la pestaña de introducción."""
    return html.Div(
        [
            page_header(
                "Gliomas: qué separa a un LGG de un GBM",
                "Análisis exploratorio de 839 pacientes de los proyectos TCGA-LGG y "
                "TCGA-GBM, variable clínica y mutación a mutación.",
                "bi-info-circle",
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
                                    "Este tablero aborda la primera mitad de ese desafío: "
                                    "identificar qué variables clínicas y qué mutaciones "
                                    "separan realmente a los dos grupos, y con qué fuerza. "
                                    "Es el paso que decide qué merece la pena medir antes "
                                    "de construir cualquier modelo."
                                ),
                            ],
                            titulo="El problema en una página",
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
                                    ),
                                    callout(
                                        "La variable objetivo es binaria: 1 si el paciente "
                                        "presenta glioblastoma multiforme (GBM), 0 si el "
                                        "glioma es de bajo grado (LGG).",
                                        titulo="Variable a predecir",
                                    ),
                                ],
                                titulo="Ficha técnica del dataset",
                            ),
                            card(
                                [
                                    paragraph(
                                        "Este tablero es la cara interactiva del análisis "
                                        "exploratorio. El desarrollo estadístico completo, "
                                        "con el detalle de cada prueba, vive en el Jupyter "
                                        "Book del proyecto."
                                    ),
                                    enlace_externo("Abrir el Jupyter Book", URL_LIBRO),
                                ],
                                titulo="Documentación del proyecto",
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
                                html.Div(html.I(className=f"bi {icono}"), className="guide-icon"),
                                html.Div(nombre, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                            ],
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
        className="vista-pestana",
    )
