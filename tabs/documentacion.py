"""
Pestaña 11 · Documentación
Integra el Jupyter Book del proyecto: lo enlaza capítulo a capítulo y lo
embebe para poder leerlo sin salir del dashboard.

El libro se compila desde jbook/ y se publica en la rama gh-pages del
mismo repositorio (joshua2412-a/Proyecto_Viz); aquí solo se consume la versión
publicada.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    enlace_externo,
    page_header,
    paragraph,
    section_title,
)
from utils.config import (
    URL_LIBRO,
    URL_LIBRO_EDA,
    URL_LIBRO_MODELO,
    URL_REPO_LIBRO,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

CAPITULOS = [
    (
        "📕",
        "Portada e introducción",
        "Planteamiento del problema clínico y ficha técnica del dataset.",
        URL_LIBRO,
    ),
    (
        "📊",
        "Análisis exploratorio",
        "Univariado, bivariado, pruebas de hipótesis y diagnóstico de "
        "multicolinealidad, con el desarrollo completo de cada prueba.",
        URL_LIBRO_EDA,
    ),
    (
        "🤖",
        "Modelado · baseline",
        "Preprocesamiento, búsqueda de hiperparámetros, evaluación en prueba e "
        "interpretación de los coeficientes.",
        URL_LIBRO_MODELO,
    ),
    (
        "💻",
        "Repositorio",
        "Código fuente del libro y del sitio publicado en GitHub Pages.",
        URL_REPO_LIBRO,
    ),
]


def layout() -> html.Div:
    """Layout de la pestaña de documentación."""
    return html.Div(
        [
            page_header(
                "Documentación del proyecto",
                "El dashboard resume; el libro demuestra. Aquí está el análisis "
                "completo, con cada prueba estadística y su justificación.",
                "📖",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(icono, className="guide-icon"),
                                html.Div(titulo, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                                html.Div(
                                    enlace_externo("Abrir", url), className="mt-2"
                                ),
                            ],
                            color=SERIES[i % len(SERIES)],
                            className="guide-card",
                        ),
                        lg=3,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (icono, titulo, descripcion, url) in enumerate(CAPITULOS)
                ],
                className="g-3",
            ),
            section_title("Cómo se relacionan las dos piezas"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Libro y dashboard no son dos versiones del mismo "
                                    "contenido: son dos capas del mismo proyecto, y "
                                    "comparten una única copia de los datos."
                                ),
                                bullet_list(
                                    [
                                        html.Span(
                                            [
                                                html.B("dataset/ "),
                                                "guarda el único CSV. Los notebooks y la "
                                                "app leen de ahí, así que ninguna cifra "
                                                "puede desincronizarse.",
                                            ]
                                        ),
                                        html.Span(
                                            [
                                                html.B("jbook/ "),
                                                "contiene la fuente del libro: notebooks, "
                                                "_config.yml y _toc.yml.",
                                            ]
                                        ),
                                        html.Span(
                                            [
                                                html.B("model/train_model.py "),
                                                "reproduce el mismo pipeline del notebook "
                                                "de modelado, con la misma semilla y los "
                                                "mismos hiperparámetros.",
                                            ]
                                        ),
                                        html.Span(
                                            [
                                                html.B("scripts/publicar_libro.py "),
                                                "compila el libro y actualiza el sitio en "
                                                "GitHub Pages.",
                                            ]
                                        ),
                                    ],
                                    color=COLOR_LGG,
                                ),
                                callout(
                                    "Si cambia el dataset, basta reentrenar y recompilar: "
                                    "las cifras del dashboard se recalculan solas y las del "
                                    "libro se regeneran al compilar con ejecución forzada.",
                                    titulo="Una sola fuente de verdad",
                                    color=SERIES[2],
                                ),
                            ],
                            titulo="Arquitectura del proyecto",
                            color=COLOR_LGG,
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph("Para trabajar el libro en local:"),
                                html.Pre(
                                    "pip install -r jbook/requirements.txt\n"
                                    "jupyter-book build jbook/\n"
                                    "# abre jbook/_build/html/index.html\n\n"
                                    "# compilar y publicar en GitHub Pages:\n"
                                    "python scripts/publicar_libro.py",
                                    className="code-block",
                                ),
                                paragraph(
                                    "Por defecto el libro se compila con las salidas ya "
                                    "guardadas en los notebooks (rápido y sin dependencias "
                                    "científicas). Para recalcular todo desde el CSV:"
                                ),
                                html.Pre(
                                    "python scripts/publicar_libro.py --forzar-ejecucion",
                                    className="code-block",
                                ),
                            ],
                            titulo="Compilar y publicar",
                            color=COLOR_GBM,
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("El libro, aquí mismo"),
            card(
                [
                    html.Iframe(
                        src=URL_LIBRO,
                        className="libro-iframe",
                        style={
                            "width": "100%",
                            "height": "78vh",
                            "border": "1px solid #E4E9F0",
                            "borderRadius": "10px",
                            "background": "#FFFFFF",
                        },
                    ),
                    html.Div(
                        [
                            "Si el libro no carga en este marco, ábrelo en una pestaña "
                            "nueva: ",
                            enlace_externo("Jupyter Book del proyecto", URL_LIBRO),
                        ],
                        className="graph-note mt-2",
                    ),
                ],
                titulo="Vista embebida",
                subtitulo="Versión publicada en GitHub Pages",
                color=SERIES[3],
            ),
        ],
        className="tab-content",
    )
