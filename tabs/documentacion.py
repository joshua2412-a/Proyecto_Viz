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
    badge_numero,
    bullet_list,
    callout,
    card,
    enlace_externo,
    page_header,
    paragraph,
    section_title,
)
from utils.config import (
    AUTORES,
    URL_LIBRO,
    URL_LIBRO_EDA,
    URL_REPO_LIBRO,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

CAPITULOS = [
    (
        "bi-file-earmark-text",
        "Portada e introducción",
        "Planteamiento del problema clínico y ficha técnica del dataset.",
        URL_LIBRO,
    ),
    (
        "bi-bar-chart-line",
        "Análisis exploratorio",
        "Univariado, bivariado, pruebas de hipótesis y diagnóstico de "
        "multicolinealidad, con el desarrollo completo de cada prueba.",
        URL_LIBRO_EDA,
    ),
    (
        "bi-code-slash",
        "Repositorio",
        "Código fuente del libro y del sitio publicado en GitHub Pages.",
        URL_REPO_LIBRO,
    ),
]


# Perfiles que puede tener un autor: (clave en AUTORES, etiqueta, icono).
ENLACES_AUTOR = (
    ("github", "GitHub", "bi-github"),
    ("linkedin", "LinkedIn", "bi-linkedin"),
)


def _iniciales(nombre: str) -> str:
    """Las dos primeras iniciales del nombre: "Alejandro Cantillo..." -> "AC"."""
    return "".join(parte[0] for parte in nombre.split()[:2]).upper()


def ficha_autor(autor: dict) -> dbc.Card:
    """Tarjeta de un autor: monograma, nombre y los perfiles que tenga.

    Los enlaces se dibujan solo si existen en AUTORES, así que un perfil que
    falte no deja un botón que lleve a ninguna parte.
    """
    nombre = autor["nombre"]
    enlaces = [
        html.A(
            [html.I(className=f"bi {icono}"), html.Span(etiqueta)],
            href=autor[clave],
            target="_blank",
            rel="noopener noreferrer",
            className="autor-enlace",
            **{"aria-label": f"{etiqueta} de {nombre}"},
        )
        for clave, etiqueta, icono in ENLACES_AUTOR
        if autor.get(clave)
    ]

    return card(
        html.Div(
            [
                html.Div(
                    _iniciales(nombre),
                    className="autor-monograma",
                    **{"aria-hidden": "true"},
                ),
                html.Div(
                    [
                        html.Div(nombre, className="autor-nombre"),
                        html.Div(enlaces, className="autor-enlaces"),
                    ],
                    className="autor-cuerpo",
                ),
            ],
            className="autor-ficha",
        ),
    )


def layout() -> html.Div:
    """Layout de la pestaña de documentación."""
    return html.Div(
        [
            page_header(
                "Documentación del proyecto",
                "El dashboard cubre el análisis exploratorio; el libro lo demuestra "
                "paso a paso, con el desarrollo de cada prueba.",
                "bi-book",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(badge_numero(html.I(className=f"bi {icono}")),
                                         className="guide-badge"),
                                html.Div(titulo, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                                html.Div(
                                    enlace_externo("Abrir", url), className="mt-2"
                                ),
                            ],
                            className="guide-card",
                        ),
                        lg=4,
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
                                                html.B("utils/ "),
                                                "es la única puerta de entrada a los "
                                                "datos: carga el CSV, reconstruye la "
                                                "partición y calcula los contrastes que "
                                                "consumen las pestañas.",
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
                                ),
                                callout(
                                    "Si cambia el dataset, basta recompilar: las cifras del "
                                    "dashboard se recalculan solas y las del libro se "
                                    "regeneran al compilar con ejecución forzada.",
                                    titulo="Una sola fuente de verdad",
                                ),
                            ],
                            titulo="Arquitectura del proyecto",
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
            ),
            section_title("Autores"),
            dbc.Row(
                [
                    dbc.Col(ficha_autor(autor), md=6, xs=12, className="mb-3")
                    for autor in AUTORES
                ],
                className="g-3",
            ),
        ],
        className="vista-pestana",
    )
