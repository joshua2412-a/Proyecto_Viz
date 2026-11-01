"""
Pestaña 13 · Documentación
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
    icono as icono_ms,
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
    URL_LIBRO_MODELADO,
    URL_REPO_LIBRO,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

CAPITULOS = [
    (
        "description",
        "Portada e introducción",
        "Planteamiento del problema clínico y ficha técnica del dataset.",
        URL_LIBRO,
    ),
    (
        "bar_chart",
        "Análisis exploratorio",
        "Univariado, bivariado, pruebas de hipótesis y diagnóstico de "
        "multicolinealidad, con el desarrollo completo de cada prueba.",
        URL_LIBRO_EDA,
    ),
    (
        "account_tree",
        "Modelado predictivo",
        "Las 48 combinaciones, la validación anidada, la evaluación en prueba, la "
        "interpretabilidad y las pruebas estadísticas entre modelos.",
        URL_LIBRO_MODELADO,
    ),
    (
        "data_object",
        "Repositorio",
        "Código fuente del libro y del sitio publicado en GitHub Pages.",
        URL_REPO_LIBRO,
    ),
]


# Perfiles que puede tener un autor: (clave en AUTORES, etiqueta, icono).
ENLACES_AUTOR = (
    ("github", "GitHub", "code"),
    ("linkedin", "LinkedIn", "badge"),
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
            [icono_ms(icono), html.Span(etiqueta)],
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
                "El dashboard resume el análisis exploratorio y el modelado; el libro "
                "los desarrolla paso a paso, con cada prueba y el código que la produce.",
                "book",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(badge_numero(icono_ms(icono)),
                                         className="guide-badge"),
                                html.Div(titulo, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                                html.Div(
                                    enlace_externo("Abrir", url), className="mt-2"
                                ),
                            ],
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
                                                "app leen de ahí, así que parten de los "
                                                "mismos datos.",
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
                                    "Si cambia el dataset, hay que volver a ejecutar los dos "
                                    "notebooks, cada uno con su entorno, y exportar los modelos. "
                                    "Las cifras que calcula el dashboard se actualizan solas; "
                                    "las que van escritas en los textos hay que revisarlas.",
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
                                # Tres entornos porque sus dependencias chocan; el
                                # notebook de modelado nunca se ejecuta con el del
                                # libro (sus resultados salen de otras versiones).
                                paragraph(
                                    "El proyecto usa tres entornos, porque sus "
                                    "dependencias no son compatibles entre sí:"
                                ),
                                bullet_list([
                                    "Dashboard (requirements.txt): Dash y lo necesario "
                                    "para servir el tablero.",
                                    "Libro y EDA (jbook/requirements.txt): Jupyter Book 1.x, "
                                    "pandas 2 y scikit-learn 1.7.",
                                    "Modelado (jbook/requirements-modelado.txt): pandas 3, "
                                    "scikit-learn 1.9, XGBoost, Optuna y DEAP, con las "
                                    "versiones exactas registradas en "
                                    "jbook/resultados/entorno.json.",
                                ]),
                                paragraph(
                                    "Cada notebook se ejecuta con su entorno y el libro se "
                                    "publica con las salidas guardadas. Los dos usan la misma "
                                    "partición (semilla 42, estratificada 80/20): los 168 "
                                    "pacientes de prueba son idénticos."
                                ),
                                html.Pre(
                                    "# modelado (entorno gliomas-modelado, desde jbook/)\n"
                                    "python -m modelado.experimentos\n"
                                    "jupyter nbconvert --to notebook --execute --inplace "
                                    "02_Modelado.ipynb\n"
                                    "python -m modelado.exportar\n\n"
                                    "# libro (entorno gliomas-libro)\n"
                                    "jupyter-book build jbook/\n"
                                    "python scripts/publicar_libro.py",
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
