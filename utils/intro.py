"""
Portada animada del tablero: un cerebro de partículas con el glioma dentro.

Este módulo solo arma el marcado. La animación vive en assets/intro.js (WebGL
puro, sin librerías) y el estilo en assets/style.css, sección "Portada".

Las cifras salen del dataset, no se escriben a mano: el número de partículas
grandes de cada grado es el número de pacientes de ese grado en TCGA.
"""

from __future__ import annotations

import pandas as pd
from dash import html

from utils.components import icono
from utils.config import AUTORES, DATA_PATH


def _conteos() -> tuple[int, int]:
    """Pacientes LGG (Grade = 0) y GBM (Grade = 1) del dataset completo."""
    grados = pd.read_csv(DATA_PATH, usecols=["Grade"])["Grade"]
    return int((grados == 0).sum()), int((grados == 1).sum())


def _rasgo(clase: str, texto: str) -> html.Li:
    return html.Li([html.Span(className=f"intro-punto {clase}"), html.Span(texto)])


def portada() -> html.Div:
    """Capa a pantalla completa que se muestra antes del tablero."""
    n_lgg, n_gbm = _conteos()
    total = n_lgg + n_gbm
    autores = " · ".join(a["nombre"] for a in AUTORES)
    return html.Div(
        [
            html.Canvas(className="intro-lienzo", **{"aria-hidden": "true"}),
            html.Div(
                [
                    html.H1(
                        ["Gliomas de bajo grado", html.Br(), "frente a glioblastoma"],
                        className="intro-titulo",
                    ),
                    html.P(
                        f"{total} pacientes, 23 variables clínicas y genéticas y tres "
                        "modelos para distinguir un grado del otro.",
                        className="intro-texto",
                    ),
                    html.Div(
                        [
                            html.Button("GBM", className="intro-modo",
                                        **{"data-modo": "gbm", "aria-pressed": "true"}),
                            html.Button("LGG", className="intro-modo",
                                        **{"data-modo": "lgg", "aria-pressed": "false"}),
                        ],
                        className="intro-segmentado",
                        role="group",
                        **{"aria-label": "Grado que se muestra en el cerebro"},
                    ),
                    html.Ul(
                        [
                            _rasgo("rim", "Anillo grueso e irregular que capta contraste"),
                            _rasgo("nec", "Necrosis en el centro"),
                            _rasgo("ede", "Edema e infiltración alrededor"),
                        ],
                        className="intro-rasgos intro-rasgos-gbm",
                    ),
                    html.Ul(
                        [
                            _rasgo("lgg", "Lesión homogénea y difusa, sin realce"),
                            _rasgo("lgg-borde", "Bordes bastante definidos"),
                            _rasgo("sin", "Sin necrosis y con poco edema"),
                        ],
                        className="intro-rasgos intro-rasgos-lgg",
                    ),
                    html.Button(
                        [
                            html.Span("Explorar el dashboard"),
                            html.Span(icono("arrow_forward"),
                                      className="intro-entrar-icono"),
                        ],
                        className="intro-entrar",
                    ),
                    html.Div(autores, className="intro-autores"),
                ],
                className="intro-panel",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Strong("Glioblastoma (GBM)"),
                         html.Span(f"{n_gbm} pacientes")],
                        className="intro-etiqueta-gbm",
                    ),
                    html.Div(
                        [html.Strong("Glioma de bajo grado (LGG)"),
                         html.Span(f"{n_lgg} pacientes")],
                        className="intro-etiqueta-lgg",
                    ),
                ],
                className="intro-etiqueta",
            ),
            html.Div([icono("drag_pan"),
                      html.Span("Arrastra para girar el cerebro")],
                     className="intro-pista"),
            html.Div("Ubicación ilustrativa: el dataset no registra dónde estaba cada tumor.",
                     className="intro-nota"),
        ],
        id="intro",
        className="intro",
        **{"data-lgg": str(n_lgg), "data-gbm": str(n_gbm), "data-modo": "gbm"},
    )
