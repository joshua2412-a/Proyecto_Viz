"""
Dashboard analítico del grado tumoral en gliomas · aplicación principal.

Responsabilidades de este archivo, y solo estas:
  1. Avisar al arrancar si falta el dataset.
  2. Instanciar la app de Dash con Bootstrap.
  3. Componer la barra superior y el sistema de pestañas.
  4. Enrutar la pestaña activa hacia el `layout()` del módulo correspondiente.

Todo el contenido vive en tabs/*.py y toda la lógica de datos en utils/*.py.
El análisis completo, con su desarrollo estadístico, vive en el Jupyter Book de
jbook/ (publicado en GitHub Pages y enlazado desde la pestaña Documentación).

Ejecución:
    python app.py     ->    http://127.0.0.1:8080
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback, dcc, html

from utils.config import DATA_PATH, URL_LIBRO
from utils.theme import BG_PAGE, INK_MUTED, STATUS_CRITICAL

# Módulos de pestañas: cada uno expone una función layout()
from tabs import (
    conclusiones,
    contexto,
    documentacion,
    introduccion,
    limitaciones,
    marco_teorico,
    metodologia,
    objetivos,
    problema,
    resultados,
)

# --------------------------------------------------------------------------- #
# 1. Registro de pestañas: (id, etiqueta, módulo)
#    Añadir una pestaña nueva = crear tabs/mi_pestana.py y sumar una fila aquí.
# --------------------------------------------------------------------------- #
PESTANAS = [
    ("tab-introduccion", "Introducción", introduccion),
    ("tab-contexto", "Contexto clínico", contexto),
    ("tab-problema", "Problema", problema),
    ("tab-objetivos", "Objetivos", objetivos),
    ("tab-marco", "Marco teórico", marco_teorico),
    ("tab-metodologia", "Metodología", metodologia),
    ("tab-resultados", "Resultados", resultados),
    ("tab-limitaciones", "Limitaciones", limitaciones),
    ("tab-conclusiones", "Conclusiones", conclusiones),
    ("tab-documentacion", "Documentación", documentacion),
]

LAYOUTS = {tab_id: modulo.layout for tab_id, _, modulo in PESTANAS}
TAB_INICIAL = PESTANAS[0][0]


# --------------------------------------------------------------------------- #
# 2. Comprobación del dataset
# --------------------------------------------------------------------------- #
def comprobar_dataset() -> None:
    """Avisa al arrancar si falta el CSV, en vez de fallar pestaña a pestaña."""
    if not DATA_PATH.exists():
        print(
            f"[setup] Falta el dataset: {DATA_PATH}\n"
            "[setup] Coloca TCGA_InfoWithGrade.csv en dataset/ (ver dataset/README.md).\n"
            "[setup] La app arrancará, pero las pestañas con datos mostrarán un aviso."
        )


comprobar_dataset()


# --------------------------------------------------------------------------- #
# 3. Instancia de la aplicación
# --------------------------------------------------------------------------- #
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    # Los callbacks de las pestañas apuntan a componentes que no están en el
    # layout inicial (se montan al abrir la pestaña), así que hay que permitirlo.
    suppress_callback_exceptions=True,
    title="Gliomas LGG vs GBM · Dashboard Analítico",
    update_title="Calculando...",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # objeto WSGI que gunicorn sirve en el contenedor


# --------------------------------------------------------------------------- #
# 4. Componentes de la estructura
# --------------------------------------------------------------------------- #
def barra_superior() -> html.Div:
    """Encabezado con el título del tablero, su descripción y el enlace al libro."""
    return html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                "Dashboard analítico · TCGA-LGG y TCGA-GBM",
                                className="brand-eyebrow",
                            ),
                            html.H1(
                                "Grado tumoral en gliomas",
                                className="brand-title",
                            ),
                            html.Div(
                                "Análisis exploratorio de la edad, el perfil "
                                "demográfico y 20 mutaciones genéticas en 839 "
                                "pacientes de TCGA-LGG y TCGA-GBM",
                                className="brand-subtitle",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.A(
                                    "Jupyter Book",
                                    href=URL_LIBRO,
                                    target="_blank",
                                    rel="noopener noreferrer",
                                    className="tech-pill tech-pill-link",
                                ),
                                html.Span("Dash", className="tech-pill"),
                                html.Span("Plotly", className="tech-pill"),
                                html.Span("pandas", className="tech-pill"),
                            ],
                            className="tech-pills",
                        ),
                        md=4,
                        className="d-flex align-items-end justify-content-md-end",
                    ),
                ],
                className="align-items-center",
            ),
            fluid=True,
        ),
        className="topbar",
    )


def navegacion() -> html.Div:
    """Barra de pestañas principal."""
    return html.Div(
        dbc.Container(
            dbc.Tabs(
                [
                    dbc.Tab(
                        label=etiqueta,
                        tab_id=tab_id,
                        # label_* aplica la clase al <a>; tab_* la aplicaria al
                        # <li> de fuera, y entonces el subrayado se dibuja dos veces.
                        label_class_name="main-tab",
                        active_label_class_name="main-tab-active",
                    )
                    for tab_id, etiqueta, _ in PESTANAS
                ],
                id="tabs-principal",
                active_tab=TAB_INICIAL,
                className="main-tabs",
            ),
            fluid=True,
        ),
        className="navbar-tabs",
    )


def pie_pagina() -> html.Div:
    """Pie de página con la ficha técnica del proyecto."""
    return html.Div(
        dbc.Container(
            [
                html.Span("Proyecto de visualización y analítica de datos"),
                html.Span(" · ", className="footer-sep"),
                html.Span("Dataset: Glioma Grading Clinical and Mutation Features (TCGA, 839 pacientes)"),
                html.Span(" · ", className="footer-sep"),
                html.Span("Análisis exploratorio · Dash y Plotly"),
                html.Span(" · ", className="footer-sep"),
                html.A(
                    "Documentación completa",
                    href=URL_LIBRO,
                    target="_blank",
                    rel="noopener noreferrer",
                    className="footer-link",
                ),
            ],
            fluid=True,
        ),
        className="footer",
    )


# --------------------------------------------------------------------------- #
# 5. Layout general
# --------------------------------------------------------------------------- #
app.layout = html.Div(
    [
        barra_superior(),
        navegacion(),
        dbc.Container(
            dcc.Loading(
                html.Div(id="contenido-pestana"),
                type="dot",
                color=INK_MUTED,
                parent_className="loading-wrapper",
            ),
            fluid=True,
            className="page-container",
        ),
        pie_pagina(),
    ],
    className="app-root",
    style={"background": BG_PAGE},
)


# --------------------------------------------------------------------------- #
# 6. Enrutado de pestañas
# --------------------------------------------------------------------------- #
@callback(Output("contenido-pestana", "children"), Input("tabs-principal", "active_tab"))
def mostrar_pestana(tab_activa: str):
    """Devuelve el layout del módulo asociado a la pestaña seleccionada.

    Las pestañas se construyen bajo demanda: solo se renderiza la que el
    usuario está viendo, lo que mantiene ligera la carga inicial.
    """
    constructor = LAYOUTS.get(tab_activa, LAYOUTS[TAB_INICIAL])
    try:
        return constructor()
    except Exception as error:  # noqa: BLE001 - mostramos el error en la interfaz
        return dbc.Alert(
            [
                html.H5("No se pudo construir esta pestaña", className="mb-2"),
                html.Div(str(error)),
                html.Hr(),
                html.Div(
                    "Revisa que exista dataset/TCGA_InfoWithGrade.csv "
                    "(ver dataset/README.md).",
                    className="small",
                ),
            ],
            color="danger",
            className="mt-4",
            style={"borderLeft": f"3px solid {STATUS_CRITICAL}"},
        )


# --------------------------------------------------------------------------- #
# 7. Punto de entrada
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Este bloque SOLO corre al lanzar `python app.py` a mano. En el contenedor
    # y en Cloud Run arranca gunicorn con `app:server`, que ya escucha en
    # 0.0.0.0:8080 por su cuenta (ver Dockerfile), así que aquí no hace falta
    # abrir la aplicación a toda la red: 127.0.0.1 evita que el firewall de
    # Windows pida permiso y que el modo debug quede expuesto en la red local.
    #
    # El material del curso usa host="0.0.0.0" en esta línea. Cambiarlo no
    # afecta al despliegue: el puerto (8080) es el mismo en los dos casos.
    print("\nDashboard disponible en http://127.0.0.1:8080\n")
    app.run(debug=True, host="127.0.0.1", port=8080)
