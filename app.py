"""
Dashboard de rotación laboral (employee attrition) · aplicación principal.

Responsabilidades de este archivo, y solo estas:
  1. Verificar que existan los artefactos (dataset y modelo) y crearlos si faltan.
  2. Instanciar la app de Dash con Bootstrap.
  3. Componer la barra superior y el sistema de pestañas.
  4. Enrutar la pestaña activa hacia el `layout()` del módulo correspondiente.

Todo el contenido vive en tabs/*.py y toda la lógica de datos en utils/*.py.

Ejecución:
    python app.py     ->    http://127.0.0.1:8050
"""

from __future__ import annotations

import subprocess
import sys

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback, dcc, html

from utils.config import BASE_DIR, DATA_PATH, METRICS_PATH, MODEL_PATH
from utils.theme import BG_CARD, COLOR_ABANDONA, COLOR_PERMANECE

# Módulos de pestañas: cada uno expone una función layout()
from tabs import (
    conclusiones,
    contexto,
    introduccion,
    limitaciones,
    marco_teorico,
    metodologia,
    objetivos,
    prediccion,
    problema,
    resultados,
)

# --------------------------------------------------------------------------- #
# 1. Registro de pestañas: (id, etiqueta, módulo)
#    Añadir una pestaña nueva = crear tabs/mi_pestana.py y sumar una fila aquí.
# --------------------------------------------------------------------------- #
PESTANAS = [
    ("tab-introduccion", "📘 Introducción", introduccion),
    ("tab-contexto", "🏢 Contexto", contexto),
    ("tab-problema", "❗ Problema", problema),
    ("tab-objetivos", "🎯 Objetivos", objetivos),
    ("tab-marco", "📚 Marco teórico", marco_teorico),
    ("tab-metodologia", "🧪 Metodología", metodologia),
    ("tab-resultados", "📊 Resultados", resultados),
    ("tab-prediccion", "🔮 Predicción", prediccion),
    ("tab-limitaciones", "⚠️ Limitaciones", limitaciones),
    ("tab-conclusiones", "✅ Conclusiones", conclusiones),
]

LAYOUTS = {tab_id: modulo.layout for tab_id, _, modulo in PESTANAS}
TAB_INICIAL = PESTANAS[0][0]


# --------------------------------------------------------------------------- #
# 2. Preparación de artefactos
# --------------------------------------------------------------------------- #
def preparar_artefactos() -> None:
    """Genera el dataset y entrena el modelo si aún no existen.

    Hace que el proyecto arranque con un único comando la primera vez, sin
    quitarle independencia a los scripts: `generate_data.py` y `train_model.py`
    siguen siendo ejecutables por separado.
    """
    if not DATA_PATH.exists():
        print("[setup] No se encontró el dataset. Generando datos sintéticos...")
        subprocess.run(
            [sys.executable, str(BASE_DIR / "data" / "generate_data.py")], check=True
        )

    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        print("[setup] No se encontró el modelo entrenado. Entrenando...")
        subprocess.run(
            [sys.executable, str(BASE_DIR / "model" / "train_model.py")], check=True
        )


preparar_artefactos()


# --------------------------------------------------------------------------- #
# 3. Instancia de la aplicación
# --------------------------------------------------------------------------- #
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    # Los callbacks de las pestañas apuntan a componentes que no están en el
    # layout inicial (se montan al abrir la pestaña), así que hay que permitirlo.
    suppress_callback_exceptions=True,
    title="Rotación Laboral · Dashboard Analítico",
    update_title="Calculando...",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expuesto para despliegue con gunicorn/waitress


# --------------------------------------------------------------------------- #
# 4. Componentes de la estructura
# --------------------------------------------------------------------------- #
def barra_superior() -> html.Div:
    """Encabezado con el título del tablero y su descripción."""
    return html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Dashboard analítico", className="brand-eyebrow"),
                            html.H1("Rotación Laboral", className="brand-title"),
                            html.Div(
                                "Análisis descriptivo y predictivo del abandono de "
                                "empleados con regresión logística",
                                className="brand-subtitle",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Span("Dash", className="tech-pill"),
                                html.Span("Plotly", className="tech-pill"),
                                html.Span("scikit-learn", className="tech-pill"),
                                html.Span("Bootstrap", className="tech-pill"),
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
                        tab_class_name="main-tab",
                        active_tab_class_name="main-tab-active",
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
                html.Span("Dataset sintético de 2.000 empleados"),
                html.Span(" · ", className="footer-sep"),
                html.Span("Modelo: regresión logística (scikit-learn)"),
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
                color=COLOR_PERMANECE,
                parent_className="loading-wrapper",
            ),
            fluid=True,
            className="page-container",
        ),
        pie_pagina(),
    ],
    className="app-root",
    style={"background": BG_CARD},
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
                    "Revisa que existan data/employee_attrition.csv y model/model.pkl. "
                    "Puedes regenerarlos con: python data/generate_data.py && "
                    "python model/train_model.py",
                    className="small",
                ),
            ],
            color="danger",
            className="mt-4",
            style={"borderLeft": f"5px solid {COLOR_ABANDONA}"},
        )


# --------------------------------------------------------------------------- #
# 7. Punto de entrada
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("\nDashboard disponible en http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
