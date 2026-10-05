"""
Dashboard analítico del grado tumoral en gliomas · aplicación principal.

Responsabilidades de este archivo, y solo estas:
  1. Avisar al arrancar si falta el dataset.
  2. Instanciar la app de Dash con Bootstrap.
  3. Componer la barra superior y el sistema de pestañas.
  4. Enrutar la pestaña activa hacia el `layout()` del módulo correspondiente.
  5. Atender a los botones de la guía de la portada, que cambian de pestaña.

Todo el contenido vive en tabs/*.py y toda la lógica de datos en utils/*.py.
El análisis completo, con su desarrollo estadístico, vive en el Jupyter Book de
jbook/ (publicado en GitHub Pages y enlazado desde la pestaña Documentación).

Ejecución:
    python app.py     ->    http://127.0.0.1:8080
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback,
    ctx,
    dcc,
    html,
    no_update,
)

from utils.config import AUTORES, DATA_PATH, URL_LIBRO
from utils.formato import lista_y
from utils.navegacion import BLOQUES, DESCRIPCIONES, ETIQUETAS, ORDEN
from utils.theme import BG_PAGE, INK_MUTED, STATUS_CRITICAL

# Módulos de pestañas: cada uno expone una función layout()
from tabs import (
    conclusiones,
    contexto,
    documentacion,
    exploracion,
    introduccion,
    limitaciones,
    marco_teorico,
    metodologia,
    objetivos,
    problema,
    resultados,
)

# --------------------------------------------------------------------------- #
# 1. Registro de pestañas: qué módulo pinta cada una
#    El orden, las etiquetas, los iconos y las descripciones viven en
#    utils/navegacion.py, que es lo que también lee la guía de la portada.
#    Añadir una pestaña = crear tabs/mi_pestana.py, sumarla allí y aquí.
# --------------------------------------------------------------------------- #
MODULOS = {
    "tab-introduccion": introduccion,
    "tab-contexto": contexto,
    "tab-problema": problema,
    "tab-objetivos": objetivos,
    "tab-marco": marco_teorico,
    "tab-metodologia": metodologia,
    "tab-exploracion": exploracion,
    "tab-resultados": resultados,
    "tab-limitaciones": limitaciones,
    "tab-conclusiones": conclusiones,
    "tab-documentacion": documentacion,
}

LAYOUTS = {tab_id: modulo.layout for tab_id, modulo in MODULOS.items()}
TAB_INICIAL = ORDEN[0]


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
                        # Solo el enlace al libro. Las etiquetas de Dash, Plotly
                        # y pandas parecian botones y no llevaban a ningun sitio:
                        # decoraban la cabecera y restaban peso al unico enlace
                        # que si hace algo.
                        html.Div(
                            html.A(
                                [
                                    html.I(className="bi bi-journal-text me-2"),
                                    html.Span("Abrir el", className="boton-libro-largo"),
                                    "Jupyter Book",
                                ],
                                href=URL_LIBRO,
                                target="_blank",
                                rel="noopener noreferrer",
                                className="boton-libro",
                            ),
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
        id="barra-superior",
        className="topbar",
    )


def _panel_indice() -> list[html.Div]:
    """Las pestañas del índice, agrupadas en las mismas etapas que la portada."""
    return [
        html.Div(
            [html.Div(bloque["titulo"], className="indice-bloque-titulo")]
            + [
                html.Button(
                    [
                        html.Span(ETIQUETAS[tab_id], className="indice-item-nombre"),
                        html.Span(
                            DESCRIPCIONES[tab_id], className="indice-item-texto"
                        ),
                    ],
                    id={"type": "indice-item", "index": tab_id},
                    className="indice-item",
                    n_clicks=0,
                )
                for tab_id in bloque["pestanas"]
            ],
            className=f"indice-bloque indice-bloque-{numero}",
        )
        for numero, bloque in enumerate(BLOQUES)
    ]


def _barra_indice() -> html.Div:
    """Índice desplegable, barra de progreso y botones anterior/siguiente.

    El desplegable es un <details> nativo: se abre y se cierra sin callbacks y
    responde al teclado. assets/indice.js solo añade lo que <details> no trae,
    que es cerrarse al elegir, con Escape y al hacer clic fuera.
    """
    return html.Div(
        [
            html.Details(
                [
                    html.Summary(
                        [
                            html.I(className="bi bi-list indice-icono"),
                            html.Span(
                                id="indice-posicion", className="indice-posicion"
                            ),
                            html.Span(id="indice-actual", className="indice-actual"),
                            html.I(className="bi bi-chevron-down indice-flecha"),
                        ],
                        className="indice-boton",
                    ),
                    html.Div(_panel_indice(), className="indice-panel"),
                ],
                id="indice",
                className="indice",
            ),
            html.Div(
                html.Div(id="indice-progreso", className="indice-progreso-relleno"),
                className="indice-progreso",
            ),
            html.Div(
                [
                    html.Button(
                        id="indice-anterior", className="indice-vecino", n_clicks=0
                    ),
                    html.Button(
                        id="indice-siguiente", className="indice-vecino", n_clicks=0
                    ),
                ],
                className="indice-vecinos",
            ),
        ],
        className="indice-barra",
    )


def navegacion() -> html.Div:
    """Navegación: las once pestañas en la portada, el índice en el resto.

    Las dos piezas están siempre en el DOM y se turnan con una clase que pone
    un callback de cliente. En la introducción se ven las once pestañas, que es
    donde se presenta el tablero y conviene que se vea de un vistazo todo lo que
    hay; en cuanto entras en una sección se cambian por el índice, que ocupa lo
    mismo pero además dice dónde estás, cuánto llevas y deja saltar a cualquier
    otra sin volver arriba. El panel del índice repite la guía de la portada a
    propósito: nunca se ven a la vez.

    `dbc.Tabs` sigue siendo quien guarda la pestaña activa. El índice solo le
    escribe, así que el enrutado y la guía de la portada no se enteran de nada.
    """
    return html.Div(
        dbc.Container(
            [
                html.Div(
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                label=ETIQUETAS[tab_id],
                                tab_id=tab_id,
                                # label_* aplica la clase al <a>; tab_* la aplicaria
                                # al <li> de fuera, y entonces el subrayado se
                                # dibuja dos veces.
                                label_class_name="main-tab",
                                active_label_class_name="main-tab-active",
                            )
                            for tab_id in ORDEN
                        ],
                        id="tabs-principal",
                        active_tab=TAB_INICIAL,
                        className="main-tabs",
                    ),
                    className="nav-pestanas",
                ),
                html.Div(_barra_indice(), className="nav-indice"),
            ],
            fluid=True,
        ),
        id="barra-navegacion",
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
                html.Span(
                    "Autores: " + lista_y(autor["nombre"] for autor in AUTORES)
                ),
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
        # Salida muda del callback de scroll: un callback de Dash siempre tiene
        # que escribir en algun sitio, y aqui lo que importa es el efecto.
        dcc.Store(id="ancla-scroll"),
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
# 7. Navegación desde la guía de la introducción
# --------------------------------------------------------------------------- #
@callback(
    Output("tabs-principal", "active_tab"),
    Input({"type": "ir-a-pestana", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def ir_a_pestana(clics: list[int | None]):
    """Lleva a la pestaña que anuncia la tarjeta pulsada en la portada.

    Un solo callback atiende a todos los botones gracias al pattern matching:
    el identificador de cada uno lleva dentro su `tab_id` de destino, que es el
    mismo de utils/navegacion.py. Añadir una entrada a la guía no obliga a
    tocar esto.

    El guardia del principio es necesario: Dash dispara el callback también
    cuando los botones se montan (al abrir la introducción), no solo al
    pulsarlos, y sin él la pestaña saltaría sola.
    """
    if not ctx.triggered_id or not any(clics or []):
        return no_update
    return ctx.triggered_id["index"]


# Cambiar de pestaña desde el pie de la portada dejaba al usuario a media
# página, mirando el centro de la pestaña nueva. Esto lo devuelve arriba.
app.clientside_callback(
    """
    function (pestana) {
        var suave = !(window.matchMedia
            && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
        window.scrollTo({ top: 0, behavior: suave ? 'smooth' : 'auto' });
        return pestana;
    }
    """,
    Output("ancla-scroll", "data"),
    Input("tabs-principal", "active_tab"),
)


# La barra superior completa solo tiene sentido en la portada. En el resto de
# pestañas se queda en una franja fina con el título y el enlace al libro, para
# que el contenido empiece más arriba. Se hace en el cliente: es cambiar una
# clase y no merece un viaje al servidor.
app.clientside_callback(
    """
    function (pestana) {
        return pestana === '{TAB_INICIAL}' ? 'topbar' : 'topbar topbar-fina';
    }
    """.replace("{TAB_INICIAL}", TAB_INICIAL),
    Output("barra-superior", "className"),
    Input("tabs-principal", "active_tab"),
)


# La navegación también cambia de forma al salir de la portada: las once
# pestañas dejan paso al índice. Misma clase, mismo criterio y el mismo viaje
# ahorrado al servidor que en la barra de arriba.
app.clientside_callback(
    """
    function (pestana) {
        return pestana === '{TAB_INICIAL}'
            ? 'navbar-tabs'
            : 'navbar-tabs en-seccion';
    }
    """.replace("{TAB_INICIAL}", TAB_INICIAL),
    Output("barra-navegacion", "className"),
    Input("tabs-principal", "active_tab"),
)


# --------------------------------------------------------------------------- #
# 8. Índice desplegable
# --------------------------------------------------------------------------- #
@callback(
    Output("tabs-principal", "active_tab", allow_duplicate=True),
    Input({"type": "indice-item", "index": ALL}, "n_clicks"),
    Input("indice-anterior", "n_clicks"),
    Input("indice-siguiente", "n_clicks"),
    State("tabs-principal", "active_tab"),
    prevent_initial_call=True,
)
def navegar_con_indice(_items, _anterior, _siguiente, actual: str):
    """Cambia de pestaña desde el índice o desde anterior/siguiente.

    El guardia mira el valor que disparó el callback y no solo quién lo
    disparó: al montarse, los botones llegan con n_clicks=0 y eso no es un clic.
    """
    disparo = ctx.triggered_id
    if not disparo or not ctx.triggered[0]["value"]:
        return no_update
    posicion = ORDEN.index(actual)
    if disparo == "indice-anterior":
        return ORDEN[max(posicion - 1, 0)]
    if disparo == "indice-siguiente":
        return ORDEN[min(posicion + 1, len(ORDEN) - 1)]
    return disparo["index"]


@callback(
    Output("indice-posicion", "children"),
    Output("indice-actual", "children"),
    Output("indice-progreso", "style"),
    Output("indice-anterior", "children"),
    Output("indice-anterior", "disabled"),
    Output("indice-siguiente", "children"),
    Output("indice-siguiente", "disabled"),
    Output({"type": "indice-item", "index": ALL}, "className"),
    Input("tabs-principal", "active_tab"),
)
def actualizar_indice(actual: str):
    """Rellena el botón del índice, la barra de progreso y los vecinos."""
    posicion = ORDEN.index(actual)
    total = len(ORDEN)
    anterior = ETIQUETAS[ORDEN[posicion - 1]] if posicion > 0 else "Inicio"
    siguiente = ETIQUETAS[ORDEN[posicion + 1]] if posicion < total - 1 else "Fin"
    clases = [
        "indice-item indice-item-activo"
        if salida["id"]["index"] == actual
        else "indice-item"
        for salida in ctx.outputs_list[-1]
    ]
    return (
        f"{posicion + 1} de {total}",
        ETIQUETAS[actual],
        {"width": f"{(posicion + 1) / total * 100:.1f}%"},
        [html.I(className="bi bi-arrow-left me-1"), anterior],
        posicion == 0,
        [siguiente, html.I(className="bi bi-arrow-right ms-1")],
        posicion == total - 1,
        clases,
    )


# --------------------------------------------------------------------------- #
# 9. Punto de entrada
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
