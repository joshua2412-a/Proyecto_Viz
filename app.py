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

import json
import threading
from functools import lru_cache

import dash_bootstrap_components as dbc
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from utils import asistente_panel
from utils.components import icono
from utils.config import AUTORES, DATA_PATH, URL_LIBRO
from utils.intro import portada
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
    modelos,
    objetivos,
    prediccion,
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
    "tab-modelos": modelos,
    "tab-prediccion": prediccion,
    "tab-limitaciones": limitaciones,
    "tab-conclusiones": conclusiones,
    "tab-documentacion": documentacion,
}

# Cada pestaña se construye una sola vez por proceso y se reutiliza: los datos
# no cambian mientras la app está viva y Dash serializa los componentes sin
# modificarlos. Antes cada cambio de pestaña rehacía su layout (figuras
# incluidas) y la transición se quedaba esperando al servidor, que era lo que
# cortaba la animación de las partículas al pasar a Contexto.
LAYOUTS = {tab_id: lru_cache(maxsize=1)(modulo.layout) for tab_id, modulo in MODULOS.items()}


def _precalentar() -> None:
    """Construye en segundo plano todas las pestañas al arrancar.

    Quedan en la caché de LAYOUTS, así que la primera visita a cada una ya la
    encuentra hecha. Va en un hilo aparte para no retrasar el arranque.
    """
    for tab_id in ORDEN:
        try:
            LAYOUTS[tab_id]()
        except Exception as error:  # noqa: BLE001 - si falla, la pestaña avisará sola
            print(f"[setup] No se pudo precalcular {tab_id}: {error}")


threading.Thread(target=_precalentar, daemon=True).start()
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
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap",
        # Íconos: Material Symbols de Google Fonts (fonts.google.com/icons).
        # display=block evita que se vea el nombre en texto mientras carga.
        "https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:"
        "opsz,wght,FILL,GRAD@20..48,300..600,0..1,0&display=block",
    ],
    # Los callbacks de las pestañas apuntan a componentes que no están en el
    # layout inicial (se montan al abrir la pestaña), así que hay que permitirlo.
    suppress_callback_exceptions=True,
    title="Gliomas LGG vs GBM · Dashboard Analítico",
    update_title="Calculando...",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # objeto WSGI que gunicorn sirve en el contenedor

# Modo claro u oscuro antes del primer pintado. Dash carga los .js de /assets
# al final del <body>: si el tema lo pusiera assets/tema.js, la página se
# vería un instante en claro y luego saltaría a oscuro. Este script mínimo va
# en el <head> y solo pone el atributo; todo lo demás lo hace tema.js.
app.index_string = """<!DOCTYPE html>
<html lang="es">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <script>
            (function () {
                var guardado = null;
                try { guardado = localStorage.getItem("tema"); } catch (e) {}
                var oscuro = guardado ? guardado === "oscuro"
                    : !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
                document.documentElement.setAttribute("data-tema", oscuro ? "oscuro" : "claro");
            })();
        </script>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


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
                            # El título lleva a la portada. Es un enlace normal
                            # (recarga la página): reabrir la portada en caliente
                            # obligaría a reiniciar su WebGL, la hélice de
                            # banda.js y la entrada de Glía, y la recarga lo
                            # deja todo como en la primera visita.
                            html.H1(
                                html.A(
                                    "Grado tumoral en gliomas",
                                    href="/",
                                    className="brand-title-enlace",
                                    title="Volver a la portada",
                                ),
                                className="brand-title",
                            ),
                            html.Div(
                                "Exploración, modelado y predicción del grado "
                                "tumoral a partir de la edad, el perfil "
                                "demográfico y 20 mutaciones en 839 pacientes "
                                "de TCGA-LGG y TCGA-GBM",
                                className="brand-subtitle",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        # Solo el enlace al libro y el cambio de tema. Las
                        # etiquetas de Dash, Plotly y pandas parecian botones y
                        # no llevaban a ningun sitio: decoraban la cabecera y
                        # restaban peso a lo que si hace algo.
                        html.Div([
                            # Claro / oscuro. Los dos iconos están siempre y se
                            # cruzan con CSS; el texto accesible y el estado
                            # los pone assets/tema.js.
                            html.Button(
                                [
                                    icono("dark_mode", "tema-icono tema-icono-luna"),
                                    icono("light_mode", "tema-icono tema-icono-sol"),
                                ],
                                id="boton-tema",
                                className="boton-tema",
                                type="button",
                                title="Cambiar entre modo claro y oscuro",
                                **{"aria-label": "Cambiar entre modo claro y oscuro"},
                            ),
                            html.A(
                                [
                                    icono("menu_book", "me-2"),
                                    html.Span("Abrir el", className="boton-libro-largo"),
                                    "Jupyter Book",
                                ],
                                href=URL_LIBRO,
                                target="_blank",
                                rel="noopener noreferrer",
                                title="Abrir el Jupyter Book",
                                className="boton-libro",
                            ),
                        ], className="tech-pills"),
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
            [html.Div(
                ([html.Span(bloque["numero"], className="indice-bloque-num")]
                 if bloque["numero"] else [])
                + [bloque["titulo"]],
                className="indice-bloque-titulo",
            )]
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
                            icono("menu", "indice-icono"),
                            html.Span(
                                id="indice-posicion", className="indice-posicion"
                            ),
                            html.Span(id="indice-actual", className="indice-actual"),
                            icono("expand_more", "indice-flecha"),
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
    """Navegación: las trece pestañas en la portada, el índice en el resto.

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
        # Portada animada (assets/intro.js): tapa el tablero hasta pulsar
        # "Explorar el dashboard".
        portada(),
        barra_superior(),
        navegacion(),
        dbc.Container(
            dcc.Loading(
                html.Div(id="contenido-pestana"),
                # Solo el cambio de pestaña enciende el indicador. Sin esto,
                # cualquier callback de dentro (marcar un gen en Predicción,
                # mover un umbral) tapaba la pestaña entera mientras corría y
                # la página parpadeaba a cada clic.
                target_components={"contenido-pestana": "children"},
                # Con las pestañas ya construidas (LAYOUTS en caché) el cambio
                # tarda poco; el indicador solo sale si de verdad se demora,
                # para no tapar la pestaña mientras viajan las partículas.
                delay_show=700,
                type="dot",
                color=INK_MUTED,
                parent_className="loading-wrapper",
            ),
            fluid=True,
            className="page-container",
        ),
        pie_pagina(),
        # Asistente virtual flotante, presente en todas las pestañas
        # (utils/asistente_panel.py). Sus almacenes viven aquí y no en una
        # pestaña para que la conversación no se pierda al navegar.
        asistente_panel.layout(),
        # Salida muda del callback de scroll: un callback de Dash siempre tiene
        # que escribir en algun sitio, y aqui lo que importa es el efecto.
        dcc.Store(id="ancla-scroll"),
        # Gráfico invisible: obliga a cargar y arrancar Plotly mientras se ve
        # la portada. Introducción no tiene gráficos, así que sin esto la
        # primera carga de Plotly (unos 750 ms con el navegador bloqueado)
        # caía justo en el viaje de las partículas hacia Contexto clínico.
        dcc.Graph(
            id="precarga-plotly",
            figure={"data": [{"type": "scatter", "x": [0], "y": [0]}],
                    "layout": {"width": 10, "height": 10, "margin": {"l": 0, "r": 0, "t": 0, "b": 0}}},
            config={"staticPlot": True, "displayModeBar": False},
            style={"position": "fixed", "left": "-40px", "top": "0", "width": "10px",
                   "height": "10px", "opacity": 0, "pointerEvents": "none"},
        ),
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
# Va en el cliente: cambiar de pestaña es solo escribir `active_tab`, y
# hacerlo en el servidor añadía un viaje de ida y vuelta (en Cloud Run, unos
# cientos de milisegundos) antes de que la pestaña nueva empezara a pedirse.
# Un solo callback atiende a todos los botones gracias al pattern matching:
# el identificador de cada uno lleva dentro su `tab_id` de destino, que es el
# mismo de utils/navegacion.py.
#
# El guardia del principio es necesario: Dash dispara el callback también
# cuando los botones se montan (al abrir la introducción), no solo al
# pulsarlos, y sin él la pestaña saltaría sola.
app.clientside_callback(
    """
    function (clics) {
        var c = window.dash_clientside.callback_context;
        var sinCambio = window.dash_clientside.no_update;
        if (!c.triggered.length || !c.triggered[0].value || !c.triggered_id) return sinCambio;
        return c.triggered_id.index;
    }
    """,
    Output("tabs-principal", "active_tab"),
    Input({"type": "ir-a-pestana", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)


# Cambiar de pestaña desde el pie de la portada dejaba al usuario a media
# página, mirando el centro de la pestaña nueva. Esto lo devuelve arriba.
# Suave solo si estás cerca de arriba: desde el pie de una pestaña larga, el
# desplazamiento suave recorría miles de píxeles justo mientras se montaba la
# pestaña nueva (el momento en que el navegador está más ocupado) y se veía
# a saltos; además, ese contenido se va a sustituir, no hay nada que seguir.
app.clientside_callback(
    """
    function (pestana) {
        var suave = !(window.matchMedia
            && window.matchMedia('(prefers-reduced-motion: reduce)').matches)
            && window.scrollY < window.innerHeight * 0.6;
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
# Los dos callbacks del índice corren en el navegador. En el servidor, pulsar
# «siguiente» hacía dos viajes seguidos (calcular la pestaña nueva y, después,
# pedir su contenido) y el «3 de 13», el progreso y los vecinos llegaban con
# el segundo: medio segundo largo en el que el clic no parecía hacer nada.
# Ahora la pestaña cambia en el mismo fotograma del clic y solo el contenido
# sigue viniendo del servidor. El orden y los nombres salen de
# utils/navegacion.py, igual que antes.
_NAV_JS = json.dumps({"orden": ORDEN, "etiquetas": ETIQUETAS}, ensure_ascii=False)

app.clientside_callback(
    """
    function (_items, _anterior, _siguiente, actual) {
        var c = window.dash_clientside.callback_context;
        var sinCambio = window.dash_clientside.no_update;
        // Al montarse, los botones llegan con n_clicks=0: eso no es un clic
        if (!c.triggered.length || !c.triggered[0].value || !c.triggered_id) return sinCambio;
        var orden = NAV.orden, pos = orden.indexOf(actual), id = c.triggered_id;
        if (id === "indice-anterior") return orden[Math.max(pos - 1, 0)];
        if (id === "indice-siguiente") return orden[Math.min(pos + 1, orden.length - 1)];
        return id.index;
    }
    """.replace("NAV", _NAV_JS),
    Output("tabs-principal", "active_tab", allow_duplicate=True),
    Input({"type": "indice-item", "index": ALL}, "n_clicks"),
    Input("indice-anterior", "n_clicks"),
    Input("indice-siguiente", "n_clicks"),
    State("tabs-principal", "active_tab"),
    prevent_initial_call=True,
)

# Rellena el botón del índice, la barra de progreso y los vecinos. Los
# iconos van como el JSON de `icono()` (un Span de Material Symbols).
app.clientside_callback(
    """
    function (actual) {
        var orden = NAV.orden, nombres = NAV.etiquetas;
        var pos = Math.max(0, orden.indexOf(actual)), total = orden.length;
        function icono(nombre, clase) {
            return { type: "Span", namespace: "dash_html_components",
                     props: { children: nombre, className: "material-symbols-rounded " + clase,
                              "aria-hidden": "true" } };
        }
        var anterior = pos > 0 ? nombres[orden[pos - 1]] : "Inicio";
        var siguiente = pos < total - 1 ? nombres[orden[pos + 1]] : "Fin";
        var salidas = window.dash_clientside.callback_context.outputs_list;
        var clases = salidas[salidas.length - 1].map(function (s) {
            return s.id.index === actual ? "indice-item indice-item-activo" : "indice-item";
        });
        return [
            (pos + 1) + " de " + total,
            nombres[actual],
            { width: ((pos + 1) / total * 100).toFixed(1) + "%" },
            [icono("arrow_back", "me-1"), anterior],
            pos === 0,
            [siguiente, icono("arrow_forward", "ms-1")],
            pos === total - 1,
            clases
        ];
    }
    """.replace("NAV", _NAV_JS),
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
