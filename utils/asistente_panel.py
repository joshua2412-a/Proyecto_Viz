"""
Asistente virtual «Glía» · interfaz y callbacks.

Glía es un astrocito de partículas (assets/glia.js): un botón flotante en la
esquina inferior derecha abre un panel en azul noche que está en todas las
pestañas, y cada gráfico lleva un chip «Pregúntale a Glía». La lógica (qué
responder y cómo) vive en utils/asistente.py; aquí solo se compone y se
conecta.

Reparto entre cliente y servidor:
- Abrir, cerrar, Escape, foco, desplazamiento y la pista de cada pestaña van
  en assets/asistente.js; el dibujo, los estados y el tinte de Glía, en
  assets/glia.js. Son cambios de clase y lienzos: no merecen un viaje al
  servidor.
- Responder va en el servidor, porque necesita los modelos (lectura de la
  predicción) y, si hay clave, la API de la IA, que nunca debe tocar el
  navegador.
- Ir a una pestaña desde una respuesta es un callback de cliente que escribe
  `active_tab`, igual que la guía de la portada.
"""

from __future__ import annotations

from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from utils import asistente
from utils.components import GRAFICOS_GLIA, icono
from utils.navegacion import ETIQUETAS

ID = "asistente"
ID_ABRIR = "asistente-abrir"
ID_CERRAR = "asistente-cerrar"
ID_PANEL = "asistente-panel"
ID_MENSAJES = "asistente-mensajes"
ID_SUGERENCIAS = "asistente-sugerencias"
ID_ENTRADA = "asistente-entrada"
ID_ENVIAR = "asistente-enviar"
ID_ESCRIBIENDO = "asistente-escribiendo"
ID_HISTORIAL = "asistente-historial"
ID_MODO = "asistente-modo"
ID_PISTA = "glia-pista"
# Lo escribe la pestaña Predicción (tabs/prediccion.py) cada vez que cambia el
# perfil o el modelo. Vive aquí, fuera de la pestaña, para que el asistente lo
# lea desde cualquier otra.
ID_PREDICCION = "asistente-prediccion"

# Mensajes que se guardan como máximo: el historial viaja entero en cada
# pregunta, y una conversación larga lo haría crecer sin necesidad.
MAX_MENSAJES = 40

# La presentación de Glía va en la cabecera del panel; el primer mensaje solo
# dice qué puede hacer.
BIENVENIDA = {
    "rol": "asistente",
    "inicial": True,
    "fuente": "guiada",
    "texto": (
        "Puedo **explicarte los resultados**, **definir conceptos**, **interpretar tu "
        "predicción** y **llevarte a cualquier pestaña**. Elige una pregunta, escríbeme "
        "o pulsa «Pregúntale a Glía» junto a cualquier gráfico."
    ),
}

ETIQUETA_FUENTE = {
    "ia": ("auto_awesome", "Respondido con IA"),
    "prediccion": ("calculate", "Calculado con los modelos"),
}


# --------------------------------------------------------------------------- #
# Piezas
# --------------------------------------------------------------------------- #
def _avatar(tono: str | None = None, fuerza: float = 0.0, tamano: int = 28) -> html.Canvas:
    """Glía en pequeño, teñida del grado del que habla (la pinta glia.js)."""
    return html.Canvas(
        className="glia-avatar",
        width=tamano,
        height=tamano,
        **{"data-tono": tono or "", "data-fuerza": str(fuerza or 0), "aria-hidden": "true"},
    )


def _globo(mensaje: dict, numero: int) -> html.Div:
    """Un mensaje. Los de Glía admiten Markdown y fórmulas en LaTeX."""
    if mensaje.get("rol") == "usuario":
        return html.Div(html.Div(mensaje["texto"], className="asistente-globo"),
                        className="asistente-mensaje asistente-mensaje-usuario")

    hijos = []
    if mensaje.get("aviso"):
        hijos.append(html.Div([icono("info", "me-1"), mensaje["aviso"]], className="asistente-aviso-ia"))
    hijos.append(dcc.Markdown(mensaje["texto"], mathjax=True, link_target="_blank",
                              className="asistente-markdown"))
    pie = []
    destino = mensaje.get("ir")
    if destino in ETIQUETAS:
        # El número de mensaje va en el id: dos respuestas pueden apuntar a la
        # misma pestaña y Dash no admite ids repetidos.
        pie.append(html.Button(
            [f"Ir a {ETIQUETAS[destino]}", icono("arrow_forward", "ms-1")],
            id={"type": "asistente-ir", "index": f"{numero}|{destino}"},
            className="asistente-ir", n_clicks=0, type="button",
        ))
    fuente = ETIQUETA_FUENTE.get(mensaje.get("fuente"))
    if fuente:
        texto = (f"Respondido con {mensaje['proveedor']}" if mensaje.get("proveedor")
                 else fuente[1])
        pie.append(html.Span([icono(fuente[0], "me-1"), texto], className="asistente-fuente"))
    if pie:
        hijos.append(html.Div(pie, className="asistente-pie-mensaje"))
    return html.Div(
        [html.Span(_avatar(mensaje.get("tono"), mensaje.get("fuerza", 0)), className="asistente-avatar"),
         html.Div(hijos, className="asistente-globo")],
        className="asistente-mensaje asistente-mensaje-bot",
    )


def _globos(historial: list[dict]) -> list:
    return [_globo(m, i) for i, m in enumerate(historial or [])]


def _modo() -> str:
    proveedor = asistente.proveedor_ia()
    nombre = asistente.NOMBRE_PROVEEDOR[proveedor] if proveedor else None
    return f"Tu guía por el tablero · con {nombre}" if nombre else "Tu guía por el tablero"


def layout() -> html.Div:
    """Botón flotante, pista, panel y los dos almacenes de Glía."""
    return html.Div(
        [
            # Pista de cada pestaña (A4): la escribe `mostrar_sugerencias` y la
            # enseña asistente.js una vez por pestaña, sin interrumpir.
            html.Div(
                [
                    html.Button(id=ID_PISTA, className="glia-pista-texto", type="button"),
                    html.Button(icono("close"), className="glia-pista-cerrar", type="button",
                                title="Descartar", **{"aria-label": "Descartar la pista"}),
                ],
                className="glia-pista",
                **{"aria-live": "polite"},
            ),
            html.Button(
                [html.Canvas(id="glia-fab", className="glia-fab-lienzo", width=42, height=42,
                             **{"aria-hidden": "true"}),
                 html.Span("Pregúntale a Glía", className="asistente-fab-texto")],
                id=ID_ABRIR,
                className="asistente-fab",
                type="button",
                title="Pregúntale a Glía, la asistente del tablero",
                **{"aria-label": "Abrir a Glía, la asistente del tablero", "aria-expanded": "false",
                   "aria-controls": ID_PANEL},
            ),
            html.Section(
                [
                    # Partículas que derivan al fondo del panel (glia.js)
                    html.Canvas(id="asistente-fondo", className="asistente-fondo",
                                **{"aria-hidden": "true"}),
                    html.Button(
                        icono("close"), id=ID_CERRAR, className="asistente-cerrar",
                        type="button", title="Cerrar",
                        **{"aria-label": "Cerrar a Glía"},
                    ),
                    # Glía preside el panel; al empezar a conversar se encoge y
                    # sube a la cabecera (clase `conversando`, la pone glia.js).
                    html.Div(
                        [
                            html.Canvas(id="glia-heroe", className="glia-heroe-lienzo", width=84, height=84,
                                        **{"aria-hidden": "true"}),
                            html.Div(
                                [
                                    html.Div("Glía", className="asistente-titulo"),
                                    html.Div("Los gliomas nacen de células como yo. "
                                             "Te ayudo a leer este tablero.",
                                             className="asistente-lema"),
                                    html.Div(_modo(), id=ID_MODO, className="asistente-modo"),
                                ],
                                className="asistente-cabecera-texto",
                            ),
                        ],
                        className="asistente-heroe",
                    ),
                    html.Div(
                        _globos([BIENVENIDA]),
                        id=ID_MENSAJES,
                        className="asistente-mensajes",
                        role="log",
                        **{"aria-live": "polite"},
                    ),
                    html.Div(
                        [html.Span(), html.Span(), html.Span(),
                         html.Em("Glía está pensando…", className="asistente-escribiendo-texto")],
                        id=ID_ESCRIBIENDO,
                        className="asistente-escribiendo",
                        **{"aria-hidden": "true"},
                    ),
                    html.Div(id=ID_SUGERENCIAS, className="asistente-sugerencias"),
                    html.Div(
                        [
                            dcc.Input(
                                id=ID_ENTRADA, type="text", value="", n_submit=0,
                                placeholder="Pregúntale a Glía…",
                                maxLength=asistente.MAX_PREGUNTA, autoComplete="off",
                                # dcc.Input no admite aria-label: lo pone
                                # assets/asistente.js al abrir el panel.
                                className="asistente-entrada",
                            ),
                            html.Button(
                                icono("send"), id=ID_ENVIAR, n_clicks=0,
                                className="asistente-enviar", type="button", title="Enviar",
                                **{"aria-label": "Enviar la pregunta"},
                            ),
                        ],
                        className="asistente-formulario",
                    ),
                    html.Div("Herramienta docente: no sustituye un diagnóstico médico.",
                             className="asistente-nota"),
                ],
                id=ID_PANEL,
                className="asistente-panel",
                role="dialog",
                **{"aria-label": "Glía, la asistente del tablero", "aria-hidden": "true"},
            ),
            dcc.Store(id=ID_HISTORIAL, data=[BIENVENIDA]),
            dcc.Store(id=ID_PREDICCION),
        ],
        id=ID,
        className="asistente",
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@callback(
    Output(ID_SUGERENCIAS, "children"),
    Output(ID_PISTA, "children"),
    Input("tabs-principal", "active_tab"),
)
def mostrar_sugerencias(pestana):
    """Tres preguntas pensadas para la pestaña, y la pista que Glía ofrece al entrar."""
    sugerencias = asistente.sugerencias(pestana)
    botones = [
        html.Button(texto, id={"type": "asistente-sugerencia", "index": clave},
                    className="asistente-sugerencia", n_clicks=0, type="button")
        for clave, texto in sugerencias
    ]
    # La pista es la primera sugerencia de la pestaña, dicha por Glía. En la
    # introducción no hay: allí el usuario acaba de llegar de la portada.
    pista = "" if pestana in (None, "tab-introduccion") else f"Pregúntame: «{sugerencias[0][1]}»"
    return botones, pista


@callback(
    Output(ID_HISTORIAL, "data"),
    Output(ID_ENTRADA, "value"),
    Output("tabs-principal", "active_tab", allow_duplicate=True),
    Output(ID_MODO, "children"),
    Input(ID_ENVIAR, "n_clicks"),
    Input(ID_ENTRADA, "n_submit"),
    Input({"type": "asistente-sugerencia", "index": ALL}, "n_clicks"),
    Input({"type": "glia-grafico", "index": ALL}, "n_clicks"),
    State(ID_ENTRADA, "value"),
    State(ID_HISTORIAL, "data"),
    State(ID_PREDICCION, "data"),
    State("tabs-principal", "active_tab"),
    prevent_initial_call=True,
    # Mientras responde: Glía «piensa» (glia.js lo lee de esta clase) y el
    # botón de enviar se desactiva, para que un doble clic no mande la
    # pregunta dos veces.
    running=[
        (Output(ID_ESCRIBIENDO, "className"),
         "asistente-escribiendo asistente-escribiendo-activo", "asistente-escribiendo"),
        (Output(ID_ENVIAR, "disabled"), True, False),
    ],
)
def conversar(_enviar, _intro, _sugerencias, _graficos, texto, historial, estado_pred, pestana):
    """Añade la pregunta y su respuesta al historial."""
    disparo = ctx.triggered_id
    # Sugerencias y chips se vuelven a montar al cambiar de pestaña con
    # n_clicks=0, y Dash lo cuenta como disparo: eso no es un clic.
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update, no_update, no_update, no_update
    historial = list(historial or [])

    if isinstance(disparo, dict) and disparo.get("type") == "asistente-sugerencia":
        clave = disparo["index"]
        pregunta = asistente.texto_sugerencia(clave)
        respuesta = asistente.responder_sugerencia(clave, estado_pred)
    elif isinstance(disparo, dict) and disparo.get("type") == "glia-grafico":
        grafico = GRAFICOS_GLIA.get(disparo["index"])
        if not grafico:
            return no_update, no_update, no_update, no_update
        pregunta = f"Explícame el gráfico «{grafico['titulo']}»"
        respuesta = asistente.explicar_grafico(grafico["titulo"], grafico["nota"], historial,
                                               pestana, estado_pred)
    else:
        pregunta = (texto or "").strip()
        if not pregunta:
            return no_update, no_update, no_update, no_update
        respuesta = asistente.responder(pregunta, historial, pestana, estado_pred)

    historial.append({"rol": "usuario", "texto": pregunta[: asistente.MAX_PREGUNTA]})
    historial.append({"rol": "asistente", **{k: v for k, v in respuesta.items() if k != "navegar"}})
    historial = historial[-MAX_MENSAJES:]

    ir = respuesta["ir"] if respuesta.get("navegar") and respuesta.get("ir") != pestana else no_update
    return historial, "", ir, _modo()


@callback(
    Output(ID_MENSAJES, "children"),
    Input(ID_HISTORIAL, "data"),
    prevent_initial_call=True,
)
def pintar_mensajes(historial):
    return _globos(historial)


# El botón «Ir a …» de una respuesta cambia de pestaña en el navegador, sin
# pasar por el servidor (mismo criterio que la guía de la portada).
clientside_callback(
    """
    function (clics) {
        var c = window.dash_clientside.callback_context;
        var sinCambio = window.dash_clientside.no_update;
        if (!c.triggered.length || !c.triggered[0].value || !c.triggered_id) return sinCambio;
        return c.triggered_id.index.split("|")[1];
    }
    """,
    Output("tabs-principal", "active_tab", allow_duplicate=True),
    Input({"type": "asistente-ir", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
