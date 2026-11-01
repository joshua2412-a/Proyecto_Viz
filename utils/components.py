"""
Componentes reutilizables de interfaz.

Estas funciones encapsulan los patrones visuales que se repiten en las
pestañas (cabeceras de sección, tarjetas, KPIs, listas con viñetas...).
Así cada pestaña se limita a describir su contenido, no su maquetación.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import dash_bootstrap_components as dbc
from dash import dcc, html

from utils.theme import (
    ACCENT_NEUTRAL,
    BG_SOFT,
    COLOR_GBM,
    COLOR_LGG,
    INK_MUTED,
)


# --------------------------------------------------------------------------- #
# Notación matemática
# --------------------------------------------------------------------------- #
def tex(texto: str, className: str = "") -> dcc.Markdown:
    """Texto con fórmulas en LaTeX entre $...$, compuestas con MathJax.

    Antes los símbolos se escribían a mano (|r|, χ², R²) y se veían como texto
    corriente, no como notación. Ahora cualquier texto del tablero puede llevar
    LaTeX en línea: `tex("con $|r| \\geq 0{,}10$")`. La coma decimal va entre
    llaves, `0{,}10`, porque en modo matemático una coma suelta deja un espacio
    detrás (ver `formato.tnum`).
    """
    return dcc.Markdown(texto, mathjax=True, className=f"tex {className}".strip())


def con_formulas(contenido):
    """Pasa por `tex` los textos que llevan $...$ y deja el resto como está.

    Lo usan todas las piezas de este módulo que reciben texto (párrafos,
    celdas, listas, notas), así que basta con escribir la fórmula en la cadena:
    la pieza se encarga de componerla.
    """
    if isinstance(contenido, str) and "$" in contenido:
        return tex(contenido)
    if isinstance(contenido, list):
        return [con_formulas(parte) for parte in contenido]
    return contenido


# --------------------------------------------------------------------------- #
# Cabeceras y textos
# --------------------------------------------------------------------------- #
def icono(nombre: str, className: str = "") -> html.Span:
    """Ícono de Material Symbols (Google Fonts), por su nombre: "bar_chart".

    La fuente se carga en app.py. Se escribe el nombre como texto y la
    ligadura de la fuente lo convierte en el dibujo; aria-hidden porque el
    ícono siempre acompaña a un texto que ya dice lo mismo.
    """
    clases = f"material-symbols-rounded {className}".strip()
    return html.Span(nombre, className=clases, **{"aria-hidden": "true"})


icono_ = icono  # alias: page_header tiene un parámetro llamado icono


def page_header(titulo: str, subtitulo: str, icono: str = "") -> html.Div:
    """Cabecera de pestaña: título grande + línea de contexto.

    `icono` es el nombre de un ícono de Material Symbols (p. ej. "bar_chart"),
    cuya fuente carga app.py. Se usa un set monocromo en vez de emoji para que el
    color de la página siga reservado a los datos.
    """
    encabezado = []
    if icono:
        encabezado.append(icono_(icono, "page-icon"))
    # "Contexto clínico: dos tumores..." se compone en dos pesos, como los
    # titulares: la parte de antes de los dos puntos en fino y el resto en
    # negrita. El texto es el mismo; solo cambia cómo se compone.
    if ": " in titulo:
        antes, despues = titulo.split(": ", 1)
        encabezado.append(html.Span([html.Span(antes + ":", className="page-title-ligero"),
                                     html.Br(), despues]))
    else:
        encabezado.append(html.Span(titulo))
    return html.Div(
        [
            html.H2(encabezado, className="page-title"),
            html.P(subtitulo, className="page-subtitle"),
            html.Hr(className="page-rule"),
        ],
        className="page-header",
    )


def titular(eyebrow: str, linea: str, destacado: str, derecha=None, ayuda: str = "",
            texto: str = "") -> html.Div:
    """Titular de pestaña en dos pesos, con un control opcional a la derecha.

    Es la cabecera de la mezcla A + B: una línea fina y otra en negrita
    ("Describe un paciente / y mira qué predicen"). `derecha` es el control
    principal de la pestaña (un segmentado, unas subpestañas) y `ayuda`, la
    etiqueta pequeña que va encima de él.
    """
    izquierda = [
        html.Div(eyebrow, className="titular-eyebrow"),
        html.H2([linea, html.Br(), html.Strong(destacado)], className="titular"),
    ]
    if texto:
        izquierda.append(html.P(texto, className="titular-texto"))
    hijos = [html.Div(izquierda, className="titular-izquierda")]
    if derecha is not None:
        selector = [html.Div(ayuda, className="titular-ayuda")] if ayuda else []
        hijos.append(html.Div(selector + [derecha], className="titular-selector"))
    return html.Div(hijos, className="titular-fila")


def section_title(texto: str, nivel: int = 4) -> html.Div:
    """Título de sección dentro de una pestaña."""
    heading = {3: html.H3, 4: html.H4, 5: html.H5}.get(nivel, html.H4)
    return heading(texto, className="section-title")


def paragraph(texto) -> html.P:
    """Párrafo con el estilo de lectura del dashboard."""
    if isinstance(texto, str) and "$" in texto:
        return tex(texto, "body-text")
    return html.P(con_formulas(texto), className="body-text")


def bullet_list(items: Iterable, color: str = ACCENT_NEUTRAL) -> html.Ul:
    """Lista con viñetas. El punto es neutro salvo que se pida otro color."""
    return html.Ul(
        [html.Li(con_formulas(item), className="body-text") for item in items],
        className="bullet-list",
        style={"--bullet-color": color},
    )


def enlace_externo(texto: str, url: str, icono: str = "arrow_outward") -> html.A:
    """Enlace que abre en una pestaña nueva (libro, dataset, repositorio)."""
    return html.A(
        [texto, " ", icono_(icono, "icono-enlace")] if icono else texto,
        href=url,
        target="_blank",
        rel="noopener noreferrer",
        className="link-externo",
    )


# --------------------------------------------------------------------------- #
# Tarjetas
# --------------------------------------------------------------------------- #
def card(
    children,
    titulo: str | None = None,
    subtitulo: str | None = None,
    color: str | None = None,
    className: str = "",
    tono: str = "",
) -> dbc.Card:
    """Tarjeta genérica con borde superior de color y cuerpo libre.

    `tono` tiñe el fondo: "acento" para las tarjetas que interpretan o resumen,
    "panel" para las que solo agrupan. Es cromo, no dato: el azul de LGG y el
    coral de GBM siguen siendo los únicos colores que significan algo.
    """
    cuerpo = []
    if titulo:
        cuerpo.append(html.H5(titulo, className="card-title-custom"))
    if subtitulo:
        cuerpo.append(html.Div(con_formulas(subtitulo), className="card-subtitle-custom"))
    cuerpo.append(html.Div(children))

    # Sin color por defecto: el color del tablero esta reservado a los datos.
    estilo = {"borderTop": f"2px solid {color}"} if color else {}
    clases = " ".join(
        parte for parte in ("soft-card", f"tono-{tono}" if tono else "", className)
        if parte
    )
    return dbc.Card(dbc.CardBody(cuerpo), className=clases, style=estilo)


def kpi_card(valor: str, etiqueta: str, detalle: str = "", color: str = ACCENT_NEUTRAL) -> dbc.Card:
    """Tarjeta de indicador: cifra protagonista + etiqueta + detalle opcional.

    La franja de color es un div absoluto, no un `border-left`. El motivo es que
    asi `color` admite cualquier valor de fondo CSS y no solo un color plano: un
    indicador que compara los dos grados puede llevar un degradado partido (azul
    de LGG arriba, coral de GBM abajo), que es justo lo que hace la brecha de
    IDH1 en la introduccion. Con un borde eso no se puede expresar.

    El color sigue siendo semantico: neutro cuando la cifra no habla de un grado
    concreto, y el color del grado cuando si lo hace.
    """
    cuerpo = [
        html.Div(etiqueta, className="kpi-label"),
        html.Div(valor, className="kpi-value"),
    ]
    if detalle:
        cuerpo.append(html.Div(con_formulas(detalle), className="kpi-detail"))
    return dbc.Card(
        [
            html.Div(className="kpi-accent", style={"background": color}),
            dbc.CardBody(cuerpo),
        ],
        className="soft-card kpi-card",
    )


def kpi_row(kpis: Sequence[dict], md: int = 3) -> dbc.Row:
    """Fila de tarjetas KPI a partir de una lista de diccionarios."""
    return dbc.Row(
        [
            dbc.Col(kpi_card(**kpi), md=md, sm=6, xs=12, className="mb-3")
            for kpi in kpis
        ],
        className="g-3",
    )


def callout(texto, titulo: str | None = None, color: str = ACCENT_NEUTRAL) -> html.Div:
    """Bloque destacado para ideas clave o advertencias."""
    hijos = []
    if titulo:
        hijos.append(html.Div(titulo, className="callout-title"))
    hijos.append(html.Div(con_formulas(texto), className="callout-text"))
    return html.Div(
        hijos,
        className="callout",
        style={"borderLeft": f"3px solid {color}", "background": BG_SOFT},
    )


# --------------------------------------------------------------------------- #
# Piezas de la exploración interactiva
# --------------------------------------------------------------------------- #
def banda_oscura(titulo: str, texto: str, eyebrow: str = "") -> html.Div:
    """Banda de tinta con trama diagonal, para abrir una pestaña o un módulo.

    Es el único sitio donde el fondo lleva materia: separa lo interactivo del
    resto sin recurrir a más color, que está reservado a los datos.
    """
    hijos = []
    if eyebrow:
        hijos.append(html.Div(eyebrow, className="banda-eyebrow"))
    hijos.append(html.H3(titulo, className="banda-titulo"))
    hijos.append(html.P(texto, className="banda-texto"))
    return html.Div(hijos, className="banda-oscura")


def stat_list(filas: Sequence[tuple[str, str]]) -> html.Div:
    """Lista de estadísticos: etiqueta a la izquierda, cifra a la derecha."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(con_formulas(etiqueta), className="stat-etiqueta"),
                    html.Span(con_formulas(valor), className="stat-valor"),
                ],
                className="stat-fila",
            )
            for etiqueta, valor in filas
        ],
        className="stat-list",
    )


def badge_numero(texto: str, variante: str = "") -> html.Span:
    """Distintivo numerado. `variante="gbm"` lo pinta en coral."""
    clase = "badge-numero badge-gbm" if variante == "gbm" else "badge-numero"
    return html.Span(texto, className=clase)


def chip_tipo(texto: str) -> html.Span:
    """Etiqueta redonda para el tipo de variable."""
    return html.Span(texto, className="chip-tipo")


def lectura_guiada(lineas: Iterable, titulo: str = "Lectura guiada") -> dbc.Card:
    """Tarjeta con el mensaje del gráfico, para poner justo a su lado.

    Es la pieza que convierte una figura en un hallazgo: sin ella, el visitante
    ve la forma pero tiene que deducir solo qué significa.
    """
    return card(
        html.Ul([html.Li(con_formulas(linea)) for linea in lineas], className="lectura-lista"),
        titulo=titulo,
        className="lectura-card",
    )


def desplegable(titulo: str, contenido, abierto: bool = False) -> dbc.Accordion:
    """Bloque plegable, al estilo de los desplegables de genes del Jupyter Book."""
    return dbc.Accordion(
        [dbc.AccordionItem(contenido, title=titulo)],
        start_collapsed=not abierto,
        flush=False,
        className="desplegable",
    )


# --------------------------------------------------------------------------- #
# Gráficos y tablas
# --------------------------------------------------------------------------- #
def altura_grafico(figura) -> dict | None:
    """Estilo con la altura en px que fija la figura (None si no la fija)."""
    try:
        alto = figura.layout.height
    except AttributeError:
        alto = None
    return {"height": f"{int(alto)}px"} if alto else None


# Gráficos que tienen chip «Pregúntale a Glía» (A4): número -> título y nota.
# El asistente (utils/asistente_panel.py) los busca aquí al pulsar el chip,
# así el botón solo lleva un número y no el texto entero en su id. Las
# pestañas se construyen una vez por proceso (LAYOUTS en app.py), así que
# cada gráfico conserva su número mientras la app vive.
GRAFICOS_GLIA: dict[str, dict] = {}


def chip_glia(titulo: str, nota: str = "") -> html.Button:
    """Chip «Pregúntale a Glía» que abre el asistente preguntando por un gráfico."""
    clave = str(len(GRAFICOS_GLIA) + 1)
    GRAFICOS_GLIA[clave] = {"titulo": titulo, "nota": nota if isinstance(nota, str) else ""}
    return html.Button(
        # La célula la pinta assets/glia.js en el lienzo (avatar quieto)
        [html.Canvas(className="glia-avatar", width=22, height=22, **{"aria-hidden": "true"}),
         html.Span("Pregúntale a Glía", className="glia-chip-texto")],
        id={"type": "glia-grafico", "index": clave},
        className="glia-chip",
        n_clicks=0,
        type="button",
        title=f"Pregúntale a Glía por «{titulo}»",
    )


def graph_card(figura, titulo: str, nota: str = "", graph_id: str | None = None,
               className: str = "") -> dbc.Card:
    """Tarjeta que envuelve un gráfico de Plotly con título y nota al pie."""
    configuracion = {"displaylogo": False, "responsive": True}
    # Con `responsive`, dcc.Graph ignora layout.height y estira la gráfica al
    # 100 % de su contenedor. Si la tarjeta es la única de su fila, ese 100 % no
    # se refiere a nada y la gráfica se dibujaba encima del contenido siguiente:
    # el contenedor recibe aquí la altura de la figura, de forma explícita.
    estilo = altura_grafico(figura)
    grafico = (
        dcc.Graph(figure=figura, id=graph_id, config=configuracion, className="graph",
                  style=estilo)
        if graph_id
        else dcc.Graph(figure=figura, config=configuracion, className="graph", style=estilo)
    )

    cuerpo = [
        html.Div(
            [html.H5(titulo, className="card-title-custom"), chip_glia(titulo, nota)],
            className="card-titulo-fila",
        ),
        grafico,
    ]
    if nota:
        cuerpo.append(html.Div(con_formulas(nota), className="graph-note",
                               style={"color": INK_MUTED}))
    return dbc.Card(dbc.CardBody(cuerpo), className=f"soft-card {className}".strip())


def data_table(
    encabezados: Sequence[str],
    filas: Sequence[Sequence[str]],
    resaltar_primera_columna: bool = True,
) -> dbc.Table:
    """Tabla Bootstrap con estilo propio (usada para operacionalización y EDA)."""
    header = html.Thead(html.Tr([html.Th(con_formulas(h)) for h in encabezados]))
    cuerpo = []
    for fila in filas:
        celdas = []
        for i, valor in enumerate(fila):
            clase = "cell-strong" if (i == 0 and resaltar_primera_columna) else ""
            celdas.append(html.Td(con_formulas(valor), className=clase))
        cuerpo.append(html.Tr(celdas))
    return dbc.Table(
        [header, html.Tbody(cuerpo)],
        bordered=False,
        hover=True,
        responsive=True,
        className="soft-table",
    )


def legend_chip(texto: str, color: str) -> html.Span:
    """Etiqueta con punto de color (refuerza la identidad sin depender del color)."""
    return html.Span(
        [html.Span(className="chip-dot", style={"background": color}), texto],
        className="legend-chip",
    )


def series_chips() -> html.Div:
    """Leyenda fija LGG / GBM para acompañar a los gráficos."""
    return html.Div(
        [
            legend_chip("LGG · glioma de bajo grado", COLOR_LGG),
            legend_chip("GBM · glioblastoma multiforme", COLOR_GBM),
        ],
        className="legend-chips",
    )

