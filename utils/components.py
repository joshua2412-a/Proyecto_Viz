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
# Cabeceras y textos
# --------------------------------------------------------------------------- #
def page_header(titulo: str, subtitulo: str, icono: str = "") -> html.Div:
    """Cabecera de pestaña: título grande + línea de contexto.

    `icono` es una clase de Bootstrap Icons (p. ej. "bi-bar-chart-line"), que ya
    viene cargada en app.py. Se usa un set monocromo en vez de emoji para que el
    color de la página siga reservado a los datos.
    """
    encabezado = []
    if icono:
        encabezado.append(html.I(className=f"bi {icono} page-icon"))
    encabezado.append(html.Span(titulo))
    return html.Div(
        [
            html.H2(encabezado, className="page-title"),
            html.P(subtitulo, className="page-subtitle"),
            html.Hr(className="page-rule"),
        ],
        className="page-header",
    )


def section_title(texto: str, nivel: int = 4) -> html.Div:
    """Título de sección dentro de una pestaña."""
    heading = {3: html.H3, 4: html.H4, 5: html.H5}.get(nivel, html.H4)
    return heading(texto, className="section-title")


def paragraph(texto) -> html.P:
    """Párrafo con el estilo de lectura del dashboard."""
    return html.P(texto, className="body-text")


def bullet_list(items: Iterable, color: str = ACCENT_NEUTRAL) -> html.Ul:
    """Lista con viñetas. El punto es neutro salvo que se pida otro color."""
    return html.Ul(
        [html.Li(item, className="body-text") for item in items],
        className="bullet-list",
        style={"--bullet-color": color},
    )


def enlace_externo(texto: str, url: str, icono: str = "↗") -> html.A:
    """Enlace que abre en una pestaña nueva (libro, dataset, repositorio)."""
    return html.A(
        f"{texto} {icono}".strip(),
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
) -> dbc.Card:
    """Tarjeta genérica con borde superior de color y cuerpo libre."""
    cuerpo = []
    if titulo:
        cuerpo.append(html.H5(titulo, className="card-title-custom"))
    if subtitulo:
        cuerpo.append(html.P(subtitulo, className="card-subtitle-custom"))
    cuerpo.append(html.Div(children))

    # Sin color por defecto: el color del tablero esta reservado a los datos.
    estilo = {"borderTop": f"2px solid {color}"} if color else {}
    return dbc.Card(
        dbc.CardBody(cuerpo),
        className=f"soft-card {className}".strip(),
        style=estilo,
    )


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
        html.Div(etiqueta.upper(), className="kpi-label"),
        html.Div(valor, className="kpi-value"),
    ]
    if detalle:
        cuerpo.append(html.Div(detalle, className="kpi-detail"))
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
    hijos.append(html.Div(texto, className="callout-text"))
    return html.Div(
        hijos,
        className="callout",
        style={"borderLeft": f"3px solid {color}", "background": BG_SOFT},
    )


# --------------------------------------------------------------------------- #
# Gráficos y tablas
# --------------------------------------------------------------------------- #
def graph_card(figura, titulo: str, nota: str = "", graph_id: str | None = None) -> dbc.Card:
    """Tarjeta que envuelve un gráfico de Plotly con título y nota al pie."""
    configuracion = {"displaylogo": False, "responsive": True}
    grafico = (
        dcc.Graph(figure=figura, id=graph_id, config=configuracion, className="graph")
        if graph_id
        else dcc.Graph(figure=figura, config=configuracion, className="graph")
    )

    cuerpo = [html.H5(titulo, className="card-title-custom"), grafico]
    if nota:
        cuerpo.append(html.Div(nota, className="graph-note", style={"color": INK_MUTED}))
    return dbc.Card(dbc.CardBody(cuerpo), className="soft-card")


def data_table(
    encabezados: Sequence[str],
    filas: Sequence[Sequence[str]],
    resaltar_primera_columna: bool = True,
) -> dbc.Table:
    """Tabla Bootstrap con estilo propio (usada para operacionalización y EDA)."""
    header = html.Thead(html.Tr([html.Th(h) for h in encabezados]))
    cuerpo = []
    for fila in filas:
        celdas = []
        for i, valor in enumerate(fila):
            clase = "cell-strong" if (i == 0 and resaltar_primera_columna) else ""
            celdas.append(html.Td(valor, className=clase))
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

