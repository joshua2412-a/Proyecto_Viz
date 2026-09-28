"""
Tema visual del dashboard: paleta pastel y plantilla de Plotly.

La paleta de series (azul / coral) está validada para daltonismo y contraste,
por lo que se usa SIEMPRE en el mismo orden:
    slot 1 -> LGG (glioma de bajo grado)    slot 2 -> GBM (glioblastoma)

Es el mismo criterio de color del libro (azul = LGG, rojo = GBM), de modo que
una figura del dashboard y su equivalente en el Jupyter Book se leen igual.

Como el coral y el verde quedan por debajo de 3:1 de contraste contra el fondo
blanco, ningún gráfico comunica identidad solo con color: todos llevan leyenda
y etiquetas directas.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
# Paleta
# --------------------------------------------------------------------------- #
# Superficies pastel
BG_PAGE = "#F4F7FB"       # fondo general de la página
BG_CARD = "#FFFFFF"       # fondo de tarjetas y gráficos
BG_SOFT = "#EAF1FA"       # bloques destacados (azul pastel)
BG_SOFT_ALT = "#FDEEE8"   # bloques destacados (coral pastel)
BG_SOFT_MINT = "#E7F5F0"  # bloques destacados (menta pastel)
BG_SOFT_LILAC = "#F0EBF9" # bloques destacados (lila pastel)

# Tinta / texto
INK = "#1F2933"
INK_SOFT = "#52606D"
INK_MUTED = "#7B8794"
GRID = "#E4E9F0"
AXIS = "#C7D0DB"

# Series categóricas (orden fijo, nunca ciclado)
SERIES = ["#4E8CD9", "#E0785A", "#2FA987", "#9B7BD4", "#E0A83C"]
COLOR_LGG = SERIES[0]
COLOR_GBM = SERIES[1]

# Mapa de colores para la variable objetivo
COLOR_GRADE_MAP = {"LGG": COLOR_LGG, "GBM": COLOR_GBM}
ORDEN_GRADE = ["LGG", "GBM"]

# Rampa secuencial azul (magnitud: mapas de calor, matriz de confusión)
SEQ_BLUE = [
    "#EDF4FD",
    "#CDE2FB",
    "#9EC5F4",
    "#6DA7EC",
    "#4E8CD9",
    "#2A78D6",
    "#1C5CAB",
]

# Rampa divergente azul <-> coral con gris neutro (polaridad: correlaciones,
# coeficientes, contribuciones al log-odds)
DIV_BLUE_CORAL = [
    [0.0, "#1C5CAB"],
    [0.25, "#9EC5F4"],
    [0.5, "#F0EFEC"],
    [0.75, "#F3B39B"],
    [1.0, "#C74B27"],
]

# Colores de estado (reservados: nunca se usan como "serie N")
STATUS_GOOD = "#0CA30C"
STATUS_WARNING = "#E09B12"
STATUS_CRITICAL = "#D03B3B"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# --------------------------------------------------------------------------- #
# Plantilla de Plotly
# --------------------------------------------------------------------------- #
PASTEL_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        colorway=SERIES,
        font=dict(family=FONT_FAMILY, size=13, color=INK_SOFT),
        title=dict(font=dict(size=16, color=INK), x=0.01, xanchor="left"),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        margin=dict(l=60, r=24, t=56, b=52),
        hoverlabel=dict(
            bgcolor=BG_CARD,
            bordercolor=AXIS,
            font=dict(family=FONT_FAMILY, size=12, color=INK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=AXIS,
            ticks="outside",
            tickcolor=AXIS,
            tickfont=dict(color=INK_MUTED, size=12),
            title_font=dict(color=INK_SOFT, size=13),
        ),
        yaxis=dict(
            gridcolor=GRID,
            gridwidth=1,
            zeroline=False,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(color=INK_MUTED, size=12),
            title_font=dict(color=INK_SOFT, size=13),
        ),
    )
)

pio.templates["pastel"] = PASTEL_TEMPLATE
pio.templates.default = "pastel"


def apply_theme(fig: go.Figure, height: int | None = None, **layout_kwargs) -> go.Figure:
    """Aplica la plantilla pastel a una figura y ajusta opciones comunes.

    Se usa como último paso en cada gráfico para que todas las figuras del
    dashboard compartan tipografía, márgenes y colores sin repetir código.
    """
    fig.update_layout(template=PASTEL_TEMPLATE, **layout_kwargs)
    if height is not None:
        fig.update_layout(height=height)
    return fig
