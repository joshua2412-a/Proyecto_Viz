"""
Tema visual del dashboard: paleta sobria y plantilla de Plotly.

CRITERIO DE COLOR
-----------------
El color hace un trabajo o no está. Solo dos colores significan algo en todo el
tablero, y significan siempre lo mismo:

    COLOR_LGG -> glioma de bajo grado        COLOR_GBM -> glioblastoma

Son los mismos que usa el Jupyter Book (azul = LGG, coral = GBM), de modo que
una figura del dashboard y su equivalente en el libro se leen igual. No se
cambian sin recompilar el libro.

Todo lo demás —tarjetas, cabeceras, bloques destacados, indicadores— es neutro.
Las superficies son grises sin tinte para que el azul y el coral sean el único
color de la página, que es lo que los hace legibles como código semántico.

Reglas que se siguen aquí:
  - Una serie sola no se colorea por grupo: usa COLOR_MARK.
  - El texto lleva tokens de texto (INK*), nunca el color de una serie.
  - Los colores de estado están reservados para estado; no son "serie N".
  - Magnitud -> rampa de un solo tono. Polaridad -> dos tonos con gris neutro
    en el medio. Nunca un arcoíris.

CONTRASTE
---------
La paleta pasa las seis comprobaciones de validate_palette.js salvo una: el
coral queda en 2,92:1 contra el blanco, por debajo del 3:1 recomendado. Es el
precio de mantener el color del libro, y se compensa como exige la norma: cada
figura lleva leyenda o etiquetas directas, así que la identidad nunca depende
solo del color. Si algún día se recompila el libro, #C85A32 sube a 3,4:1.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
# Superficies frías (gris con cast azul)
# --------------------------------------------------------------------------- #
BG_PAGE = "#FFFFFF"   # fondo general de la página
BG_CARD = "#FFFFFF"   # tarjetas y área de gráficos
BG_SOFT = "#F2F3FA"   # bloques destacados e insertos (único, ya no hay pasteles)

# Cabecera en tinta profunda: marco corporativo sin comprometer el cuerpo
HEADER_BG = "#141B4D"
HEADER_INK = "#EEF3F9"
HEADER_INK_MUTED = "#A9B1DB"

# --------------------------------------------------------------------------- #
# Tinta y líneas
# --------------------------------------------------------------------------- #
INK = "#121A3A"        # títulos y cifras
INK_SOFT = "#3B4266"   # cuerpo de texto
INK_MUTED = "#5D6487"  # etiquetas, detalles, ticks
HAIRLINE = "#E6E8F2"   # bordes de tarjeta y separadores
GRID = "#ECEEF6"       # rejilla de los gráficos (más clara que el filete)
AXIS = "#CDD1E6"       # líneas de eje y marcas

# --------------------------------------------------------------------------- #
# Color con significado
# --------------------------------------------------------------------------- #
COLOR_LGG = "#4E8CD9"   # glioma de bajo grado
COLOR_GBM = "#E0785A"   # glioblastoma multiforme

# Marca de una sola serie: cuando no hay grupos que distinguir, no se usa el
# color de LGG ni el de GBM, porque ahí no significarían nada.
COLOR_MARK = "#141B4D"

SERIES = [COLOR_LGG, COLOR_GBM]

COLOR_GRADE_MAP = {"LGG": COLOR_LGG, "GBM": COLOR_GBM}
ORDEN_GRADE = ["LGG", "GBM"]

# Acento de interfaz (azul acero). Es cromo —filetes, indicadores, enlaces—,
# no un dato: más oscuro y menos saturado que COLOR_LGG para no confundirse
# con él (ΔE 21,4 entre ambos).
ACCENT_NEUTRAL = "#141B4D"

# Los tres modelos del capítulo 2. No son grados, así que no llevan azul ni
# coral: tres tintas neutras de distinta claridad (contraste con el blanco de
# 8,4 : 1, 5,6 : 1 y 3,5 : 1) que siempre van con un trazo distinto y con su
# nombre en la leyenda o junto a la marca. El color nunca es la única pista.
TONOS_MODELO = {
    "logistica": "#1E4E86",
    "random_forest": "#52698A",
    "xgboost": "#7A8BA3",
}
TRAZO_MODELO = {"logistica": "solid", "random_forest": "dash", "xgboost": "dot"}

# --------------------------------------------------------------------------- #
# Rampas
# --------------------------------------------------------------------------- #
# Magnitud: un solo tono, claro a oscuro (mapas de calor)
SEQ_BLUE = [
    "#F2F5FA",
    "#D8E3F3",
    "#B0C8E7",
    "#84A9D6",
    "#5C8AC4",
    "#3A6CAB",
    "#1F4E85",
]

# Polaridad: azul <-> coral con gris neutro en el punto medio
DIV_BLUE_CORAL = [
    [0.0, "#1F4E85"],
    [0.25, "#B0C8E7"],
    [0.5, "#EDF1F7"],
    [0.75, "#F0B49C"],
    [1.0, "#C2542F"],
]

# --------------------------------------------------------------------------- #
# Estado (reservados: nunca se usan como "serie N")
# --------------------------------------------------------------------------- #
STATUS_GOOD = "#1F7A43"
STATUS_WARNING = "#9A6B08"
STATUS_CRITICAL = "#B3261E"

# --------------------------------------------------------------------------- #
# Tipografía
# --------------------------------------------------------------------------- #
# Plus Jakarta Sans (Google Fonts, la carga app.py) con la pila del sistema de
# respaldo: si la fuente no llega, el tablero se lee igual.
FONT_FAMILY = '"Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
FONT_MONO = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace'

# --------------------------------------------------------------------------- #
# Plantilla de Plotly
# --------------------------------------------------------------------------- #
# Ejes y rejilla recesivos: la rejilla solo en el eje de magnitud, sin línea de
# cero salvo donde el signo importa, y ninguna caja alrededor del gráfico.
SOBRIO_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        colorway=SERIES,
        # Coma decimal y punto de millar: sin esto, los ejes y los tooltips
        # escriben "42.0" mientras el texto de las pestañas escribe "42,0".
        separators=",.",
        font=dict(family=FONT_FAMILY, size=14, color=INK_SOFT),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        margin=dict(l=56, r=20, t=44, b=48),
        hoverlabel=dict(
            bgcolor=BG_CARD,
            bordercolor=HAIRLINE,
            font=dict(family=FONT_FAMILY, size=13, color=INK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color=INK_SOFT),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=AXIS,
            linewidth=1,
            ticks="outside",
            ticklen=4,
            tickcolor=AXIS,
            tickfont=dict(color=INK_MUTED, size=13),
            title_font=dict(color=INK_SOFT, size=13),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor=GRID,
            gridwidth=1,
            zeroline=False,
            linecolor="rgba(0,0,0,0)",
            ticks="",
            tickfont=dict(color=INK_MUTED, size=13),
            title_font=dict(color=INK_SOFT, size=13),
            automargin=True,
        ),
    )
)

pio.templates["sobrio"] = SOBRIO_TEMPLATE
pio.templates.default = "sobrio"


def apply_theme(fig: go.Figure, height: int | None = None, **layout_kwargs) -> go.Figure:
    """Aplica la plantilla al gráfico y ajusta opciones comunes.

    Se usa como último paso en cada figura para que todas compartan tipografía,
    márgenes, rejilla y colores sin repetir código.
    """
    fig.update_layout(template=SOBRIO_TEMPLATE, **layout_kwargs)
    if height is not None:
        fig.update_layout(height=height)
    return fig
