"""
Constructores de figuras de Plotly.

Todos los gráficos del dashboard se crean aquí para que las pestañas no
repitan código de estilo y para que un cambio de paleta afecte a todo el
proyecto de una sola vez.

Criterios de diseño aplicados:
  * La forma sigue al trabajo del dato: magnitud -> barras de un solo tono,
    identidad -> dos series (azul = LGG, coral = GBM), polaridad -> escala
    divergente con gris neutro en el cero.
  * La identidad nunca depende solo del color: hay leyenda y etiquetas
    directas en todos los gráficos comparativos.
  * Todas las figuras describen los datos: el dashboard cubre el análisis
    exploratorio, no el modelado.
  * Rejilla y ejes discretos; marcas finas, con esquinas redondeadas y 2 px de
    separación entre rellenos contiguos.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.config import LABELS
from utils.data_loader import (
    asociacion_con_grado,
    distribucion_clinica,
    estadisticas_edad,
    get_dataframe,
    matriz_asociacion_genes,
    prevalencia_genes,
    proporcion_grado,
)
from utils.theme import (
    AXIS,
    BG_CARD,
    COLOR_GBM,
    COLOR_GRADE_MAP,
    COLOR_LGG,
    GRID,
    INK,
    INK_SOFT,
    ORDEN_GRADE,
    SEQ_BLUE,
    apply_theme,
)

# Tono principal para gráficos de magnitud (una sola serie)
TONO_BARRA = SEQ_BLUE[4]


def _redondear_barras(fig: go.Figure, radio: int = 4) -> go.Figure:
    """Redondea el extremo de las barras si la versión de Plotly lo soporta."""
    try:
        fig.update_traces(marker_cornerradius=radio, selector=dict(type="bar"))
    except (ValueError, TypeError):  # versiones de Plotly < 5.19
        pass
    return fig


# --------------------------------------------------------------------------- #
# 1. Composición de la cohorte (dona)
# --------------------------------------------------------------------------- #
def fig_distribucion_grado(ambito: str = "train") -> go.Figure:
    """Proporción de gliomas de bajo grado frente a glioblastomas."""
    datos = proporcion_grado(ambito)

    fig = px.pie(
        datos,
        names="grado",
        values="pacientes",
        hole=0.58,
        color="grado",
        color_discrete_map=COLOR_GRADE_MAP,
        category_orders={"grado": ORDEN_GRADE},
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=13, color=INK_SOFT),
        marker=dict(line=dict(color=BG_CARD, width=2)),  # separación de 2px
        hovertemplate="<b>%{label}</b><br>%{value} pacientes<br>%{percent}<extra></extra>",
        sort=False,
    )

    total = int(datos["pacientes"].sum())
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:12px'>pacientes</span>",
        showarrow=False,
        font=dict(size=26, color=INK),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    # Sin leyenda: cada porcion ya lleva su etiqueta y su porcentaje al lado,
    # asi que una leyenda solo repetiria la misma identidad por tercera vez.
    return apply_theme(fig, height=360, showlegend=False)


# --------------------------------------------------------------------------- #
# 2. Edad al diagnóstico
# --------------------------------------------------------------------------- #
def fig_edad_por_grado(ambito: str = "train", nbins: int = 28) -> go.Figure:
    """Distribución de la edad al diagnóstico en cada grado tumoral.

    Se normaliza a porcentaje dentro de cada grupo porque las clases no tienen
    el mismo tamaño (58/42): sin normalizar, la comparación de formas engaña.
    """
    df = get_dataframe(ambito)

    fig = px.histogram(
        df,
        x="Age_at_diagnosis",
        color="grade_label",
        barmode="overlay",
        histnorm="percent",
        nbins=nbins,
        opacity=0.72,
        color_discrete_map=COLOR_GRADE_MAP,
        category_orders={"grade_label": ORDEN_GRADE},
    )
    fig.update_traces(
        marker_line=dict(color=BG_CARD, width=1),
        hovertemplate="Edad %{x:.0f} años<br>%{y:.1f}% del grupo<extra>%{fullData.name}</extra>",
    )
    fig.update_layout(
        xaxis_title=LABELS["Age_at_diagnosis"],
        yaxis_title="% dentro de cada grado",
        legend_title_text="",
    )

    # Medianas por grupo: refuerzan la lectura sin depender del color
    for grado, color in (("LGG", COLOR_LGG), ("GBM", COLOR_GBM)):
        mediana = df.loc[df["grade_label"] == grado, "Age_at_diagnosis"].median()
        fig.add_vline(
            x=mediana,
            line_dash="dot",
            line_color=color,
            line_width=2,
            annotation_text=f"mediana {grado}: {mediana:.1f}",
            annotation_position="top",
            annotation_font=dict(size=11, color=color),
        )
    return apply_theme(fig, height=400)


def fig_boxplot_edad(ambito: str = "train") -> go.Figure:
    """Diagrama de caja de la edad al diagnóstico por grado tumoral."""
    df = get_dataframe(ambito)
    resumen = estadisticas_edad(ambito).set_index("grade_label")

    fig = px.box(
        df,
        x="grade_label",
        y="Age_at_diagnosis",
        color="grade_label",
        color_discrete_map=COLOR_GRADE_MAP,
        category_orders={"grade_label": ORDEN_GRADE},
        points="outliers",
    )
    fig.update_traces(
        marker=dict(size=6, line=dict(color=BG_CARD, width=1)),
        hovertemplate="%{y:.1f} años<extra>%{x}</extra>",
    )

    # Etiqueta directa con la media de cada grupo
    for grado in ORDEN_GRADE:
        if grado not in resumen.index:
            continue
        fila = resumen.loc[grado]
        fig.add_annotation(
            x=grado,
            y=float(fila["maximo"]),
            text=f"media {fila['media']:.1f} ± {fila['desviacion']:.1f}",
            showarrow=False,
            yshift=18,
            font=dict(size=11, color=INK_SOFT),
        )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=LABELS["Age_at_diagnosis"],
        showlegend=False,
    )
    return apply_theme(fig, height=400)


# --------------------------------------------------------------------------- #
# 3. Variables clínicas categóricas
# --------------------------------------------------------------------------- #
def fig_clinica_por_grado(variable: str, ambito: str = "train") -> go.Figure:
    """Composición de Gender o Race dentro de cada grado tumoral (en %)."""
    tabla = distribucion_clinica(variable, ambito)
    tabla["etiqueta"] = tabla["porcentaje"].map(lambda v: f"{v:.1f}%")

    fig = px.bar(
        tabla,
        x="porcentaje",
        y="categoria",
        color="grade_label",
        orientation="h",
        barmode="group",
        text="etiqueta",
        color_discrete_map=COLOR_GRADE_MAP,
        category_orders={"grade_label": ORDEN_GRADE},
        custom_data=["pacientes"],
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color=INK_SOFT),
        cliponaxis=False,
        marker_line=dict(color=BG_CARD, width=1),
        hovertemplate=(
            "<b>%{y}</b><br>%{x:.1f}% del grado"
            "<br>%{customdata[0]} pacientes<extra>%{fullData.name}</extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="% dentro de cada grado",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID, range=[0, 105]),
        legend_title_text="",
    )
    altura = 320 if variable == "Gender" else 420
    return apply_theme(
        _redondear_barras(fig), height=altura, bargap=0.3,
        margin=dict(l=210, r=60, t=56, b=52),
    )


# --------------------------------------------------------------------------- #
# 4. Prevalencia de mutaciones
# --------------------------------------------------------------------------- #
def fig_prevalencia_genes(ambito: str = "train", top_n: int = 12) -> go.Figure:
    """Prevalencia de cada mutación en LGG y en GBM, gen a gen.

    Ordenado por la diferencia entre grados: arriba quedan los genes que más
    separan a los dos grupos, que son los que el modelo acaba usando.
    """
    datos = prevalencia_genes(ambito)
    datos = (
        datos.reindex(datos["diferencia"].abs().sort_values().index)
        .tail(top_n)
    )

    fig = go.Figure()
    for grado, color in (("LGG", COLOR_LGG), ("GBM", COLOR_GBM)):
        fig.add_trace(
            go.Bar(
                y=datos["gen"],
                x=datos[grado],
                name=grado,
                orientation="h",
                marker=dict(color=color, line=dict(color=BG_CARD, width=1)),
                text=[f"{valor:.1f}%" for valor in datos[grado]],
                textposition="outside",
                textfont=dict(size=11, color=INK_SOFT),
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>Mutado en %{x:.1f}% de los casos"
                    f"<extra>{grado}</extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="% de pacientes con el gen mutado",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID),
        legend_title_text="",
    )
    return apply_theme(
        _redondear_barras(fig), height=32 * top_n + 140, bargap=0.28,
        margin=dict(l=90, r=70, t=56, b=52),
    )


# --------------------------------------------------------------------------- #
# 5. Asociación con el grado (polaridad)
# --------------------------------------------------------------------------- #
def fig_asociacion_grado(ambito: str = "train", top_n: int = 14) -> go.Figure:
    """Correlación de Spearman de cada variable con el grado tumoral.

    El signo es la información importante: negativo empuja hacia LGG, positivo
    hacia GBM. Las variables no significativas (p >= 0,05 o |r| < 0,10) se
    marcan con un patrón para no leerlas como señal.
    """
    datos = asociacion_con_grado(ambito).head(top_n).copy()
    datos["nombre"] = datos["variable"].map(lambda v: LABELS.get(v, v).replace("Mutación en ", ""))
    datos["efecto"] = np.where(datos["rho"] >= 0, "Hacia GBM", "Hacia LGG")
    datos["etiqueta"] = datos["rho"].map(lambda v: f"{v:+.2f}")
    datos = datos.sort_values("rho")

    fig = px.bar(
        datos,
        x="rho",
        y="nombre",
        orientation="h",
        color="efecto",
        text="etiqueta",
        color_discrete_map={"Hacia GBM": COLOR_GBM, "Hacia LGG": COLOR_LGG},
        custom_data=["p_valor", "v_cramer", "tipo"],
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color=INK_SOFT),
        cliponaxis=False,
        marker_line=dict(color=BG_CARD, width=1),
        hovertemplate=(
            "<b>%{y}</b> · %{customdata[2]}<br>Spearman r = %{x:.3f}"
            "<br>V de Cramér = %{customdata[1]:.3f}"
            "<br>p = %{customdata[0]:.4f}<extra></extra>"
        ),
    )

    # Marca de patrón para las variables sin significancia estadística
    no_significativas = datos.loc[~datos["significativa"], "nombre"].tolist()
    if no_significativas:
        fig.for_each_trace(
            lambda traza: traza.update(
                marker_pattern_shape=[
                    "/" if nombre in no_significativas else ""
                    for nombre in traza.y
                ]
            )
        )

    fig.update_layout(
        xaxis_title="Correlación de Spearman con el grado (0 = LGG, 1 = GBM)",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS),
        legend_title_text="",
    )
    return apply_theme(
        _redondear_barras(fig), height=32 * top_n + 140, bargap=0.3,
        margin=dict(l=170, r=70, t=56, b=52),
    )


# --------------------------------------------------------------------------- #
# 6. Multicolinealidad entre mutaciones
# --------------------------------------------------------------------------- #
def fig_matriz_genes(ambito: str = "train", top_n: int = 12) -> go.Figure:
    """Mapa de calor de la V de Cramér entre los genes más prevalentes.

    Magnitud, no polaridad: la V va de 0 a 1 y no tiene signo, así que se usa
    una rampa secuencial de un solo tono.
    """
    matriz = matriz_asociacion_genes(ambito, top_n=top_n)
    valores = matriz.values.copy()
    np.fill_diagonal(valores, np.nan)  # la diagonal es 1 por definición

    fig = px.imshow(
        valores,
        x=matriz.columns,
        y=matriz.index,
        zmin=0,
        zmax=max(0.6, float(np.nanmax(valores))),
        color_continuous_scale=SEQ_BLUE,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_traces(
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>V de Cramér = %{z:.3f}<extra></extra>",
        xgap=2,
        ygap=2,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="V", thickness=12, len=0.75),
        xaxis=dict(side="bottom", tickangle=-45),
    )
    return apply_theme(fig, height=520, margin=dict(l=90, r=24, t=40, b=90))
