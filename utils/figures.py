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
from utils.formato import num, pct
from utils.data_loader import (
    TIPO_NUMERICA,
    asociacion_con_grado,
    distribucion_clinica,
    distribucion_univariada,
    estadisticas_edad,
    get_dataframe,
    matriz_asociacion_genes,
    matriz_spearman,
    nombre_variable,
    prevalencia_genes,
    proporcion_grado,
    tabla_bivariada,
    tipo_variable,
)
from utils.theme import (
    AXIS,
    DIV_BLUE_CORAL,
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
            annotation_text=f"mediana {grado}: {num(mediana)}",
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
            text=f"media {num(fila['media'])} ± {num(fila['desviacion'])}",
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
    tabla["etiqueta"] = tabla["porcentaje"].map(pct)

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
                text=[pct(valor) for valor in datos[grado]],
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
    # Sin ρ no hay barra que dibujar: Race sale de aquí porque es nominal y el
    # libro también la excluye de su gráfico de Spearman. Su asociación se
    # reporta por V de Cramér en la tabla y en la pestaña de exploración.
    datos = datos.dropna(subset=["rho"])
    datos["efecto"] = np.where(datos["rho"] >= 0, "Hacia GBM", "Hacia LGG")
    datos["etiqueta"] = datos["rho"].map(lambda v: num(v, 2, signo=True))
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


# --------------------------------------------------------------------------- #
# 7. Exploración interactiva: una figura por tipo de variable
# --------------------------------------------------------------------------- #
# Las dos funciones que siguen alimentan la pestaña Exploración, donde el
# visitante elige la variable. La forma de la figura la decide el tipo de dato,
# no un selector aparte: una continua pide una distribución y una categórica
# pide una comparación de proporciones.
def fig_univariada(variable: str, ambito: str = "train") -> go.Figure:
    """La variable sola, sin cruzarla con el grado.

    Una sola serie, así que va en el tono neutro de magnitud: aquí el color no
    distingue grados y pintarla de azul de LGG sugeriría algo que no es.
    """
    nombre = nombre_variable(variable)

    if tipo_variable(variable) == TIPO_NUMERICA:
        serie = get_dataframe(ambito)[variable]
        fig = go.Figure(
            go.Histogram(
                x=serie,
                nbinsx=30,
                marker=dict(color=TONO_BARRA, line=dict(color=BG_CARD, width=1)),
                hovertemplate="%{x:.0f} años<br>%{y} pacientes<extra></extra>",
            )
        )
        # La mediana marcada: da el centro sin que la arrastren los extremos
        fig.add_vline(
            x=float(serie.median()),
            line=dict(color=INK_SOFT, width=1.2, dash="dash"),
            annotation_text=f"mediana {num(serie.median(), 0)}",
            annotation_position="top",
            annotation_font=dict(size=11, color=INK_SOFT),
        )
        fig.update_layout(
            xaxis_title=LABELS.get(variable, nombre),
            yaxis_title="Pacientes",
            bargap=0.04,
        )
        return apply_theme(_redondear_barras(fig, 3), height=330, showlegend=False)

    datos = distribucion_univariada(variable, ambito)
    fig = go.Figure(
        go.Bar(
            x=datos["pacientes"],
            y=datos["categoria"],
            orientation="h",
            marker=dict(color=TONO_BARRA),
            text=[pct(p) for p in datos["porcentaje"]],
            textposition="outside",
            textfont=dict(size=11, color=INK_SOFT),
            hovertemplate="<b>%{y}</b><br>%{x} pacientes<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Pacientes",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridcolor=GRID),
        margin=dict(l=8, r=64, t=20, b=44),
    )
    altura = max(230, 74 + 46 * len(datos))
    return apply_theme(_redondear_barras(fig), height=altura, showlegend=False)


def fig_bivariada(variable: str, ambito: str = "train") -> go.Figure:
    """La variable frente al grado tumoral.

    Para una categórica se dibujan proporciones dentro de cada categoría y no
    conteos: con clases de tamaños muy distintos, un conteo bruto solo muestra
    cuál es el grupo más grande. La línea de referencia marca el porcentaje de
    GBM de toda la cohorte, así se ve de un vistazo qué categorías se salen de
    la media y cuáles solo repiten el promedio.
    """
    if tipo_variable(variable) == TIPO_NUMERICA:
        df = get_dataframe(ambito)
        fig = go.Figure()
        for grado in ORDEN_GRADE:
            serie = df.loc[df["grade_label"] == grado, variable]
            fig.add_trace(
                go.Histogram(
                    x=serie,
                    name=grado,
                    nbinsx=26,
                    histnorm="percent",
                    marker=dict(color=COLOR_GRADE_MAP[grado],
                                line=dict(color=BG_CARD, width=1)),
                    opacity=0.78,
                    hovertemplate=f"<b>{grado}</b><br>%{{x:.0f}} años<br>"
                                  "%{y:.1f} % del grupo<extra></extra>",
                )
            )
        fig.update_layout(
            barmode="overlay",
            xaxis_title=LABELS.get(variable, variable),
            yaxis_title="% dentro de cada grado",
            bargap=0.04,
        )
        return apply_theme(fig, height=360)

    tabla = tabla_bivariada(variable, ambito)
    # Orden: la categoría con más GBM arriba, para que la lectura empiece por
    # donde está la señal en vez de por el orden alfabético.
    orden = (
        tabla[tabla["grade_label"] == "GBM"]
        .sort_values("porcentaje")["categoria"]
        .tolist()
    )
    faltan = [c for c in tabla["categoria"].unique() if c not in orden]
    orden = faltan + orden

    tamanos = tabla.groupby("categoria")["total_categoria"].first()
    etiquetas = {c: f"{c}  (n = {tamanos[c]})" for c in orden}

    fig = go.Figure()
    for grado in ORDEN_GRADE:
        parte = tabla[tabla["grade_label"] == grado].set_index("categoria").reindex(orden)
        fig.add_trace(
            go.Bar(
                x=parte["porcentaje"],
                y=[etiquetas[c] for c in orden],
                name=grado,
                orientation="h",
                marker=dict(color=COLOR_GRADE_MAP[grado],
                            line=dict(color=BG_CARD, width=1)),
                text=[pct(v, 0) if v >= 9 else "" for v in parte["porcentaje"]],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=11, color="#FFFFFF"),
                customdata=parte["pacientes"],
                hovertemplate=f"<b>{grado}</b><br>%{{x:.1f}} %% de la categoría"
                              "<br>%{customdata} pacientes<extra></extra>",
            )
        )

    gbm_cohorte = float(
        (get_dataframe(ambito)["grade_label"] == "GBM").mean() * 100
    )
    fig.add_vline(
        x=100 - gbm_cohorte,
        line=dict(color=INK_SOFT, width=1.2, dash="dot"),
        annotation_text=f"GBM en la cohorte: {pct(gbm_cohorte, 0)}",
        annotation_position="top left",
        annotation_font=dict(size=11, color=INK_SOFT),
    )

    fig.update_layout(
        barmode="stack",
        # Plotly invierte la leyenda de las barras apiladas por defecto, y
        # entonces no coincide con el orden en que se ven los colores.
        legend_traceorder="normal",
        xaxis=dict(title="% dentro de la categoría", range=[0, 100],
                   showgrid=True, gridcolor=GRID, ticksuffix=" %"),
        yaxis_title="",
        margin=dict(l=8, r=20, t=52, b=44),
    )
    altura = max(260, 118 + 52 * len(orden))
    return apply_theme(_redondear_barras(fig, 3), height=altura)


def fig_matriz_spearman(ambito: str = "train") -> go.Figure:
    """Matriz de Spearman del grado y las 22 predictoras con orden.

    Triángulo inferior y sin números en las casillas, como la del libro: con
    529 celdas, anotarlas todas convierte la figura en una hoja de cálculo y
    se pierde justamente lo que un mapa de calor hace bien, que es enseñar el
    patrón de un vistazo. Los valores concretos están en la tabla de
    asociación y en el detalle de cada variable.

    La escala es divergente con gris en el cero porque el signo importa: azul
    hacia LGG, coral hacia GBM, y el punto medio neutro para que la ausencia
    de correlación no parezca un valor más.
    """
    matriz = matriz_spearman(ambito)
    etiquetas = list(matriz.columns)

    # Se oculta el triángulo superior: es el reflejo del inferior y duplicarlo
    # solo añade ruido.
    valores = matriz.to_numpy(copy=True).astype(float)
    valores[np.triu_indices_from(valores, k=1)] = np.nan

    fig = go.Figure(
        go.Heatmap(
            z=valores,
            x=etiquetas,
            y=etiquetas,
            colorscale=DIV_BLUE_CORAL,
            zmid=0,
            zmin=-1,
            zmax=1,
            xgap=1,
            ygap=1,
            hovertemplate="<b>%{y}</b> y <b>%{x}</b><br>r = %{z:.2f}<extra></extra>",
            colorbar=dict(
                title=dict(text="r", side="top", font=dict(size=11, color=INK_SOFT)),
                thickness=11,
                len=0.8,
                tickfont=dict(size=10, color=INK_SOFT),
                outlinewidth=0,
            ),
        )
    )
    # Sin anclar la escala de los ejes: forzar celdas cuadradas dejaba media
    # tarjeta vacía a los lados y encogía las etiquetas hasta lo ilegible. En
    # una matriz de correlación la proporción de la celda no codifica nada.
    fig.update_layout(
        xaxis=dict(showgrid=False, tickangle=-55, tickfont=dict(size=10), ticks=""),
        yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=10),
                   ticks=""),
        margin=dict(l=4, r=4, t=10, b=4),
        plot_bgcolor=BG_CARD,
    )
    return apply_theme(fig, height=620)
