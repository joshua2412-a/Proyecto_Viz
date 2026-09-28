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
  * Rejilla y ejes discretos; marcas finas, con esquinas redondeadas y 2 px de
    separación entre rellenos contiguos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import (
    GRADE_LABELS,
    LABELS,
    UMBRAL_GBM,
    UMBRAL_LGG,
)
from utils.data_loader import (
    asociacion_con_grado,
    contribuciones_prediccion,
    distribucion_clinica,
    estadisticas_edad,
    get_dataframe,
    load_metrics,
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
    DIV_BLUE_CORAL,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    ORDEN_GRADE,
    SEQ_BLUE,
    STATUS_CRITICAL,
    STATUS_GOOD,
    STATUS_WARNING,
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
    return apply_theme(fig, height=360, showlegend=True)


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


# --------------------------------------------------------------------------- #
# 7. Desempeño del modelo
# --------------------------------------------------------------------------- #
def fig_roc() -> go.Figure:
    """Curva ROC del modelo frente al clasificador aleatorio."""
    metricas = load_metrics()
    fpr = metricas["roc_curve"]["fpr"]
    tpr = metricas["roc_curve"]["tpr"]
    auc = metricas["roc_auc"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Clasificador aleatorio",
            line=dict(color=INK_MUTED, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Regresión logística (AUC = {auc:.3f})",
            line=dict(color=COLOR_LGG, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(78,140,217,0.12)",
            hovertemplate=(
                "Falsos positivos: %{x:.2f}<br>Sensibilidad: %{y:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_annotation(
        x=0.62,
        y=0.28,
        text=f"<b>AUC = {auc:.3f}</b>",
        showarrow=False,
        font=dict(size=18, color=INK),
    )
    fig.update_layout(
        xaxis_title="Tasa de falsos positivos (1 - especificidad)",
        yaxis_title="Tasa de verdaderos positivos (sensibilidad)",
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor=GRID),
        yaxis=dict(range=[0, 1.02]),
    )
    return apply_theme(fig, height=420)


def fig_matriz_confusion() -> go.Figure:
    """Matriz de confusión del conjunto de prueba con conteos absolutos."""
    cm = np.array(load_metrics()["matriz_confusion"])
    x = [f"Predice {GRADE_LABELS[0]}", f"Predice {GRADE_LABELS[1]}"]
    y = [f"Real {GRADE_LABELS[0]}", f"Real {GRADE_LABELS[1]}"]

    fig = px.imshow(
        cm,
        x=x,
        y=y,
        color_continuous_scale=SEQ_BLUE,
        text_auto=True,
        aspect="auto",
    )
    fig.update_traces(
        textfont=dict(size=22),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Pacientes: %{z}<extra></extra>",
        xgap=3,
        ygap=3,
    )
    fig.update_layout(coloraxis_showscale=False, xaxis=dict(side="bottom"))
    return apply_theme(fig, height=380, margin=dict(l=120, r=24, t=40, b=60))


def fig_coeficientes(top_n: int = 12) -> go.Figure:
    """Coeficientes del modelo: hacia dónde empuja cada variable y cuánto."""
    coeficientes = load_metrics()["coeficientes"]
    datos = pd.DataFrame(
        [fila for fila in coeficientes if fila["coeficiente"] != 0][:top_n]
    )
    if datos.empty:
        return apply_theme(go.Figure(), height=300)

    datos["nombre"] = datos["variable"].map(_nombre_variable)
    datos["efecto"] = np.where(datos["coeficiente"] >= 0, "Hacia GBM", "Hacia LGG")
    datos["etiqueta"] = datos["odds_ratio"].map(lambda v: f"OR {v:.2f}")
    datos = datos.sort_values("coeficiente")

    fig = px.bar(
        datos,
        x="coeficiente",
        y="nombre",
        orientation="h",
        color="efecto",
        text="etiqueta",
        color_discrete_map={"Hacia GBM": COLOR_GBM, "Hacia LGG": COLOR_LGG},
        custom_data=["odds_ratio"],
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color=INK_SOFT),
        cliponaxis=False,
        marker_line=dict(color=BG_CARD, width=1),
        hovertemplate=(
            "<b>%{y}</b><br>Coeficiente: %{x:.3f}"
            "<br>Odds ratio: %{customdata[0]:.3f}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="Coeficiente (efecto sobre el log-odds de GBM)",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS),
        legend_title_text="",
    )
    return apply_theme(
        _redondear_barras(fig), height=32 * len(datos) + 150, bargap=0.3,
        margin=dict(l=200, r=80, t=56, b=52),
    )


def _nombre_variable(variable: str) -> str:
    """Traduce el nombre técnico de una variable transformada a texto legible."""
    equivalencias = {
        "Age_at_diagnosis": "Edad al diagnóstico",
        "Gender_1": "Género femenino",
        "Race_0": "Raza: White",
        "Race_1": "Raza: Black or African American",
        "Race_2": "Raza: Asian",
        "Race_3": "Raza: American Indian or Alaska Native",
    }
    if variable in equivalencias:
        return equivalencias[variable]
    if variable.startswith("Race_0"):
        return "Raza: White"
    return f"{variable} mutado"


# --------------------------------------------------------------------------- #
# 8. Predicción individual
# --------------------------------------------------------------------------- #
def fig_gauge_probabilidad(probabilidad: float) -> go.Figure:
    """Indicador de aguja con la probabilidad estimada de GBM.

    Las bandas usan colores de estado (nunca colores de serie) y siempre van
    acompañadas de una etiqueta textual en la pestaña, de modo que el nivel de
    confianza no se comunica solo con color.
    """
    valor = probabilidad * 100
    if probabilidad >= UMBRAL_GBM:
        color_aguja = STATUS_CRITICAL
    elif probabilidad >= UMBRAL_LGG:
        color_aguja = STATUS_WARNING
    else:
        color_aguja = STATUS_GOOD

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=valor,
            number={"suffix": "%", "font": {"size": 42, "color": color_aguja}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": AXIS,
                    "tickfont": {"size": 11, "color": INK_MUTED},
                },
                "bar": {"color": color_aguja, "thickness": 0.28},
                "bgcolor": BG_CARD,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, UMBRAL_LGG * 100], "color": "#E7F5F0"},
                    {"range": [UMBRAL_LGG * 100, UMBRAL_GBM * 100], "color": "#FDF3DF"},
                    {"range": [UMBRAL_GBM * 100, 100], "color": "#FBE6E6"},
                ],
                "threshold": {
                    "line": {"color": INK_SOFT, "width": 2},
                    "thickness": 0.8,
                    "value": 50,  # umbral de decisión del clasificador
                },
            },
        )
    )
    return apply_theme(fig, height=280, margin=dict(l=30, r=30, t=30, b=10))


def fig_contribuciones(registro: dict, top_n: int = 10) -> go.Figure:
    """Desglose de la predicción: cuánto aporta cada variable al log-odds.

    En un modelo lineal la predicción es exactamente la suma de estas
    contribuciones más el intercepto, así que el gráfico no es una
    aproximación del razonamiento del modelo: es el razonamiento del modelo.
    """
    datos = contribuciones_prediccion(registro, top_n=top_n)
    if datos.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Ninguna variable del perfil modifica la predicción base.",
            showarrow=False,
            font=dict(size=13, color=INK_MUTED),
        )
        return apply_theme(fig, height=300)

    datos["nombre"] = datos["variable"].map(_nombre_variable)
    datos["etiqueta"] = datos["contribucion"].map(lambda v: f"{v:+.2f}")
    datos = datos.sort_values("contribucion")

    fig = px.bar(
        datos,
        x="contribucion",
        y="nombre",
        orientation="h",
        color="empuja_hacia",
        text="etiqueta",
        color_discrete_map={"GBM": COLOR_GBM, "LGG": COLOR_LGG},
        custom_data=["coeficiente"],
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color=INK_SOFT),
        cliponaxis=False,
        marker_line=dict(color=BG_CARD, width=1),
        hovertemplate=(
            "<b>%{y}</b><br>Aporte al log-odds: %{x:+.3f}"
            "<br>Coeficiente del modelo: %{customdata[0]:.3f}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="Aporte al log-odds de GBM (negativo = empuja hacia LGG)",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS),
        legend_title_text="",
    )
    return apply_theme(
        _redondear_barras(fig), height=32 * len(datos) + 150, bargap=0.3,
        margin=dict(l=200, r=70, t=56, b=52),
    )


def fig_edad_vs_probabilidad(registro: dict, paso: float = 1.0) -> go.Figure:
    """Cómo cambia la probabilidad estimada al mover solo la edad del paciente.

    Mantiene fijo el resto del perfil: es la lectura clínica de la pendiente de
    la edad para ese perfil genético concreto, no una curva promedio.
    """
    from utils.data_loader import predecir_gbm  # import local: evita ciclo

    edades = np.arange(15.0, 90.0 + paso, paso)
    probabilidades = [
        predecir_gbm({**registro, "Age_at_diagnosis": float(edad)}) * 100
        for edad in edades
    ]
    edad_actual = float(registro["Age_at_diagnosis"])
    probabilidad_actual = predecir_gbm(registro) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edades,
            y=probabilidades,
            mode="lines",
            name="Perfil evaluado",
            line=dict(color=COLOR_LGG, width=2.5),
            hovertemplate="Edad %{x:.0f} años<br>P(GBM) = %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color=INK_MUTED,
        line_width=1.5,
        annotation_text="umbral de decisión (50%)",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color=INK_MUTED),
    )
    fig.add_trace(
        go.Scatter(
            x=[edad_actual],
            y=[probabilidad_actual],
            mode="markers+text",
            name="Edad indicada",
            marker=dict(
                size=13,
                color=INK,
                symbol="diamond",
                line=dict(color=BG_CARD, width=2),  # anillo de 2px sobre la línea
            ),
            text=[f" {probabilidad_actual:.1f}%"],
            textposition="middle right",
            textfont=dict(size=12, color=INK),
            hovertemplate="Edad indicada: %{x:.1f} años<br>P(GBM) = %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=LABELS["Age_at_diagnosis"],
        yaxis_title="Probabilidad estimada de GBM (%)",
        yaxis=dict(range=[0, 102]),
        xaxis=dict(showgrid=True, gridcolor=GRID),
    )
    return apply_theme(fig, height=360)
