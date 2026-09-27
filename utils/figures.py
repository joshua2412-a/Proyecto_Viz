"""
Constructores de figuras de Plotly.

Todos los gráficos del dashboard se crean aquí para que las pestañas no
repitan código de estilo y para que un cambio de paleta afecte a todo el
proyecto de una sola vez.

Criterios de diseño aplicados:
  * La forma sigue al trabajo del dato: magnitud -> barras de un solo tono,
    identidad -> dos series (azul = permanece, coral = abandona),
    polaridad -> escala divergente con gris neutro en el cero.
  * La identidad nunca depende solo del color: hay leyenda y etiquetas
    directas en los gráficos comparativos.
  * Rejilla y ejes discretos; marcas finas y con esquinas redondeadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import LABELS, NUMERIC_FEATURES, RIESGO_ALTO, RIESGO_MEDIO, TARGET
from utils.data_loader import (
    get_dataframe,
    load_metrics,
    matriz_correlacion,
    tasa_por_departamento,
)
from utils.theme import (
    AXIS,
    BG_CARD,
    COLOR_ABANDONA,
    COLOR_ABANDONO_MAP,
    COLOR_PERMANECE,
    DIV_BLUE_CORAL,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
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
# 1. Composición de la plantilla (dona)
# --------------------------------------------------------------------------- #
def fig_dona_abandono() -> go.Figure:
    """Dona con la proporción de empleados que abandonan frente a los que permanecen."""
    df = get_dataframe()
    conteo = (
        df["abandono_label"]
        .value_counts()
        .rename_axis("estado")
        .reset_index(name="empleados")
    )

    fig = px.pie(
        conteo,
        names="estado",
        values="empleados",
        hole=0.58,
        color="estado",
        color_discrete_map=COLOR_ABANDONO_MAP,
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=13, color=INK_SOFT),
        marker=dict(line=dict(color=BG_CARD, width=2)),  # separación de 2px entre gajos
        hovertemplate="<b>%{label}</b><br>%{value:,} empleados<br>%{percent}<extra></extra>",
        sort=False,
    )

    tasa = df[TARGET].mean()
    fig.add_annotation(
        text=f"<b>{tasa:.1%}</b><br><span style='font-size:12px'>abandono</span>",
        showarrow=False,
        font=dict(size=26, color=COLOR_ABANDONA),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    return apply_theme(fig, height=360, showlegend=True)


# --------------------------------------------------------------------------- #
# 2. Tasa de abandono por departamento
# --------------------------------------------------------------------------- #
def fig_abandono_departamento(horizontal: bool = True) -> go.Figure:
    """Barras con la tasa de abandono de cada departamento y la media global."""
    resumen = tasa_por_departamento()
    media = get_dataframe()[TARGET].mean()
    resumen["tasa_pct"] = resumen["tasa"] * 100
    resumen["etiqueta"] = resumen["tasa"].map(lambda v: f"{v:.1%}")

    if horizontal:
        datos = resumen.sort_values("tasa")  # la barra más larga arriba
        fig = px.bar(
            datos,
            x="tasa_pct",
            y="departamento",
            orientation="h",
            text="etiqueta",
            custom_data=["empleados", "abandonos"],
        )
        fig.update_layout(
            xaxis_title="Tasa de abandono (%)",
            yaxis_title=None,
            xaxis=dict(showgrid=True, gridcolor=GRID),
            yaxis=dict(showgrid=False),
        )
        fig.add_vline(
            x=media * 100,
            line_dash="dash",
            line_color=INK_MUTED,
            line_width=1.5,
            annotation_text=f"Media global {media:.1%}",
            annotation_position="top right",
            annotation_font=dict(size=11, color=INK_MUTED),
        )
    else:
        fig = px.bar(
            resumen,
            x="departamento",
            y="tasa_pct",
            text="etiqueta",
            custom_data=["empleados", "abandonos"],
        )
        fig.update_layout(xaxis_title=None, yaxis_title="Tasa de abandono (%)")
        fig.add_hline(
            y=media * 100,
            line_dash="dash",
            line_color=INK_MUTED,
            line_width=1.5,
            annotation_text=f"Media global {media:.1%}",
            annotation_font=dict(size=11, color=INK_MUTED),
        )

    fig.update_traces(
        marker_color=TONO_BARRA,
        textposition="outside",
        textfont=dict(size=12, color=INK_SOFT),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Tasa: %{x:.1f}%"
            "<br>Empleados: %{customdata[0]:,}"
            "<br>Abandonos: %{customdata[1]:,}<extra></extra>"
            if horizontal
            else "<b>%{x}</b><br>Tasa: %{y:.1f}%"
            "<br>Empleados: %{customdata[0]:,}"
            "<br>Abandonos: %{customdata[1]:,}<extra></extra>"
        ),
    )
    return apply_theme(_redondear_barras(fig), height=380, bargap=0.35, showlegend=False)


# --------------------------------------------------------------------------- #
# 3. Impacto económico estimado
# --------------------------------------------------------------------------- #
def fig_costo_departamento(meses_reemplazo: int = 6) -> go.Figure:
    """Coste anual estimado de la rotación por departamento.

    Supuesto explícito: reemplazar a una persona cuesta `meses_reemplazo` meses
    de su salario (reclutamiento, selección, curva de aprendizaje y horas extra
    del equipo mientras la vacante está abierta).
    """
    df = get_dataframe()
    costos = (
        df[df[TARGET] == 1]
        .groupby("departamento")["salario"]
        .agg(["size", "sum"])
        .reset_index()
    )
    costos["costo"] = costos["sum"] * meses_reemplazo
    costos["costo_mm"] = costos["costo"] / 1_000_000
    costos["etiqueta"] = costos["costo_mm"].map(lambda v: f"${v:,.0f}M")
    costos = costos.sort_values("costo_mm")

    fig = px.bar(
        costos,
        x="costo_mm",
        y="departamento",
        orientation="h",
        text="etiqueta",
        custom_data=["size"],
    )
    fig.update_traces(
        marker_color=SEQ_BLUE[3],
        textposition="outside",
        textfont=dict(size=12, color=INK_SOFT),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Coste estimado: $%{x:,.0f} millones COP"
            "<br>Salidas: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title=f"Coste estimado (millones COP · {meses_reemplazo} meses de salario)",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID),
    )
    return apply_theme(_redondear_barras(fig), height=360, bargap=0.35, showlegend=False)


# --------------------------------------------------------------------------- #
# 4. Histogramas comparativos
# --------------------------------------------------------------------------- #
def fig_histograma(variable: str, nbins: int = 30) -> go.Figure:
    """Distribución de una variable numérica comparando los dos grupos.

    Se normaliza a porcentaje dentro de cada grupo porque las clases están
    desbalanceadas (~19% de abandonos): sin normalizar, la curva de "Abandona"
    sería invisible.
    """
    df = get_dataframe()
    fig = px.histogram(
        df,
        x=variable,
        color="abandono_label",
        barmode="overlay",
        histnorm="percent",
        nbins=nbins,
        opacity=0.72,
        color_discrete_map=COLOR_ABANDONO_MAP,
        category_orders={"abandono_label": ["Permanece", "Abandona"]},
    )
    fig.update_traces(marker_line=dict(color=BG_CARD, width=1))
    fig.update_layout(
        xaxis_title=LABELS.get(variable, variable),
        yaxis_title="% dentro de cada grupo",
        legend_title_text="",
    )

    # Líneas de promedio por grupo: refuerzan la lectura sin depender del color
    for grupo, color in (("Permanece", COLOR_PERMANECE), ("Abandona", COLOR_ABANDONA)):
        promedio = df.loc[df["abandono_label"] == grupo, variable].mean()
        fig.add_vline(
            x=promedio,
            line_dash="dot",
            line_color=color,
            line_width=2,
            annotation_text=f"media {grupo}: {promedio:,.1f}",
            annotation_position="top",
            annotation_font=dict(size=11, color=color),
        )
    return apply_theme(fig, height=400)


def fig_boxplot(variable: str) -> go.Figure:
    """Diagrama de caja de una variable numérica por grupo de abandono."""
    df = get_dataframe()
    fig = px.box(
        df,
        x="abandono_label",
        y=variable,
        color="abandono_label",
        color_discrete_map=COLOR_ABANDONO_MAP,
        category_orders={"abandono_label": ["Permanece", "Abandona"]},
        points=False,
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=LABELS.get(variable, variable),
        showlegend=False,
    )
    return apply_theme(fig, height=400)


# --------------------------------------------------------------------------- #
# 5. Correlación
# --------------------------------------------------------------------------- #
def fig_correlacion() -> go.Figure:
    """Mapa de calor de correlaciones (escala divergente, cero en gris neutro)."""
    corr = matriz_correlacion()
    etiquetas = [LABELS.get(c, c) for c in corr.columns]

    fig = px.imshow(
        corr.values,
        x=etiquetas,
        y=etiquetas,
        zmin=-1,
        zmax=1,
        color_continuous_scale=DIV_BLUE_CORAL,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_traces(
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.2f}<extra></extra>",
        xgap=2,
        ygap=2,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="r",
            thickness=12,
            len=0.75,
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        xaxis=dict(side="bottom", tickangle=-30),
    )
    return apply_theme(fig, height=460, margin=dict(l=170, r=24, t=40, b=120))


def fig_correlacion_objetivo() -> go.Figure:
    """Correlación de cada variable numérica con el abandono, ordenada."""
    corr = matriz_correlacion()[TARGET].drop(labels=[TARGET])
    datos = pd.DataFrame(
        {
            "variable": [LABELS.get(i, i) for i in corr.index],
            "r": corr.values,
        }
    ).sort_values("r")
    datos["signo"] = np.where(datos["r"] >= 0, "Aumenta el abandono", "Reduce el abandono")
    datos["etiqueta"] = datos["r"].map(lambda v: f"{v:+.2f}")

    fig = px.bar(
        datos,
        x="r",
        y="variable",
        orientation="h",
        color="signo",
        text="etiqueta",
        color_discrete_map={
            "Aumenta el abandono": COLOR_ABANDONA,
            "Reduce el abandono": COLOR_PERMANECE,
        },
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, color=INK_SOFT),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>r = %{x:.2f}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Correlación de Pearson con el abandono",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS),
        legend_title_text="",
    )
    return apply_theme(_redondear_barras(fig), height=360, bargap=0.35)


# --------------------------------------------------------------------------- #
# 6. Desempeño del modelo
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
            line=dict(color=COLOR_PERMANECE, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(78,140,217,0.12)",
            hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
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
    x = ["Predice: permanece", "Predice: abandono"]
    y = ["Real: permanece", "Real: abandono"]

    fig = px.imshow(
        cm,
        x=x,
        y=y,
        color_continuous_scale=SEQ_BLUE,
        text_auto=True,
        aspect="auto",
    )
    fig.update_traces(
        textfont=dict(size=20),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Casos: %{z}<extra></extra>",
        xgap=3,
        ygap=3,
    )
    fig.update_layout(coloraxis_showscale=False, xaxis=dict(side="bottom"))
    return apply_theme(fig, height=380, margin=dict(l=130, r=24, t=40, b=60))


def fig_importancias(top_n: int = 10) -> go.Figure:
    """Efecto de cada variable sobre las probabilidades de abandono (odds ratio)."""
    importancias = load_metrics()["importancias"][:top_n]
    datos = pd.DataFrame(importancias)
    datos["variable_legible"] = datos["variable"].map(
        lambda v: LABELS.get(v, v.replace("departamento_", "Depto: ").replace("_", " "))
    )
    datos["efecto"] = np.where(
        datos["coeficiente"] >= 0, "Aumenta el riesgo", "Reduce el riesgo"
    )
    datos["etiqueta"] = datos["odds_ratio"].map(lambda v: f"OR {v:.2f}")
    datos = datos.sort_values("coeficiente")

    fig = px.bar(
        datos,
        x="coeficiente",
        y="variable_legible",
        orientation="h",
        color="efecto",
        text="etiqueta",
        color_discrete_map={
            "Aumenta el riesgo": COLOR_ABANDONA,
            "Reduce el riesgo": COLOR_PERMANECE,
        },
        custom_data=["odds_ratio"],
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color=INK_SOFT),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Coeficiente: %{x:.3f}"
            "<br>Odds ratio: %{customdata[0]:.3f}<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="Coeficiente de la regresión logística (escala estandarizada)",
        yaxis_title=None,
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS),
        legend_title_text="",
    )
    return apply_theme(_redondear_barras(fig), height=440, bargap=0.3,
                       margin=dict(l=190, r=70, t=56, b=52))


# --------------------------------------------------------------------------- #
# 7. Predicción individual
# --------------------------------------------------------------------------- #
def fig_gauge_riesgo(probabilidad: float) -> go.Figure:
    """Indicador de aguja con la probabilidad de abandono de un empleado.

    Las bandas usan colores de estado (nunca colores de serie) y siempre van
    acompañadas de una etiqueta textual en la pestaña, de modo que el nivel de
    riesgo no se comunica solo con color.
    """
    valor = probabilidad * 100
    if probabilidad >= RIESGO_ALTO:
        color_aguja = STATUS_CRITICAL
    elif probabilidad >= RIESGO_MEDIO:
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
                    {"range": [0, RIESGO_MEDIO * 100], "color": "#E7F5F0"},
                    {"range": [RIESGO_MEDIO * 100, RIESGO_ALTO * 100], "color": "#FDF3DF"},
                    {"range": [RIESGO_ALTO * 100, 100], "color": "#FBE6E6"},
                ],
                "threshold": {
                    "line": {"color": INK_SOFT, "width": 2},
                    "thickness": 0.8,
                    "value": RIESGO_ALTO * 100,
                },
            },
        )
    )
    return apply_theme(fig, height=280, margin=dict(l=30, r=30, t=30, b=10))


def fig_comparacion_perfil(registro: dict) -> go.Figure:
    """Compara el perfil evaluado con el promedio de quienes se quedan y se van.

    Se normaliza cada variable a percentil (0-100) dentro del dataset para que
    magnitudes distintas (salario en millones, satisfacción de 1 a 5) puedan
    vivir en un mismo eje.
    """
    df = get_dataframe()
    variables = NUMERIC_FEATURES
    etiquetas = [LABELS.get(v, v) for v in variables]

    def percentil(variable: str, valor: float) -> float:
        serie = df[variable]
        return float((serie <= valor).mean() * 100)

    perfil = [percentil(v, registro[v]) for v in variables]
    permanece = [percentil(v, df.loc[df[TARGET] == 0, v].mean()) for v in variables]
    abandona = [percentil(v, df.loc[df[TARGET] == 1, v].mean()) for v in variables]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=etiquetas,
            x=permanece,
            orientation="h",
            name="Promedio: permanece",
            marker_color=COLOR_PERMANECE,
            opacity=0.55,
            hovertemplate="%{y}<br>Percentil %{x:.0f}<extra>Permanece</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=etiquetas,
            x=abandona,
            orientation="h",
            name="Promedio: abandona",
            marker_color=COLOR_ABANDONA,
            opacity=0.55,
            hovertemplate="%{y}<br>Percentil %{x:.0f}<extra>Abandona</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=etiquetas,
            x=perfil,
            mode="markers",
            name="Empleado evaluado",
            marker=dict(
                size=13,
                color=INK,
                symbol="diamond",
                line=dict(color=BG_CARD, width=2),  # anillo de 2px sobre las barras
            ),
            hovertemplate="%{y}<br>Percentil %{x:.0f}<extra>Evaluado</extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Posición dentro de la plantilla (percentil)",
        yaxis_title=None,
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor=GRID),
    )
    return apply_theme(_redondear_barras(fig), height=420, bargap=0.28,
                       margin=dict(l=190, r=24, t=56, b=52))
