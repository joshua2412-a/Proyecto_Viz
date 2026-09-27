"""
Pestaña 7 · Resultados
Dos bloques: análisis exploratorio de los datos (EDA) y evaluación del modelo.

Contiene un callback propio para el explorador de distribuciones. El callback
se registra con el decorador global `@callback`, de modo que la pestaña no
necesita recibir la instancia de la app: sigue estando desacoplada de app.py.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    graph_card,
    kpi_row,
    page_header,
    paragraph,
    section_title,
    series_chips,
)
from utils.config import LABELS, NUMERIC_FEATURES
from utils.data_loader import get_dataframe, load_metrics
from utils.figures import (
    fig_abandono_departamento,
    fig_boxplot,
    fig_correlacion,
    fig_correlacion_objetivo,
    fig_dona_abandono,
    fig_histograma,
    fig_importancias,
    fig_matriz_confusion,
    fig_roc,
)
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# Identificadores de los controles de esta pestaña (prefijo para evitar choques)
ID_VARIABLE = "res-hist-variable"
ID_TIPO = "res-hist-tipo"
ID_GRAFICO = "res-hist-grafico"


# --------------------------------------------------------------------------- #
# Bloque 1 · Análisis exploratorio
# --------------------------------------------------------------------------- #
def _kpis_eda() -> list[dict]:
    """Indicadores descriptivos de la plantilla."""
    df = get_dataframe()
    sale = df[df["abandono"] == 1]
    queda = df[df["abandono"] == 0]
    return [
        {
            "valor": f"{df['satisfaccion'].mean():.2f}",
            "etiqueta": "Satisfacción promedio",
            "detalle": f"Abandona {sale['satisfaccion'].mean():.2f} · "
            f"Permanece {queda['satisfaccion'].mean():.2f}",
            "color": SERIES[0],
        },
        {
            "valor": f"{df['horas_trabajadas'].mean():.1f} h",
            "etiqueta": "Horas semanales promedio",
            "detalle": f"Abandona {sale['horas_trabajadas'].mean():.1f} h · "
            f"Permanece {queda['horas_trabajadas'].mean():.1f} h",
            "color": SERIES[1],
        },
        {
            "valor": f"${df['salario'].mean() / 1_000_000:,.2f}M",
            "etiqueta": "Salario promedio",
            "detalle": f"Abandona ${sale['salario'].mean() / 1_000_000:,.2f}M · "
            f"Permanece ${queda['salario'].mean() / 1_000_000:,.2f}M",
            "color": SERIES[2],
        },
        {
            "valor": f"{df['anios_empresa'].mean():.1f} años",
            "etiqueta": "Antigüedad promedio",
            "detalle": f"Abandona {sale['anios_empresa'].mean():.1f} · "
            f"Permanece {queda['anios_empresa'].mean():.1f}",
            "color": SERIES[3],
        },
    ]


def _tabla_descriptivos() -> dbc.Table:
    """Estadísticos descriptivos de las variables numéricas por grupo."""
    df = get_dataframe()
    filas = []
    for variable in NUMERIC_FEATURES:
        sale = df.loc[df["abandono"] == 1, variable]
        queda = df.loc[df["abandono"] == 0, variable]
        formato = "{:,.0f}" if variable == "salario" else "{:,.2f}"
        filas.append(
            [
                LABELS[variable],
                formato.format(df[variable].mean()),
                formato.format(df[variable].std()),
                formato.format(df[variable].min()),
                formato.format(df[variable].max()),
                formato.format(queda.mean()),
                formato.format(sale.mean()),
            ]
        )
    return data_table(
        [
            "Variable",
            "Media",
            "Desv. est.",
            "Mínimo",
            "Máximo",
            "Media · permanece",
            "Media · abandona",
        ],
        filas,
    )


def _controles_distribucion() -> dbc.Row:
    """Selector de variable y tipo de gráfico para el explorador."""
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Variable a explorar", className="control-label"),
                    dcc.Dropdown(
                        id=ID_VARIABLE,
                        options=[
                            {"label": LABELS[v], "value": v} for v in NUMERIC_FEATURES
                        ],
                        value="satisfaccion",
                        clearable=False,
                        className="control-dropdown",
                    ),
                ],
                md=6,
            ),
            dbc.Col(
                [
                    dbc.Label("Tipo de gráfico", className="control-label"),
                    dbc.RadioItems(
                        id=ID_TIPO,
                        options=[
                            {"label": "Histograma", "value": "histograma"},
                            {"label": "Diagrama de caja", "value": "caja"},
                        ],
                        value="histograma",
                        inline=True,
                        className="control-radio",
                    ),
                ],
                md=6,
            ),
        ],
        className="controls-row g-3",
    )


def _bloque_eda() -> html.Div:
    """Contenido del bloque de análisis exploratorio."""
    return html.Div(
        [
            kpi_row(_kpis_eda()),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_dona_abandono(),
                            "Composición de la plantilla",
                            "Proporción de empleados que abandonan frente a los que permanecen.",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_abandono_departamento(),
                            "Tasa de abandono por departamento",
                            "La línea discontinua marca la media global.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Explorador de distribuciones"),
            card(
                [
                    series_chips(),
                    _controles_distribucion(),
                    dcc.Graph(
                        id=ID_GRAFICO,
                        figure=fig_histograma("satisfaccion"),
                        config={"displaylogo": False, "responsive": True},
                        className="graph",
                    ),
                    html.Div(
                        "Los histogramas se normalizan a porcentaje dentro de cada grupo "
                        "porque las clases están desbalanceadas; las líneas punteadas "
                        "señalan la media de cada grupo.",
                        className="graph-note",
                    ),
                ],
                titulo="Distribución por grupo de abandono",
                color=COLOR_PERMANECE,
                className="mb-4",
            ),
            section_title("Estructura de correlaciones"),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_correlacion(),
                            "Matriz de correlación",
                            "Coeficiente de Pearson. El gris del centro de la escala "
                            "representa la ausencia de relación lineal.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_correlacion_objetivo(),
                            "Correlación de cada variable con el abandono",
                            "Valores positivos acompañan al abandono; negativos, a la permanencia.",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Estadísticos descriptivos"),
            card(
                _tabla_descriptivos(),
                subtitulo="Comparación de la media de cada variable entre los dos grupos.",
                color=SERIES[3],
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Bloque 2 · Desempeño del modelo
# --------------------------------------------------------------------------- #
def _kpis_modelo() -> list[dict]:
    """Métricas principales del clasificador."""
    m = load_metrics()
    return [
        {
            "valor": f"{m['accuracy']:.1%}",
            "etiqueta": "Accuracy",
            "detalle": "Aciertos sobre el total",
            "color": SERIES[0],
        },
        {
            "valor": f"{m['precision']:.1%}",
            "etiqueta": "Precisión",
            "detalle": "Aciertos entre los casos señalados",
            "color": SERIES[3],
        },
        {
            "valor": f"{m['recall']:.1%}",
            "etiqueta": "Recall",
            "detalle": "Abandonos detectados",
            "color": SERIES[1],
        },
        {
            "valor": f"{m['f1']:.1%}",
            "etiqueta": "F1-score",
            "detalle": "Equilibrio precisión / recall",
            "color": SERIES[2],
        },
    ]


def _tabla_confusion() -> dbc.Table:
    """Lectura en palabras de las cuatro celdas de la matriz de confusión."""
    cm = load_metrics()["matriz_confusion"]
    vn, fp = cm[0][0], cm[0][1]
    fn, vp = cm[1][0], cm[1][1]
    filas = [
        [
            "Verdaderos negativos",
            f"{vn}",
            "Permanece y el modelo acierta",
            "Sin acción requerida",
        ],
        [
            "Falsos positivos",
            f"{fp}",
            "Permanece pero el modelo lo marca como riesgo",
            "Coste bajo: una conversación de más",
        ],
        [
            "Falsos negativos",
            f"{fn}",
            "Abandona y el modelo no lo detecta",
            "Coste alto: salida no anticipada",
        ],
        [
            "Verdaderos positivos",
            f"{vp}",
            "Abandona y el modelo lo detecta",
            "Caso donde el modelo aporta valor",
        ],
    ]
    return data_table(
        ["Celda", "Casos", "Interpretación", "Consecuencia operativa"], filas
    )


def _bloque_modelo() -> html.Div:
    """Contenido del bloque de evaluación del modelo."""
    m = load_metrics()
    return html.Div(
        [
            kpi_row(_kpis_modelo()),
            callout(
                f"El modelo alcanza un AUC de {m['roc_auc']:.3f} en el conjunto de prueba "
                f"y {m['cv_auc_media']:.3f} ± {m['cv_auc_desviacion']:.3f} en validación "
                "cruzada. La cercanía entre ambos valores indica que no hay sobreajuste "
                "relevante.",
                titulo="Lectura general",
                color=COLOR_PERMANECE,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_roc(),
                            "Curva ROC",
                            "Cuanto más se aleja la curva de la diagonal, mejor separa el "
                            "modelo a quienes abandonan de quienes permanecen.",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_matriz_confusion(),
                            "Matriz de confusión",
                            f"Conjunto de prueba: {m['n_test']} empleados no vistos "
                            "durante el entrenamiento.",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                ]
            ),
            card(
                _tabla_confusion(),
                titulo="Los cuatro tipos de resultado, en palabras",
                subtitulo="El modelo se calibró para reducir falsos negativos: en "
                "retención, no detectar una salida cuesta más que una conversación "
                "innecesaria.",
                color=COLOR_ABANDONA,
                className="mb-4",
            ),
            section_title("Factores asociados al abandono"),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_importancias(),
                            "Efecto de cada variable sobre el riesgo",
                            "Coeficientes de la regresión logística sobre variables "
                            "estandarizadas; OR = razón de odds.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Los coeficientes se calculan sobre variables "
                                    "estandarizadas, así que su magnitud es directamente "
                                    "comparable entre sí: un salto de una desviación "
                                    "estándar en cada variable produce el cambio indicado "
                                    "en el logaritmo de las odds."
                                ),
                                bullet_list(
                                    [
                                        "OR > 1 → la variable aumenta las probabilidades de abandono.",
                                        "OR < 1 → la variable las reduce.",
                                        "OR ≈ 1 → sin efecto apreciable, manteniendo el resto constante.",
                                    ],
                                    color=COLOR_ABANDONA,
                                ),
                                callout(
                                    "Estos coeficientes describen asociación, no causalidad. "
                                    "Señalan dónde mirar primero, no qué palanca accionar sin "
                                    "más verificación.",
                                    titulo="Advertencia de interpretación",
                                    color=COLOR_ABANDONA,
                                ),
                            ],
                            titulo="Cómo leer los coeficientes",
                            color=COLOR_ABANDONA,
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def layout() -> html.Div:
    """Layout de la pestaña de resultados (EDA + métricas)."""
    return html.Div(
        [
            page_header(
                "Resultados",
                "Primero los datos: qué patrones muestran. Después el modelo: "
                "cuánto acierta y en qué se equivoca.",
                "📊",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(
                        _bloque_eda(),
                        label="Análisis exploratorio",
                        tab_class_name="inner-tab",
                        active_tab_class_name="inner-tab-active",
                    ),
                    dbc.Tab(
                        _bloque_modelo(),
                        label="Desempeño del modelo",
                        tab_class_name="inner-tab",
                        active_tab_class_name="inner-tab-active",
                    ),
                ],
                className="inner-tabs",
            ),
        ],
        className="tab-content",
    )


# --------------------------------------------------------------------------- #
# Callbacks propios de la pestaña
# --------------------------------------------------------------------------- #
@callback(
    Output(ID_GRAFICO, "figure"),
    Input(ID_VARIABLE, "value"),
    Input(ID_TIPO, "value"),
)
def actualizar_distribucion(variable: str, tipo: str):
    """Redibuja el explorador de distribuciones según los controles."""
    if not variable:
        variable = "satisfaccion"
    if tipo == "caja":
        return fig_boxplot(variable)
    return fig_histograma(variable)
