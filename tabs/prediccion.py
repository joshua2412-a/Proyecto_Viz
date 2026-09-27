"""
Pestaña 8 · Predicción
Formulario interactivo que carga model/model.pkl y devuelve la probabilidad de
abandono de un empleado concreto en tiempo real.

El modelo NO se entrena aquí: se carga desde disco a través de
utils.data_loader, que mantiene el pipeline en caché para que la respuesta sea
inmediata.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from utils.config import (
    DEFAULT_INPUT,
    DEPARTAMENTOS,
    INPUT_RANGES,
    LABELS,
    RIESGO_ALTO,
    RIESGO_MEDIO,
)
from utils.components import bullet_list, callout, card, page_header, paragraph, section_title
from utils.data_loader import predecir_abandono, tasa_abandono_global
from utils.figures import fig_comparacion_perfil, fig_gauge_riesgo
from utils.theme import (
    COLOR_ABANDONA,
    COLOR_PERMANECE,
    SERIES,
    STATUS_CRITICAL,
    STATUS_GOOD,
    STATUS_WARNING,
)

# Identificadores de los campos del formulario
ID = {
    "edad": "pred-edad",
    "salario": "pred-salario",
    "anios_empresa": "pred-anios",
    "departamento": "pred-departamento",
    "satisfaccion": "pred-satisfaccion",
    "horas_trabajadas": "pred-horas",
    "promociones": "pred-promociones",
}
ID_BOTON = "pred-boton"
ID_RESET = "pred-reset"
ID_RESULTADO = "pred-resultado"

# Recomendaciones asociadas a cada nivel de riesgo
ACCIONES = {
    "alto": [
        "Programar una conversación de retención en los próximos días.",
        "Revisar la equidad salarial del cargo frente al mercado interno.",
        "Evaluar la carga de trabajo y redistribuirla si excede lo sostenible.",
        "Definir un plan de desarrollo con hitos concretos a seis meses.",
    ],
    "medio": [
        "Incluir a la persona en el seguimiento trimestral de clima.",
        "Verificar que tenga una ruta de promoción visible y realista.",
        "Confirmar que la carga semanal se mantiene dentro del rango previsto.",
    ],
    "bajo": [
        "Mantener el seguimiento habitual de desempeño y clima.",
        "Aprovechar el perfil como referencia de buenas prácticas del área.",
    ],
}


# --------------------------------------------------------------------------- #
# Formulario
# --------------------------------------------------------------------------- #
def _campo_slider(variable: str, formato=None, n_marcas: int = 5) -> html.Div:
    """Campo de formulario con slider para una variable numérica.

    `formato` es una función que convierte el valor de una marca en su texto;
    permite mostrar el salario en millones sin saturar el eje.
    """
    minimo, maximo, paso = INPUT_RANGES[variable]
    if formato is None:
        formato = (lambda v: f"{v:.1f}") if isinstance(paso, float) else (lambda v: f"{v:.0f}")

    marcas = {}
    for i in range(n_marcas + 1):
        valor = minimo + (maximo - minimo) * i / n_marcas
        clave = round(valor, 1) if isinstance(paso, float) else int(valor)
        marcas[clave] = formato(valor)

    return html.Div(
        [
            dbc.Label(LABELS[variable], className="control-label"),
            dcc.Slider(
                id=ID[variable],
                min=minimo,
                max=maximo,
                step=paso,
                value=DEFAULT_INPUT[variable],
                marks=marcas,
                tooltip={"placement": "bottom", "always_visible": True},
                className="form-slider",
            ),
        ],
        className="form-field",
    )


def _formulario() -> dbc.Card:
    """Tarjeta con todos los campos de entrada y los botones de acción."""
    return card(
        [
            html.Div(
                [
                    dbc.Label(LABELS["departamento"], className="control-label"),
                    dcc.Dropdown(
                        id=ID["departamento"],
                        options=[{"label": d, "value": d} for d in DEPARTAMENTOS],
                        value=DEFAULT_INPUT["departamento"],
                        clearable=False,
                        className="control-dropdown",
                    ),
                ],
                className="form-field",
            ),
            _campo_slider("edad"),
            # El salario se etiqueta en millones para que las marcas no se solapen
            _campo_slider("salario", lambda v: f"${v / 1_000_000:.1f}M", n_marcas=4),
            _campo_slider("anios_empresa"),
            _campo_slider("satisfaccion", lambda v: f"{v:.1f}", n_marcas=4),
            _campo_slider("horas_trabajadas"),
            _campo_slider("promociones", n_marcas=6),
            html.Div(
                [
                    dbc.Button(
                        "Calcular riesgo de abandono",
                        id=ID_BOTON,
                        color="primary",
                        className="btn-predecir",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Restablecer",
                        id=ID_RESET,
                        color="light",
                        className="btn-reset",
                        n_clicks=0,
                    ),
                ],
                className="form-actions",
            ),
        ],
        titulo="Perfil del empleado",
        subtitulo="Ajusta los valores y pulsa el botón para obtener la predicción.",
        color=COLOR_PERMANECE,
    )


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
def _clasificar(probabilidad: float) -> tuple[str, str, str, str]:
    """Traduce la probabilidad a nivel, color, icono y lectura textual."""
    if probabilidad >= RIESGO_ALTO:
        return (
            "alto",
            STATUS_CRITICAL,
            "🔴",
            "Riesgo alto: el perfil se parece al de quienes efectivamente abandonaron.",
        )
    if probabilidad >= RIESGO_MEDIO:
        return (
            "medio",
            STATUS_WARNING,
            "🟠",
            "Riesgo moderado: hay señales de alerta que conviene atender pronto.",
        )
    return (
        "bajo",
        STATUS_GOOD,
        "🟢",
        "Riesgo bajo: el perfil se parece al de quienes permanecen en la organización.",
    )


def _panel_resultado(registro: dict) -> html.Div:
    """Construye el panel de resultado a partir de un registro de entrada."""
    probabilidad = predecir_abandono(registro)
    nivel, color, icono, lectura = _clasificar(probabilidad)
    base = tasa_abandono_global()
    veces = probabilidad / base if base else 0

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_gauge_riesgo(probabilidad),
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.Div(
                                    [
                                        html.Span(icono, className="riesgo-icono"),
                                        html.Span(
                                            f"Riesgo {nivel}",
                                            className="riesgo-nivel",
                                            style={"color": color},
                                        ),
                                    ],
                                    className="riesgo-badge",
                                ),
                                html.Div(lectura, className="riesgo-lectura"),
                                html.Div(
                                    f"Es {veces:.1f}× la tasa promedio de la organización "
                                    f"({base:.1%}).",
                                    className="riesgo-comparacion",
                                ),
                            ],
                            titulo="Probabilidad de abandono",
                            color=color,
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_comparacion_perfil(registro),
                                    config={"displaylogo": False, "responsive": True},
                                ),
                                html.Div(
                                    "Cada variable se expresa como percentil dentro de la "
                                    "plantilla para poder compararlas en un mismo eje. El "
                                    "rombo marca el perfil evaluado.",
                                    className="graph-note",
                                ),
                            ],
                            titulo="El perfil frente a los promedios de cada grupo",
                            color=SERIES[3],
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                ]
            ),
            card(
                bullet_list(ACCIONES[nivel], color=color),
                titulo="Acciones sugeridas",
                subtitulo="Recomendaciones asociadas al nivel de riesgo estimado.",
                color=color,
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def layout() -> html.Div:
    """Layout de la pestaña de predicción."""
    return html.Div(
        [
            page_header(
                "Predicción de riesgo individual",
                "Simulador en tiempo real: describe a un empleado y el modelo "
                "estima su probabilidad de abandono.",
                "🔮",
            ),
            callout(
                "La predicción se calcula con el pipeline guardado en "
                "model/model.pkl. El archivo incluye el escalado y la codificación "
                "de categorías, por lo que el formulario envía los valores tal como "
                "se introducen, sin transformaciones intermedias.",
                titulo="Cómo funciona",
                color=COLOR_PERMANECE,
            ),
            dbc.Row(
                [
                    dbc.Col(_formulario(), lg=4, className="mb-4"),
                    dbc.Col(
                        html.Div(_panel_resultado(DEFAULT_INPUT), id=ID_RESULTADO),
                        lg=8,
                    ),
                ]
            ),
            section_title("Umbrales de interpretación"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(texto),
                            titulo=f"{icono}  {titulo}",
                            color=color,
                        ),
                        lg=4,
                        className="mb-3",
                    )
                    for icono, titulo, texto, color in [
                        (
                            "🟢",
                            f"Riesgo bajo · menos de {RIESGO_MEDIO:.0%}",
                            "Seguimiento habitual. El perfil no presenta señales "
                            "distintivas de salida.",
                            STATUS_GOOD,
                        ),
                        (
                            "🟠",
                            f"Riesgo medio · {RIESGO_MEDIO:.0%} a {RIESGO_ALTO:.0%}",
                            "Atención preventiva. Conviene revisar carga de trabajo y "
                            "expectativas de desarrollo.",
                            STATUS_WARNING,
                        ),
                        (
                            "🔴",
                            f"Riesgo alto · más de {RIESGO_ALTO:.0%}",
                            "Intervención prioritaria. El perfil coincide con el patrón "
                            "de quienes abandonaron.",
                            STATUS_CRITICAL,
                        ),
                    ]
                ],
                className="g-3",
            ),
            callout(
                "El resultado es una probabilidad estimada, no un diagnóstico. Debe "
                "usarse para priorizar conversaciones, nunca como criterio único de "
                "una decisión laboral sobre una persona.",
                titulo="Uso responsable",
                color=COLOR_ABANDONA,
            ),
        ],
        className="tab-content",
    )


# --------------------------------------------------------------------------- #
# Callbacks propios de la pestaña
# --------------------------------------------------------------------------- #
@callback(
    Output(ID_RESULTADO, "children"),
    Input(ID_BOTON, "n_clicks"),
    State(ID["edad"], "value"),
    State(ID["salario"], "value"),
    State(ID["anios_empresa"], "value"),
    State(ID["departamento"], "value"),
    State(ID["satisfaccion"], "value"),
    State(ID["horas_trabajadas"], "value"),
    State(ID["promociones"], "value"),
    prevent_initial_call=True,
)
def calcular_prediccion(
    n_clicks,
    edad,
    salario,
    anios_empresa,
    departamento,
    satisfaccion,
    horas_trabajadas,
    promociones,
):
    """Construye el registro, consulta el modelo y devuelve el panel de resultado."""
    registro = {
        "edad": edad if edad is not None else DEFAULT_INPUT["edad"],
        "salario": salario if salario is not None else DEFAULT_INPUT["salario"],
        "anios_empresa": anios_empresa
        if anios_empresa is not None
        else DEFAULT_INPUT["anios_empresa"],
        "departamento": departamento or DEFAULT_INPUT["departamento"],
        "satisfaccion": satisfaccion
        if satisfaccion is not None
        else DEFAULT_INPUT["satisfaccion"],
        "horas_trabajadas": horas_trabajadas
        if horas_trabajadas is not None
        else DEFAULT_INPUT["horas_trabajadas"],
        "promociones": promociones
        if promociones is not None
        else DEFAULT_INPUT["promociones"],
    }
    return _panel_resultado(registro)


@callback(
    Output(ID["edad"], "value"),
    Output(ID["salario"], "value"),
    Output(ID["anios_empresa"], "value"),
    Output(ID["departamento"], "value"),
    Output(ID["satisfaccion"], "value"),
    Output(ID["horas_trabajadas"], "value"),
    Output(ID["promociones"], "value"),
    Input(ID_RESET, "n_clicks"),
    prevent_initial_call=True,
)
def restablecer_formulario(n_clicks):
    """Devuelve el formulario a los valores del perfil promedio."""
    return (
        DEFAULT_INPUT["edad"],
        DEFAULT_INPUT["salario"],
        DEFAULT_INPUT["anios_empresa"],
        DEFAULT_INPUT["departamento"],
        DEFAULT_INPUT["satisfaccion"],
        DEFAULT_INPUT["horas_trabajadas"],
        DEFAULT_INPUT["promociones"],
    )
