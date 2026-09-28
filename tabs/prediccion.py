"""
Pestaña 8 · Predicción
Formulario interactivo que carga model/model.pkl y devuelve la probabilidad
estimada de glioblastoma para un perfil clínico-molecular concreto.

El modelo NO se entrena aquí: se carga desde disco a través de
utils.data_loader, que mantiene el pipeline en caché para que la respuesta sea
inmediata.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from utils.config import (
    DEFAULT_INPUT,
    GENDER_LABELS,
    GENE_DESCRIPCION,
    GENE_FEATURES,
    GENES_DESTACADOS,
    INPUT_RANGES,
    LABELS,
    RACE_LABELS,
    UMBRAL_GBM,
    UMBRAL_LGG,
)
from utils.components import (
    bullet_list,
    callout,
    card,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import predecir_gbm
from utils.figures import fig_contribuciones, fig_edad_vs_probabilidad, fig_gauge_probabilidad
from utils.theme import (
    COLOR_GBM,
    COLOR_LGG,
    SERIES,
    STATUS_CRITICAL,
    STATUS_GOOD,
    STATUS_WARNING,
)

# Identificadores de los campos del formulario
ID_EDAD = "pred-edad"
ID_GENERO = "pred-genero"
ID_RAZA = "pred-raza"
ID_GENES_CLAVE = "pred-genes-clave"
ID_GENES_RESTO = "pred-genes-resto"
ID_BOTON = "pred-boton"
ID_RESET = "pred-reset"
ID_RESULTADO = "pred-resultado"

GENES_RESTANTES = [gen for gen in GENE_FEATURES if gen not in GENES_DESTACADOS]

# Lectura clínica asociada a cada banda de probabilidad
LECTURAS = {
    "lgg": [
        "El perfil es compatible con un glioma de bajo grado.",
        "Confirmar con histología e imagen según el protocolo del centro.",
        "La secuenciación ampliada aporta poco valor añadido para decidir el grado.",
    ],
    "incierto": [
        "El perfil cae en la zona de incertidumbre del modelo.",
        "Es el caso en el que la secuenciación completa sí cambia la decisión.",
        "No usar la estimación como criterio único: pedir confirmación molecular.",
    ],
    "gbm": [
        "El perfil es compatible con un glioblastoma multiforme.",
        "Priorizar la confirmación diagnóstica y la planificación del tratamiento.",
        "Revisar la edad y las mutaciones que más pesan en el desglose de abajo.",
    ],
}


# --------------------------------------------------------------------------- #
# Formulario
# --------------------------------------------------------------------------- #
def _campo_edad() -> html.Div:
    """Slider de edad al diagnóstico."""
    minimo, maximo, paso = INPUT_RANGES["Age_at_diagnosis"]
    marcas = {int(valor): f"{int(valor)}" for valor in range(int(minimo), int(maximo) + 1, 15)}

    return html.Div(
        [
            dbc.Label(LABELS["Age_at_diagnosis"], className="control-label"),
            dcc.Slider(
                id=ID_EDAD,
                min=minimo,
                max=maximo,
                step=paso,
                value=DEFAULT_INPUT["Age_at_diagnosis"],
                marks=marcas,
                tooltip={"placement": "bottom", "always_visible": True},
                className="form-slider",
            ),
        ],
        className="form-field",
    )


def _opciones_genes(genes: list[str]) -> list[dict]:
    """Opciones de checklist con la función biológica de cada gen en el tooltip."""
    return [
        {
            "label": html.Span(
                gen,
                title=GENE_DESCRIPCION.get(gen, ""),
                className="gene-label",
            ),
            "value": gen,
        }
        for gen in genes
    ]


def _formulario() -> dbc.Card:
    """Tarjeta con todos los campos de entrada y los botones de acción."""
    return card(
        [
            _campo_edad(),
            html.Div(
                [
                    dbc.Label(LABELS["Gender"], className="control-label"),
                    dcc.Dropdown(
                        id=ID_GENERO,
                        options=[
                            {"label": etiqueta, "value": clave}
                            for clave, etiqueta in GENDER_LABELS.items()
                        ],
                        value=DEFAULT_INPUT["Gender"],
                        clearable=False,
                        className="control-dropdown",
                    ),
                ],
                className="form-field",
            ),
            html.Div(
                [
                    dbc.Label(LABELS["Race"], className="control-label"),
                    dcc.Dropdown(
                        id=ID_RAZA,
                        options=[
                            {"label": etiqueta, "value": clave}
                            for clave, etiqueta in RACE_LABELS.items()
                        ],
                        value=DEFAULT_INPUT["Race"],
                        clearable=False,
                        className="control-dropdown",
                    ),
                ],
                className="form-field",
            ),
            html.Div(
                [
                    dbc.Label(
                        "Mutaciones con señal en el grado", className="control-label"
                    ),
                    html.Div(
                        "Marca los genes mutados. Estos ocho son los que el EDA "
                        "encontró asociados al grado tumoral.",
                        className="control-hint",
                    ),
                    dbc.Checklist(
                        id=ID_GENES_CLAVE,
                        options=_opciones_genes(GENES_DESTACADOS),
                        value=[],
                        inline=True,
                        className="gene-checklist",
                    ),
                ],
                className="form-field",
            ),
            dbc.Accordion(
                dbc.AccordionItem(
                    [
                        html.Div(
                            "El modelo deja en cero el coeficiente de casi todas "
                            "estas, así que marcarlas no suele cambiar la "
                            "predicción. Están aquí para completar el panel.",
                            className="control-hint",
                        ),
                        dbc.Checklist(
                            id=ID_GENES_RESTO,
                            options=_opciones_genes(GENES_RESTANTES),
                            value=[],
                            inline=True,
                            className="gene-checklist",
                        ),
                    ],
                    title=f"Resto del panel ({len(GENES_RESTANTES)} genes)",
                ),
                start_collapsed=True,
                flush=True,
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Button(
                        "Estimar probabilidad de GBM",
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
        titulo="Perfil del paciente",
        subtitulo="Ajusta los valores y pulsa el botón para obtener la estimación.",
        color=COLOR_LGG,
    )


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
def _clasificar(probabilidad: float) -> tuple[str, str, str, str]:
    """Traduce la probabilidad a banda, color, icono y veredicto textual."""
    if probabilidad >= UMBRAL_GBM:
        return (
            "gbm",
            STATUS_CRITICAL,
            "🔴",
            "Compatible con glioblastoma multiforme (GBM)",
        )
    if probabilidad >= UMBRAL_LGG:
        return (
            "incierto",
            STATUS_WARNING,
            "🟠",
            "Zona de incertidumbre: el modelo no decide con confianza",
        )
    return (
        "lgg",
        STATUS_GOOD,
        "🟢",
        "Compatible con glioma de bajo grado (LGG)",
    )


def _registro(edad, genero, raza, genes_marcados: list[str]) -> dict:
    """Construye el registro que espera el pipeline a partir del formulario."""
    return {
        "Age_at_diagnosis": float(edad) if edad is not None
        else DEFAULT_INPUT["Age_at_diagnosis"],
        "Gender": int(genero) if genero is not None else DEFAULT_INPUT["Gender"],
        "Race": int(raza) if raza is not None else DEFAULT_INPUT["Race"],
        **{gen: int(gen in genes_marcados) for gen in GENE_FEATURES},
    }


def _panel_resultado(registro: dict) -> html.Div:
    """Construye el panel de resultado a partir de un registro de entrada."""
    probabilidad = predecir_gbm(registro)
    banda, color, icono, veredicto = _clasificar(probabilidad)
    mutados = [gen for gen in GENE_FEATURES if registro[gen] == 1]

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_gauge_probabilidad(probabilidad),
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                html.Div(
                                    [
                                        html.Span(icono, className="riesgo-icono"),
                                        html.Span(
                                            veredicto,
                                            className="riesgo-nivel",
                                            style={"color": color},
                                        ),
                                    ],
                                    className="riesgo-badge",
                                ),
                                html.Div(
                                    f"Perfil evaluado: {registro['Age_at_diagnosis']:.1f} años · "
                                    f"{GENDER_LABELS[registro['Gender']]} · "
                                    f"{RACE_LABELS[registro['Race']]}",
                                    className="riesgo-lectura",
                                ),
                                html.Div(
                                    "Mutaciones marcadas: "
                                    + (", ".join(mutados) if mutados else "ninguna"),
                                    className="riesgo-comparacion",
                                ),
                            ],
                            titulo="Probabilidad estimada de GBM",
                            color=color,
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_contribuciones(registro),
                                    config={"displaylogo": False, "responsive": True},
                                ),
                                html.Div(
                                    "En un modelo lineal la predicción es la suma de estas "
                                    "contribuciones más el intercepto, así que el desglose "
                                    "no aproxima el razonamiento del modelo: es el "
                                    "razonamiento del modelo.",
                                    className="graph-note",
                                ),
                            ],
                            titulo="Qué pesa en esta estimación",
                            color=SERIES[3],
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                dcc.Graph(
                                    figure=fig_edad_vs_probabilidad(registro),
                                    config={"displaylogo": False, "responsive": True},
                                ),
                                html.Div(
                                    "Se mueve solo la edad y se mantiene fijo el resto del "
                                    "perfil: es la pendiente de la edad para este perfil "
                                    "genético concreto, no una curva promedio.",
                                    className="graph-note",
                                ),
                            ],
                            titulo="Cómo cambiaría la estimación con la edad",
                            color=SERIES[2],
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            bullet_list(LECTURAS[banda], color=color),
                            titulo="Lectura sugerida",
                            subtitulo="Qué implica esta banda de probabilidad.",
                            color=color,
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
    """Layout de la pestaña de predicción."""
    return html.Div(
        [
            page_header(
                "Estimación del grado tumoral",
                "Simulador en tiempo real: describe el perfil clínico y molecular de "
                "un paciente y el modelo estima su probabilidad de glioblastoma.",
                "🔮",
            ),
            callout(
                "La estimación se calcula con el pipeline guardado en model/model.pkl. El "
                "archivo incluye el escalado de la edad y la codificación de las variables "
                "categóricas, por lo que el formulario envía los valores tal como se "
                "introducen, sin transformaciones intermedias.",
                titulo="Cómo funciona",
                color=COLOR_LGG,
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
            section_title("Cómo leer la probabilidad"),
            dbc.Row(
                [
                    dbc.Col(
                        card(paragraph(texto), titulo=f"{icono}  {titulo}", color=color),
                        lg=4,
                        className="mb-3",
                    )
                    for icono, titulo, texto, color in [
                        (
                            "🟢",
                            f"Menos de {UMBRAL_LGG:.0%}",
                            "Perfil compatible con LGG. El modelo clasifica con holgura "
                            "por debajo de su umbral de decisión.",
                            STATUS_GOOD,
                        ),
                        (
                            "🟠",
                            f"Entre {UMBRAL_LGG:.0%} y {UMBRAL_GBM:.0%}",
                            "Zona de incertidumbre. El clasificador decide en 0,50, pero "
                            "aquí la decisión es frágil: es donde la secuenciación "
                            "completa aporta más.",
                            STATUS_WARNING,
                        ),
                        (
                            "🔴",
                            f"Más de {UMBRAL_GBM:.0%}",
                            "Perfil compatible con GBM. Conviene priorizar la confirmación "
                            "diagnóstica.",
                            STATUS_CRITICAL,
                        ),
                    ]
                ],
                className="g-3",
            ),
            callout(
                "El resultado es una probabilidad estimada por un modelo entrenado con 671 "
                "pacientes, no un diagnóstico. No sustituye la histología, la imagen ni el "
                "criterio médico, y no debe usarse como criterio único de una decisión "
                "clínica sobre una persona.",
                titulo="Uso responsable",
                color=COLOR_GBM,
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
    State(ID_EDAD, "value"),
    State(ID_GENERO, "value"),
    State(ID_RAZA, "value"),
    State(ID_GENES_CLAVE, "value"),
    State(ID_GENES_RESTO, "value"),
    prevent_initial_call=True,
)
def calcular_prediccion(n_clicks, edad, genero, raza, genes_clave, genes_resto):
    """Construye el registro, consulta el modelo y devuelve el panel de resultado."""
    marcados = [*(genes_clave or []), *(genes_resto or [])]
    return _panel_resultado(_registro(edad, genero, raza, marcados))


@callback(
    Output(ID_EDAD, "value"),
    Output(ID_GENERO, "value"),
    Output(ID_RAZA, "value"),
    Output(ID_GENES_CLAVE, "value"),
    Output(ID_GENES_RESTO, "value"),
    Input(ID_RESET, "n_clicks"),
    prevent_initial_call=True,
)
def restablecer_formulario(n_clicks):
    """Devuelve el formulario al perfil por defecto (edad mediana, sin mutaciones)."""
    return (
        DEFAULT_INPUT["Age_at_diagnosis"],
        DEFAULT_INPUT["Gender"],
        DEFAULT_INPUT["Race"],
        [],
        [],
    )
