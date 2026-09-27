"""
Pestaña 3 · Planteamiento del problema
Localiza el abandono dentro de la organización: tasa por departamento,
brechas entre áreas y formulación de las preguntas de investigación.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

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
)
from utils.data_loader import get_dataframe, tasa_abandono_global, tasa_por_departamento
from utils.figures import fig_abandono_departamento
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# Preguntas que estructuran el resto del tablero
PREGUNTAS = [
    "¿Qué departamentos concentran la mayor pérdida de talento y con qué intensidad?",
    "¿Qué variables observables se asocian de forma sistemática con la decisión de salir?",
    "¿Es posible estimar la probabilidad de abandono de un empleado concreto con datos "
    "que la empresa ya registra?",
    "¿Qué nivel de acierto alcanza ese modelo y qué errores comete?",
]


def _kpis_problema() -> list[dict]:
    """Indicadores que dimensionan la brecha entre áreas."""
    resumen = tasa_por_departamento()
    global_ = tasa_abandono_global()
    peor = resumen.iloc[0]
    mejor = resumen.iloc[-1]
    brecha = peor["tasa"] / mejor["tasa"] if mejor["tasa"] else 0

    return [
        {
            "valor": f"{global_:.1%}",
            "etiqueta": "Tasa global de abandono",
            "detalle": "Referencia para comparar áreas",
            "color": SERIES[0],
        },
        {
            "valor": f"{peor['tasa']:.1%}",
            "etiqueta": f"Máximo · {peor['departamento']}",
            "detalle": f"{int(peor['abandonos'])} salidas de {int(peor['empleados'])}",
            "color": SERIES[1],
        },
        {
            "valor": f"{mejor['tasa']:.1%}",
            "etiqueta": f"Mínimo · {mejor['departamento']}",
            "detalle": f"{int(mejor['abandonos'])} salidas de {int(mejor['empleados'])}",
            "color": SERIES[2],
        },
        {
            "valor": f"{brecha:.1f}×",
            "etiqueta": "Brecha entre extremos",
            "detalle": "Cuántas veces más rota el área crítica",
            "color": SERIES[3],
        },
    ]


def _tabla_detalle() -> dbc.Table:
    """Tabla con la desviación de cada área respecto a la media global."""
    resumen = tasa_por_departamento()
    global_ = tasa_abandono_global()

    filas = []
    for _, fila in resumen.iterrows():
        desviacion = (fila["tasa"] - global_) * 100
        signo = "▲" if desviacion > 0 else "▼"
        estado = "Por encima de la media" if desviacion > 0 else "Por debajo de la media"
        filas.append(
            [
                fila["departamento"],
                f"{int(fila['empleados']):,}",
                f"{int(fila['abandonos']):,}",
                f"{fila['tasa']:.1%}",
                f"{signo} {abs(desviacion):.1f} p.p.",
                estado,
            ]
        )
    return data_table(
        [
            "Departamento",
            "Empleados",
            "Salidas",
            "Tasa",
            "Desviación vs media",
            "Lectura",
        ],
        filas,
    )


def layout() -> html.Div:
    """Layout de la pestaña de planteamiento del problema."""
    resumen = tasa_por_departamento()
    peor = resumen.iloc[0]
    mejor = resumen.iloc[-1]
    df = get_dataframe()
    satisfaccion_sale = df.loc[df["abandono"] == 1, "satisfaccion"].mean()
    satisfaccion_queda = df.loc[df["abandono"] == 0, "satisfaccion"].mean()

    return html.Div(
        [
            page_header(
                "Planteamiento del problema",
                "El abandono no está distribuido de forma uniforme: se concentra "
                "en áreas concretas y responde a patrones medibles.",
                "❗",
            ),
            kpi_row(_kpis_problema()),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_abandono_departamento(),
                            "Tasa de abandono por departamento",
                            "La línea discontinua marca la media global de la organización.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    f"La diferencia entre {peor['departamento']} "
                                    f"({peor['tasa']:.1%}) y {mejor['departamento']} "
                                    f"({mejor['tasa']:.1%}) no se explica por azar: son áreas "
                                    "con cargas de trabajo, expectativas salariales y "
                                    "trayectorias de promoción muy distintas."
                                ),
                                paragraph(
                                    f"En paralelo, quienes abandonan declaran una satisfacción "
                                    f"promedio de {satisfaccion_sale:.2f} sobre 5, frente a "
                                    f"{satisfaccion_queda:.2f} de quienes permanecen. La brecha "
                                    "existe, es consistente y es exactamente la señal que un "
                                    "modelo puede aprovechar."
                                ),
                                callout(
                                    "Un promedio global oculta el problema. Intervenir con una "
                                    "política única para toda la empresa gasta presupuesto donde "
                                    "no hace falta y deja sin cubrir las áreas críticas.",
                                    titulo="Consecuencia práctica",
                                    color=COLOR_ABANDONA,
                                ),
                            ],
                            titulo="Qué muestra la distribución",
                            color=COLOR_ABANDONA,
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Detalle por área"),
            card(
                _tabla_detalle(),
                subtitulo="p.p. = puntos porcentuales de diferencia frente a la media global.",
                color=COLOR_PERMANECE,
                className="mb-4",
            ),
            section_title("Formulación del problema"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(
                                "La organización registra información de sus empleados, pero "
                                "no la usa para anticipar salidas: la detección ocurre cuando "
                                "la renuncia ya es un hecho. No existe un instrumento que "
                                "estime el riesgo de abandono de forma individual ni que "
                                "identifique qué factores lo impulsan, de modo que las "
                                "acciones de retención se aplican de manera reactiva y "
                                "homogénea, sin priorización basada en evidencia."
                            ),
                            titulo="Problema identificado",
                            color=COLOR_ABANDONA,
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card(
                            bullet_list(PREGUNTAS, color=COLOR_PERMANECE),
                            titulo="Preguntas de investigación",
                            color=COLOR_PERMANECE,
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
        ],
        className="tab-content",
    )
