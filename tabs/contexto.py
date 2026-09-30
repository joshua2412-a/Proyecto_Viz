"""
Pestaña 2 · Contexto clínico
Qué son LGG y GBM, cómo se distribuyen en la cohorte y por qué la edad y el
perfil molecular son la base del diagnóstico actual.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.formato import num
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
from utils.data_loader import (
    estadisticas_edad,
    prueba_edad_por_grado,
    tamanos_particion,
)
from utils.figures import fig_boxplot_edad, fig_distribucion_grado, fig_edad_por_grado
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

AMBITO = "train"  # el EDA se describe sobre el conjunto de entrenamiento


def _kpis() -> list[dict]:
    """Indicadores clínicos de la cohorte de entrenamiento."""
    edad = estadisticas_edad(AMBITO).set_index("grade_label")
    prueba = prueba_edad_por_grado(AMBITO)
    tamanos = tamanos_particion()

    return [
        {
            "valor": f"{num(edad.loc['LGG', 'media'], 1)} años",
            "etiqueta": "Edad media en LGG",
            "detalle": f"± {num(edad.loc['LGG', 'desviacion'], 1)} · {int(edad.loc['LGG', 'pacientes'])} pacientes",
            "color": COLOR_LGG,
        },
        {
            "valor": f"{num(edad.loc['GBM', 'media'], 1)} años",
            "etiqueta": "Edad media en GBM",
            "detalle": f"± {num(edad.loc['GBM', 'desviacion'], 1)} · {int(edad.loc['GBM', 'pacientes'])} pacientes",
            "color": COLOR_GBM,
        },
        {
            "valor": f"{num(edad.loc['GBM', 'media'] - edad.loc['LGG', 'media'], 1)} años",
            "etiqueta": "Diferencia de edad",
            # La cifra es la distancia entre los dos grados, no una propiedad
            # de uno solo: la franja parte los dos colores.
            "color": f"linear-gradient(180deg, {COLOR_LGG} 0 50%, {COLOR_GBM} 50% 100%)",
            "detalle": "Mann-Whitney p < 0,0001" if prueba["p_valor"] < 0.0001
            else f"Mann-Whitney p = {num(prueba['p_valor'], 4)}",
        },
        {
            "valor": f"{tamanos['train']}",
            "etiqueta": "Cohorte descrita",
            "detalle": f"Conjunto de entrenamiento de {tamanos['full']} pacientes",
        },
    ]


def layout() -> html.Div:
    """Layout de la pestaña de contexto clínico."""
    return html.Div(
        [
            page_header(
                "Contexto clínico: dos tumores con el mismo origen",
                "Los gliomas nacen de las células gliales, pero LGG y GBM se "
                "comportan de forma distinta y se diagnostican en edades distintas.",
                "bi-heart-pulse",
            ),
            kpi_row(_kpis()),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "El glioma es un tumor que se origina en las células "
                                    "gliales del sistema nervioso central. La Organización "
                                    "Mundial de la Salud lo clasifica por grados según su "
                                    "agresividad; en este proyecto la escala se reduce a la "
                                    "distinción que más pesa en la decisión clínica:"
                                ),
                                bullet_list(
                                    [
                                        html.Span(
                                            [
                                                html.B("LGG · glioma de bajo grado. "),
                                                "Crecimiento lento, supervivencia larga y "
                                                "un tratamiento que puede ser conservador. "
                                                "Aparece con más frecuencia en adultos "
                                                "jóvenes y suele portar mutación en IDH1.",
                                            ]
                                        ),
                                        html.Span(
                                            [
                                                html.B("GBM · glioblastoma multiforme. "),
                                                "El glioma más agresivo: crecimiento "
                                                "rápido, pronóstico corto y necesidad de "
                                                "tratamiento intensivo inmediato. Se "
                                                "concentra en edades avanzadas y presenta "
                                                "alteraciones en PTEN, EGFR o TP53.",
                                            ]
                                        ),
                                    ],
                                ),
                                paragraph(
                                    "Hasta hace poco el grado se establecía por histología e "
                                    "imagen. La clasificación actual incorpora el perfil "
                                    "molecular porque dos tumores con el mismo aspecto al "
                                    "microscopio pueden tener pronósticos muy distintos "
                                    "según qué genes estén mutados."
                                ),
                                callout(
                                    "Esa es la premisa del proyecto: si el perfil molecular "
                                    "es lo que define el grado, un subconjunto pequeño de "
                                    "marcadores debería bastar para clasificarlo.",
                                    titulo="De la histología al perfil molecular",
                                ),
                            ],
                            titulo="Qué distingue a LGG de GBM",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        [
                            graph_card(
                                fig_distribucion_grado(AMBITO),
                                "Composición de la cohorte",
                                "Desbalance leve (ratio ≈ 1,4:1), no severo: no hace falta "
                                "reponderar las clases.",
                            ),
                            html.Div(series_chips(), className="mt-3"),
                        ],
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("La edad al diagnóstico separa a los dos grupos"),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_edad_por_grado(AMBITO),
                            "Distribución de la edad por grado tumoral",
                            "Porcentaje dentro de cada grado para que el tamaño distinto "
                            "de los grupos no distorsione la comparación.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_boxplot_edad(AMBITO),
                            "Rango y dispersión por grado",
                            "Los puntos son valores atípicos dentro de cada grupo: casos de "
                            "GBM inusualmente jóvenes y de LGG inusualmente mayores.",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            card(
                [
                    paragraph(
                        "La separación es clínicamente relevante y estadísticamente "
                        "significativa: el 75 % de los pacientes con LGG tiene 54,3 años o "
                        "menos, mientras que el 75 % de los casos de GBM se diagnostica a "
                        "partir de los 52,8 años. La prueba U de Mann-Whitney (elegida "
                        "porque la edad no sigue una distribución normal: es bimodal, con "
                        "picos cerca de los 35 y los 55 años) rechaza la hipótesis de "
                        "igualdad de distribuciones con p < 0,0001."
                    ),
                    data_table(
                        ["Grado", "Pacientes", "Media", "Mediana", "Q1 - Q3", "Rango"],
                        [
                            [
                                fila["grade_label"],
                                f"{int(fila['pacientes'])}",
                                f"{num(fila['media'], 2)} ± {num(fila['desviacion'], 2)}",
                                f"{num(fila['mediana'], 2)}",
                                f"{num(fila['q1'], 2)} - {num(fila['q3'], 2)}",
                                f"{num(fila['minimo'], 2)} - {num(fila['maximo'], 2)}",
                            ]
                            for _, fila in estadisticas_edad(AMBITO).iterrows()
                        ],
                    ),
                    callout(
                        "La edad es un correlato fuerte, no una causa: hay pacientes de 22 "
                        "años con glioblastoma y de 87 con glioma de bajo grado. Sirve para "
                        "estimar riesgo, nunca para descartar un diagnóstico.",
                        titulo="Lectura prudente",
                    ),
                ],
                titulo="Edad al diagnóstico por grado tumoral",
                subtitulo="Conjunto de entrenamiento · valores en años",
            ),
        ],
        className="vista-pestana",
    )
