"""
Pestaña 10 · Conclusiones
Qué quedó demostrado, qué significa para el objetivo de reducir el panel de
secuenciación y qué falta por hacer.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    enlace_externo,
    kpi_row,
    page_header,
    paragraph,
    section_title,
)
from utils.config import URL_LIBRO, URL_REPO_LIBRO
from utils.data_loader import load_metrics
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

SIGUIENTES_PASOS = [
    (
        "Contrastar contra modelos más complejos",
        "Entrenar los demás baselines previstos (árboles, ensembles, redes) con la "
        "misma partición y las mismas métricas, y comparar en las dos dimensiones: "
        "error y coste computacional. Si el baseline lineal no queda por debajo, se "
        "reporta así.",
    ),
    (
        "Cuantificar el panel mínimo",
        "Reentrenar con paneles progresivamente más pequeños (solo IDH1; IDH1 + edad; "
        "los cinco genes con coeficiente no nulo) y medir cuánta AUC se pierde en "
        "cada recorte. Es la respuesta directa al objetivo del proyecto.",
    ),
    (
        "Evaluar la calibración",
        "Añadir curvas de calibración y, si hace falta, un calibrador, para que la "
        "probabilidad que muestra el simulador pueda interpretarse como frecuencia "
        "esperada y no solo como puntuación de riesgo.",
    ),
    (
        "Completar el capítulo de preprocesamiento",
        "El notebook de preprocesamiento quedó como borrador. Consolidar ahí las "
        "decisiones que hoy están repartidas entre el EDA y el pipeline haría el "
        "libro autocontenido.",
    ),
]


def _kpis() -> list[dict]:
    """Las cuatro cifras que resumen el resultado del proyecto."""
    metricas = load_metrics()
    return [
        {
            "valor": f"{metricas['roc_auc']:.3f}",
            "etiqueta": "AUC-ROC en prueba",
            "detalle": "Capacidad de discriminación LGG vs GBM",
            "color": SERIES[0],
        },
        {
            "valor": f"{metricas['recall']:.1%}",
            "etiqueta": "Recall de GBM",
            "detalle": "El error clínicamente más costoso, minimizado",
            "color": SERIES[1],
        },
        {
            "valor": f"{metricas['n_variables_activas']}",
            "etiqueta": "Columnas que usa el modelo",
            "detalle": f"De {metricas['n_columnas_modelo']} tras el preprocesamiento",
            "color": SERIES[2],
        },
        {
            "valor": f"{metricas['tiempos_segundos']['inferencia'] * 1000:.1f} ms",
            "etiqueta": "Tiempo de inferencia",
            "detalle": f"Para {metricas['n_test']} pacientes",
            "color": SERIES[3],
        },
    ]


def _tabla_hallazgos() -> dbc.Table:
    """Hallazgos principales con su evidencia cuantitativa."""
    return data_table(
        ["Hallazgo", "Evidencia", "Consecuencia"],
        [
            [
                "IDH1 es el marcador dominante",
                "r ≈ -0,70 con el grado · odds ratio 0,034 en el modelo",
                "Su mutación reduce las probabilidades de GBM a una fracción mínima: "
                "es casi por sí solo un clasificador.",
            ],
            [
                "La edad aporta señal clínica independiente",
                "r ≈ +0,53 · odds ratio ≈ 1,95 por desviación estándar",
                "Una variable que ya está en la historia clínica y no cuesta nada "
                "medir mejora la clasificación.",
            ],
            [
                "La mayoría del panel no discrimina el grado",
                "Coeficiente exactamente cero tras la penalización L1",
                "Sostiene la hipótesis central: un panel reducido podría bastar para "
                "decidir el grado.",
            ],
            [
                "No hay multicolinealidad estructural",
                "Ningún par de variables con V de Cramér > 0,70",
                "Los marcadores retenidos pueden entrar juntos en el modelo sin "
                "inestabilizar la estimación.",
            ],
            [
                "El modelo se equivoca hacia el lado seguro",
                "Recall GBM 0,93 frente a precisión 0,79",
                "Sobre-detecta glioblastomas: marca casos de más antes que dejar "
                "pasar un tumor agresivo.",
            ],
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de conclusiones."""
    metricas = load_metrics()

    return html.Div(
        [
            page_header(
                "Conclusiones",
                "Un modelo lineal con una edad y un puñado de mutaciones clasifica el "
                "grado del glioma con solvencia, y dice cuáles son esas mutaciones.",
                "✅",
            ),
            kpi_row(_kpis()),
            card(
                [
                    paragraph(
                        f"El clasificador alcanza un AUC-ROC de {metricas['roc_auc']:.3f} y "
                        f"una accuracy de {metricas['accuracy']:.1%} sobre "
                        f"{metricas['n_test']} pacientes que no intervinieron en el "
                        "entrenamiento, con una diferencia mínima frente al AUC de "
                        f"validación cruzada ({metricas['cv_auc_media']:.3f}): el desempeño "
                        "es estable, no un artefacto de la partición."
                    ),
                    paragraph(
                        f"Lo relevante para el objetivo del proyecto no es esa cifra, sino "
                        f"con cuánta información se consigue: la penalización L1 dejó "
                        f"activas {metricas['n_variables_activas']} de "
                        f"{metricas['n_columnas_modelo']} columnas. El modelo no solo clasifica; "
                        "señala explícitamente qué parte del panel molecular está "
                        "sosteniendo la decisión y qué parte no aporta nada al grado."
                    ),
                    callout(
                        "La señal biológica que encuentra el modelo coincide con la "
                        "literatura: IDH1 e IDH2 hacia LGG, TP53, PTEN y la edad hacia "
                        "GBM. Que un modelo entrenado a ciegas reproduzca el conocimiento "
                        "clínico establecido es la mejor validación disponible sin una "
                        "cohorte externa.",
                        titulo="Coherencia con el conocimiento clínico",
                        color=SERIES[2],
                    ),
                ],
                titulo="Resultado principal",
                color=COLOR_LGG,
            ),
            section_title("Hallazgos y su evidencia"),
            card(
                _tabla_hallazgos(),
                titulo="Cinco hallazgos con respaldo cuantitativo",
                subtitulo="Cifras del EDA sobre entrenamiento y del modelo sobre prueba",
                color=SERIES[3],
                className="mb-4",
            ),
            section_title("Qué falta por hacer"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(descripcion),
                            titulo=titulo,
                            color=SERIES[i % len(SERIES)],
                        ),
                        lg=6,
                        className="mb-3",
                    )
                    for i, (titulo, descripcion) in enumerate(SIGUIENTES_PASOS)
                ],
                className="g-3",
            ),
            card(
                [
                    paragraph("El proyecto se entrega en tres piezas complementarias:"),
                    bullet_list(
                        [
                            html.Span(
                                [
                                    html.B("Jupyter Book: "),
                                    "el análisis completo con su desarrollo estadístico. ",
                                    enlace_externo("Abrirlo", URL_LIBRO),
                                ]
                            ),
                            html.Span(
                                [
                                    html.B("Dashboard: "),
                                    "esta capa interactiva, con el EDA navegable y el "
                                    "simulador de perfiles.",
                                ]
                            ),
                            html.Span(
                                [
                                    html.B("Código: "),
                                    "pipeline reproducible con una semilla fija, del CSV "
                                    "al modelo entrenado. ",
                                    enlace_externo("Repositorio del libro", URL_REPO_LIBRO),
                                ]
                            ),
                        ],
                        color=COLOR_GBM,
                    ),
                ],
                titulo="Entregables",
                color=COLOR_GBM,
            ),
        ],
        className="tab-content",
    )
