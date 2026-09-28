"""
Pestaña 5 · Marco teórico
Los conceptos que hacen falta para leer el resto del tablero: el panel de
genes, la regresión logística, el odds ratio y las métricas de evaluación.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    enlace_externo,
    page_header,
    paragraph,
    section_title,
)
from utils.config import (
    GENDER_LABELS,
    GENE_DESCRIPCION,
    GENE_FEATURES,
    RACE_LABELS,
    URL_LIBRO_MODELO,
)
from utils.data_loader import asociacion_con_grado, prevalencia_genes
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

AMBITO = "train"

# Fórmulas en LaTeX (dcc.Markdown con mathjax=True las renderiza)
FORMULA_LOGISTICA = r"""
$$
P(\text{GBM} \mid \mathbf{x}) \;=\; \sigma\!\left(\mathbf{w}^{\top}\mathbf{x} + b\right)
\;=\; \frac{1}{1 + e^{-(\mathbf{w}^{\top}\mathbf{x} + b)}}
$$
"""

FORMULA_ODDS = r"""
$$
\text{odds ratio}_i \;=\; e^{w_i}
\qquad\text{y}\qquad
\log\frac{P(\text{GBM})}{1 - P(\text{GBM})} \;=\; b + \sum_i w_i x_i
$$
"""

METRICAS = [
    (
        "Accuracy",
        "Proporción de pacientes clasificados correctamente.",
        "Fácil de comunicar, pero insuficiente sola: con clases desbalanceadas "
        "premia acertar en la clase mayoritaria.",
    ),
    (
        "Precisión (GBM)",
        "De los pacientes que el modelo marca como GBM, cuántos lo son de verdad.",
        "Baja precisión significa enviar a revisión casos que no lo necesitaban.",
    ),
    (
        "Recall / sensibilidad (GBM)",
        "De los pacientes que realmente tienen GBM, cuántos detecta el modelo.",
        "La métrica crítica de este problema: un GBM no detectado es el error más "
        "costoso clínicamente.",
    ),
    (
        "F1-score (GBM)",
        "Media armónica entre precisión y recall.",
        "Resume el equilibrio entre los dos tipos de error en una sola cifra.",
    ),
    (
        "AUC-ROC",
        "Probabilidad de que el modelo asigne mayor riesgo a un GBM que a un LGG "
        "tomados al azar.",
        "Independiente del umbral de decisión y robusta al desbalance: es la métrica "
        "con la que se seleccionaron los hiperparámetros.",
    ),
]


def _tabla_variables() -> dbc.Table:
    """Operacionalización de las variables del dataset."""
    return data_table(
        ["Variable", "Tipo", "Codificación", "Papel en el modelo"],
        [
            [
                "Grade",
                "Binaria",
                "0 = LGG · 1 = GBM",
                "Variable objetivo",
            ],
            [
                "Age_at_diagnosis",
                "Continua (años)",
                "Decimales incluidos: recogen los días exactos",
                "Predictora · estandarizada con StandardScaler",
            ],
            [
                "Gender",
                "Binaria",
                " · ".join(f"{clave} = {valor}" for clave, valor in GENDER_LABELS.items()),
                "Predictora · one-hot (drop='if_binary')",
            ],
            [
                "Race",
                "Categórica (4 niveles)",
                " · ".join(f"{clave} = {valor}" for clave, valor in RACE_LABELS.items()),
                "Predictora · one-hot",
            ],
            [
                f"{len(GENE_FEATURES)} genes",
                "Binarias",
                "0 = no mutado (wildtype) · 1 = mutado",
                "Predictoras · passthrough (ya son indicadores)",
            ],
        ],
    )


def _tabla_genes() -> dbc.Table:
    """Panel de genes con su función biológica y su prevalencia observada."""
    prevalencia = prevalencia_genes(AMBITO).set_index("gen")
    asociacion = asociacion_con_grado(AMBITO).set_index("variable")

    filas = []
    for gen in prevalencia.index:  # ya viene ordenado por prevalencia
        fila_prev = prevalencia.loc[gen]
        rho = asociacion.loc[gen, "rho"]
        hacia = "LGG" if rho < 0 else "GBM"
        marca = "" if asociacion.loc[gen, "significativa"] else " (no significativa)"
        filas.append(
            [
                gen,
                GENE_DESCRIPCION.get(gen, ""),
                f"{fila_prev['prevalencia']:.1f}%",
                f"{rho:+.2f} → {hacia}{marca}",
            ]
        )
    return data_table(
        ["Gen", "Función biológica", "Prevalencia", "Asociación con el grado"], filas
    )


def layout() -> html.Div:
    """Layout de la pestaña de marco teórico."""
    return html.Div(
        [
            page_header(
                "Marco teórico",
                "Qué mide cada variable, cómo funciona el modelo y cómo se "
                "interpretan sus resultados.",
                "📚",
            ),
            section_title("Operacionalización de las variables"),
            card(
                [
                    paragraph(
                        "El dataset llega ya codificado numéricamente. Conocer esa "
                        "codificación es imprescindible para leer los coeficientes: un "
                        "coeficiente positivo en Gender no significa 'ser hombre aumenta "
                        "el riesgo', sino que la categoría codificada como 1 (femenino) lo "
                        "hace respecto a la de referencia."
                    ),
                    _tabla_variables(),
                ],
                titulo="Las 23 predictoras y la variable objetivo",
                color=COLOR_LGG,
            ),
            section_title("El modelo: regresión logística"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Dado un vector de características, el modelo estima la "
                                    "probabilidad de que el paciente pertenezca a la clase "
                                    "GBM aplicando la función logística (sigmoide) a una "
                                    "combinación lineal de las predictoras:"
                                ),
                                dcc.Markdown(
                                    FORMULA_LOGISTICA,
                                    mathjax=True,
                                    className="formula-block",
                                ),
                                paragraph(
                                    "Se eligió como modelo de referencia por cuatro razones, "
                                    "todas ellas explícitas en el libro:"
                                ),
                                bullet_list(
                                    [
                                        "Interpretabilidad: sus coeficientes se traducen "
                                        "directamente en odds ratios, el estándar de facto "
                                        "en la literatura de oncología genómica.",
                                        "Simplicidad y bajo costo computacional: "
                                        "entrenamiento e inferencia casi inmediatos.",
                                        "Robustez con pocas muestras: bajo riesgo de "
                                        "sobreajuste con 671 pacientes y 23 predictoras.",
                                        "Comparabilidad: es el modelo base reportado en la "
                                        "mayoría de estudios de clasificación de grado en "
                                        "gliomas.",
                                    ],
                                    color=COLOR_LGG,
                                ),
                            ],
                            titulo="Formulación",
                            color=COLOR_LGG,
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Cada coeficiente indica cómo influye su variable en la "
                                    "probabilidad estimada de GBM. Su exponencial es el "
                                    "odds ratio: por cuánto se multiplican las "
                                    "probabilidades de GBM al aumentar esa variable en una "
                                    "unidad (o, en las binarias, al pasar de 0 a 1)."
                                ),
                                dcc.Markdown(
                                    FORMULA_ODDS,
                                    mathjax=True,
                                    className="formula-block",
                                ),
                                bullet_list(
                                    [
                                        "Coeficiente positivo (OR > 1): la variable empuja "
                                        "la predicción hacia GBM.",
                                        "Coeficiente negativo (OR < 1): la empuja hacia LGG.",
                                        "OR = 2,0 duplica las chances de GBM; OR = 0,5 las "
                                        "reduce a la mitad.",
                                    ],
                                    color=COLOR_GBM,
                                ),
                                callout(
                                    "La penalización L1 (Lasso) lleva a cero los "
                                    "coeficientes de las variables poco informativas: hace "
                                    "selección de variables dentro del propio ajuste. Por "
                                    "eso el modelo final usa muchas menos variables de las "
                                    "que recibe.",
                                    titulo="Por qué L1",
                                    color=SERIES[3],
                                ),
                            ],
                            titulo="Odds ratio: cómo se lee un coeficiente",
                            color=COLOR_GBM,
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Métricas de evaluación"),
            card(
                data_table(
                    ["Métrica", "Qué mide", "Por qué importa aquí"],
                    [[nombre, definicion, motivo] for nombre, definicion, motivo in METRICAS],
                ),
                titulo="Qué se reporta y por qué",
                subtitulo="Las cinco métricas que aparecen en la pestaña de resultados",
                color=SERIES[2],
            ),
            section_title("El panel de genes"),
            card(
                [
                    paragraph(
                        "Los 20 genes del panel son los de mayor frecuencia de mutación en "
                        "los proyectos TCGA-LGG y TCGA-GBM. La prevalencia y la asociación "
                        "se calculan sobre el conjunto de entrenamiento, así que cambian "
                        "solas si se reemplaza el dataset."
                    ),
                    _tabla_genes(),
                    html.Div(
                        enlace_externo(
                            "Ver el desarrollo completo del modelo en el libro",
                            URL_LIBRO_MODELO,
                        ),
                        className="mt-3",
                    ),
                ],
                titulo="Función biológica, prevalencia y dirección de cada gen",
                subtitulo="Ordenado por prevalencia de mutación en la cohorte de entrenamiento",
                color=SERIES[3],
            ),
        ],
        className="tab-content",
    )
