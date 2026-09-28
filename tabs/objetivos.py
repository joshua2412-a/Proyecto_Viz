"""
Pestaña 4 · Objetivos
Objetivo general, objetivos específicos y criterios con los que se considera
que el proyecto cumplió.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    card,
    data_table,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import load_metrics
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

OBJETIVOS_ESPECIFICOS = [
    (
        "1",
        "Caracterizar la cohorte",
        "Describir el perfil clínico (edad, género, grupo racial) y mutacional de "
        "los 839 pacientes, y verificar la calidad del dataset: duplicados, valores "
        "faltantes, valores atípicos y supuestos distribucionales.",
    ),
    (
        "2",
        "Medir la asociación de cada variable con el grado",
        "Contrastar cada predictora contra el grado tumoral con la prueba adecuada a "
        "su naturaleza (Mann-Whitney para la edad, chi-cuadrado y V de Cramér para "
        "las binarias y categóricas) y ordenar las variables por capacidad "
        "discriminativa.",
    ),
    (
        "3",
        "Descartar redundancia entre marcadores",
        "Evaluar la co-ocurrencia entre mutaciones para detectar multicolinealidad y "
        "confirmar que el conjunto retenido puede entrar completo en un modelo lineal "
        "sin inestabilidad en la estimación de los coeficientes.",
    ),
    (
        "4",
        "Construir un clasificador interpretable de referencia",
        "Entrenar una regresión logística con búsqueda de hiperparámetros y "
        "validación cruzada estratificada, y traducir sus coeficientes a odds ratios "
        "para poder discutirlos clínicamente.",
    ),
    (
        "5",
        "Evaluar con métricas sensibles al contexto clínico",
        "Reportar accuracy, precisión, recall, F1 y AUC-ROC sobre un conjunto de "
        "prueba reservado, prestando especial atención al recall de GBM: dejar pasar "
        "un tumor agresivo es el error más costoso.",
    ),
    (
        "6",
        "Poner el análisis a disposición de otros",
        "Publicar el desarrollo completo como Jupyter Book y una capa interactiva "
        "—este dashboard— que permita explorar el EDA y simular perfiles de paciente "
        "sin tocar el código.",
    ),
]


def _tabla_criterios() -> dbc.Table:
    """Criterios de cumplimiento contrastados con las métricas obtenidas."""
    metricas = load_metrics()
    filas = [
        [
            "AUC-ROC en prueba ≥ 0,85",
            f"{metricas['roc_auc']:.3f}",
            "Cumple" if metricas["roc_auc"] >= 0.85 else "No cumple",
        ],
        [
            "Recall de GBM ≥ 0,85",
            f"{metricas['recall']:.3f}",
            "Cumple" if metricas["recall"] >= 0.85 else "No cumple",
        ],
        [
            "Accuracy en prueba ≥ 0,80",
            f"{metricas['accuracy']:.3f}",
            "Cumple" if metricas["accuracy"] >= 0.80 else "No cumple",
        ],
        [
            "Modelo interpretable variable a variable",
            f"{metricas['n_variables_activas']} columnas activas de {metricas['n_columnas_modelo']}",
            "Cumple",
        ],
        [
            "Estabilidad entre validación cruzada y prueba",
            f"AUC CV {metricas['cv_auc_media']:.3f} ± {metricas['cv_auc_desviacion']:.3f}",
            "Cumple"
            if abs(metricas["cv_auc_media"] - metricas["roc_auc"]) < 0.05
            else "Revisar",
        ],
    ]
    return data_table(["Criterio", "Resultado obtenido", "Estado"], filas)


def layout() -> html.Div:
    """Layout de la pestaña de objetivos."""
    return html.Div(
        [
            page_header(
                "Objetivos del proyecto",
                "Qué se propone construir, con qué pasos y con qué criterio se "
                "considera suficiente el resultado.",
                "🎯",
            ),
            card(
                [
                    paragraph(
                        "Construir una solución analítica integral que identifique el "
                        "subconjunto óptimo de factores clínicos y mutaciones genéticas "
                        "para clasificar con precisión la severidad del glioma (LGG frente "
                        "a GBM), reduciendo los costos asociados a pruebas moleculares "
                        "innecesarias."
                    ),
                    paragraph(
                        "El énfasis está en la palabra óptimo: no se busca el modelo más "
                        "complejo, sino el que consiga una precisión clínicamente útil con "
                        "la menor cantidad de información molecular posible y con un "
                        "razonamiento que un especialista pueda auditar."
                    ),
                ],
                titulo="Objetivo general",
                color=COLOR_LGG,
            ),
            section_title("Objetivos específicos"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(numero, className="guide-icon"),
                                html.Div(titulo, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                            ],
                            color=SERIES[i % len(SERIES)],
                            className="guide-card",
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (numero, titulo, descripcion) in enumerate(
                        OBJETIVOS_ESPECIFICOS
                    )
                ],
                className="g-3",
            ),
            section_title("Criterios de cumplimiento"),
            card(
                [
                    paragraph(
                        "Los umbrales se fijaron antes de entrenar, tomando como referencia "
                        "el desempeño que reporta la literatura de clasificación de grado "
                        "tumoral en gliomas con variables clínicas y mutacionales. La "
                        "columna de resultado se calcula en vivo a partir de "
                        "model/metrics.json, así que se actualiza sola al reentrenar."
                    ),
                    _tabla_criterios(),
                ],
                titulo="Estado frente a los criterios definidos",
                subtitulo="Métricas sobre el conjunto de prueba reservado (168 pacientes)",
                color=COLOR_GBM,
            ),
        ],
        className="tab-content",
    )
