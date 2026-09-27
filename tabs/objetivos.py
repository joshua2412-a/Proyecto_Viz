"""
Pestaña 4 · Objetivos y justificación
Objetivo general, objetivos específicos, alcance y justificación del proyecto.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    page_header,
    paragraph,
    section_title,
)
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

OBJETIVO_GENERAL = (
    "Desarrollar una herramienta analítica interactiva que caracterice la rotación "
    "laboral de la organización y estime la probabilidad de abandono de cada "
    "empleado mediante un modelo de regresión logística, con el fin de apoyar "
    "decisiones de retención basadas en evidencia."
)

OBJETIVOS_ESPECIFICOS = [
    (
        "1",
        "Construir la base de datos",
        "Generar un conjunto de datos de empleados con las variables relevantes "
        "para el análisis de rotación y documentar su proceso de construcción.",
    ),
    (
        "2",
        "Caracterizar el fenómeno",
        "Describir la magnitud y distribución del abandono mediante análisis "
        "exploratorio: composición de la plantilla, tasas por departamento, "
        "distribuciones y correlaciones entre variables.",
    ),
    (
        "3",
        "Entrenar el modelo",
        "Ajustar un modelo de regresión logística que clasifique el abandono, "
        "con preprocesamiento reproducible y validación sobre datos no vistos.",
    ),
    (
        "4",
        "Evaluar el desempeño",
        "Medir accuracy, precisión, recall, F1 y AUC, e interpretar la matriz de "
        "confusión en términos del coste de cada tipo de error.",
    ),
    (
        "5",
        "Identificar factores",
        "Interpretar los coeficientes del modelo como razones de odds para "
        "determinar qué variables incrementan o reducen el riesgo de salida.",
    ),
    (
        "6",
        "Entregar un simulador",
        "Publicar un formulario interactivo que devuelva la probabilidad de "
        "abandono de un perfil concreto en tiempo real.",
    ),
]

JUSTIFICACION = [
    (
        "Pertinencia práctica",
        "Anticipar una salida abre una ventana de acción —conversación, ajuste "
        "salarial, cambio de equipo, plan de carrera— que desaparece por completo "
        "cuando la renuncia ya está presentada.",
    ),
    (
        "Viabilidad técnica",
        "El modelo usa siete variables que cualquier área de gestión humana ya "
        "registra en su sistema de nómina. No requiere nuevos instrumentos de "
        "recolección ni inversión en infraestructura.",
    ),
    (
        "Interpretabilidad",
        "La regresión logística entrega coeficientes traducibles a razones de "
        "odds: el área usuaria entiende por qué el modelo marca un caso, lo que "
        "es condición necesaria para que la recomendación se aplique.",
    ),
    (
        "Eficiencia del gasto",
        "Priorizar permite concentrar el presupuesto de retención en los casos "
        "con riesgo real, en lugar de repartirlo de forma uniforme.",
    ),
]

ALCANCE = {
    "Incluye": [
        "Análisis descriptivo de la plantilla y de la rotación.",
        "Modelo de clasificación binaria con validación en datos de prueba.",
        "Interpretación de factores asociados al abandono.",
        "Simulador de riesgo individual.",
    ],
    "No incluye": [
        "Inferencia causal sobre los motivos de la renuncia.",
        "Predicción de la fecha concreta de salida.",
        "Datos reales de empleados identificables.",
        "Integración con sistemas de nómina en producción.",
    ],
}


def layout() -> html.Div:
    """Layout de la pestaña de objetivos."""
    return html.Div(
        [
            page_header(
                "Objetivos y justificación",
                "Qué se propone lograr el proyecto, con qué alcance y por qué "
                "vale la pena hacerlo.",
                "🎯",
            ),
            card(
                html.Div(OBJETIVO_GENERAL, className="objetivo-general"),
                titulo="Objetivo general",
                color=COLOR_PERMANECE,
                className="mb-4",
            ),
            section_title("Objetivos específicos"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(numero, className="objetivo-numero"),
                                html.Div(titulo, className="guide-name"),
                                html.Div(texto, className="guide-text"),
                            ],
                            color=SERIES[i % len(SERIES)],
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (numero, titulo, texto) in enumerate(OBJETIVOS_ESPECIFICOS)
                ],
                className="g-3 mb-4",
            ),
            section_title("Justificación"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(texto),
                            titulo=titulo,
                            color=SERIES[i % len(SERIES)],
                        ),
                        lg=6,
                        className="mb-3",
                    )
                    for i, (titulo, texto) in enumerate(JUSTIFICACION)
                ],
                className="g-3 mb-4",
            ),
            section_title("Alcance del trabajo"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            bullet_list(ALCANCE["Incluye"], color=COLOR_PERMANECE),
                            titulo="✅ El proyecto incluye",
                            color=COLOR_PERMANECE,
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card(
                            bullet_list(ALCANCE["No incluye"], color=COLOR_ABANDONA),
                            titulo="⛔ El proyecto no incluye",
                            color=COLOR_ABANDONA,
                        ),
                        lg=6,
                        className="mb-3",
                    ),
                ]
            ),
            callout(
                "Delimitar el alcance es parte del rigor: el modelo estima "
                "asociación estadística, no causalidad. Una variable con un "
                "coeficiente alto señala dónde mirar, no qué cambiar sin más "
                "verificación.",
                titulo="Nota metodológica",
                color=SERIES[3],
            ),
        ],
        className="tab-content",
    )
