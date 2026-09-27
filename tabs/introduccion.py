"""
Pestaña 1 · Introducción
Presenta el problema de la rotación laboral y la guía de lectura del dashboard.

Cada pestaña expone una única función pública `layout()` y es totalmente
autónoma: no importa nada de las demás pestañas.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    kpi_row,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import get_dataframe, load_metrics, tasa_por_departamento
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# Guía de navegación: (icono, pestaña, qué encontrará el usuario)
GUIA = [
    ("📘", "Introducción", "Qué es la rotación laboral y cómo leer este tablero."),
    ("🏢", "Contexto", "Por qué le cuesta dinero a la organización."),
    ("❗", "Problema", "Dónde se concentra el abandono dentro de la empresa."),
    ("🎯", "Objetivos", "Qué se propone resolver el proyecto y por qué importa."),
    ("📚", "Marco teórico", "Conceptos, antecedentes y operacionalización de variables."),
    ("🧪", "Metodología", "Cómo se construyeron los datos y el modelo."),
    ("📊", "Resultados", "Análisis exploratorio y desempeño del clasificador."),
    ("🔮", "Predicción", "Simulador interactivo de riesgo individual."),
    ("⚠️", "Limitaciones", "Qué NO se puede concluir con este trabajo."),
    ("✅", "Conclusiones", "Hallazgos y recomendaciones accionables."),
]


def _kpis() -> list[dict]:
    """Indicadores de portada calculados a partir del dataset y las métricas."""
    df = get_dataframe()
    resumen = tasa_por_departamento()
    metricas = load_metrics()
    critico = resumen.iloc[0]

    return [
        {
            "valor": f"{len(df):,}",
            "etiqueta": "Empleados analizados",
            "detalle": "Registros del dataset simulado",
            "color": SERIES[0],
        },
        {
            "valor": f"{df['abandono'].mean():.1%}",
            "etiqueta": "Tasa de abandono",
            "detalle": f"{int(df['abandono'].sum()):,} salidas registradas",
            "color": SERIES[1],
        },
        {
            "valor": critico["departamento"],
            "etiqueta": "Área más crítica",
            "detalle": f"{critico['tasa']:.1%} de abandono",
            "color": SERIES[3],
        },
        {
            "valor": f"{metricas['roc_auc']:.3f}",
            "etiqueta": "AUC del modelo",
            "detalle": "Regresión logística sobre datos de prueba",
            "color": SERIES[2],
        },
    ]


def layout() -> html.Div:
    """Layout de la pestaña de introducción."""
    return html.Div(
        [
            page_header(
                "Rotación laboral: entender por qué se va el talento",
                "Tablero analítico de employee attrition construido con Dash, "
                "Plotly y un modelo de regresión logística.",
                "📘",
            ),
            kpi_row(_kpis()),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "La rotación laboral (o employee attrition) es la salida "
                                    "de empleados de una organización durante un periodo "
                                    "determinado. Cuando esa salida es voluntaria y afecta a "
                                    "perfiles con experiencia, deja de ser un indicador de "
                                    "recursos humanos y se convierte en un problema "
                                    "estratégico: la empresa pierde conocimiento, continuidad "
                                    "en la relación con el cliente y capacidad de ejecución."
                                ),
                                paragraph(
                                    "El problema no es que exista rotación —siempre existe—, "
                                    "sino que la organización la descubra tarde. Cuando la "
                                    "carta de renuncia llega ya no hay nada que negociar. La "
                                    "pregunta útil no es cuántos se fueron el año pasado, sino "
                                    "quién tiene hoy un riesgo alto de irse y qué factores "
                                    "están empujando esa decisión."
                                ),
                                paragraph(
                                    "Este proyecto responde a esa pregunta con dos piezas: un "
                                    "análisis descriptivo que localiza dónde se concentra el "
                                    "abandono y un modelo de clasificación que estima la "
                                    "probabilidad de salida de un empleado concreto a partir "
                                    "de siete variables observables."
                                ),
                            ],
                            titulo="El problema en una página",
                            color=COLOR_PERMANECE,
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        [
                            card(
                                [
                                    paragraph(
                                        "El tablero se apoya en siete variables recogidas "
                                        "habitualmente por cualquier área de gestión humana:"
                                    ),
                                    bullet_list(
                                        [
                                            "Edad del empleado",
                                            "Salario mensual",
                                            "Antigüedad en la empresa",
                                            "Departamento al que pertenece",
                                            "Satisfacción laboral (escala 1 a 5)",
                                            "Horas trabajadas por semana",
                                            "Promociones recibidas",
                                        ],
                                        color=COLOR_PERMANECE,
                                    ),
                                    callout(
                                        "La variable objetivo es binaria: 1 si el empleado "
                                        "abandonó la organización, 0 si permanece.",
                                        titulo="Variable a predecir",
                                        color=COLOR_ABANDONA,
                                    ),
                                ],
                                titulo="Qué se observa de cada empleado",
                                color=SERIES[3],
                            )
                        ],
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Cómo recorrer el tablero"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(icono, className="guide-icon"),
                                html.Div(nombre, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                            ],
                            color=SERIES[i % len(SERIES)],
                            className="guide-card",
                        ),
                        lg=3,
                        md=4,
                        sm=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (icono, nombre, descripcion) in enumerate(GUIA)
                ],
                className="g-3",
            ),
        ],
        className="tab-content",
    )
