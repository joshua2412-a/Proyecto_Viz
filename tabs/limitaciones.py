"""
Pestaña 9 · Limitaciones
Qué no puede afirmarse con este trabajo, y qué haría falta para poder afirmarlo.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import load_metrics
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# (categoría, limitación, implicación práctica)
LIMITACIONES = [
    (
        "Datos",
        "Los datos son sintéticos",
        "El dataset se generó con un proceso conocido y no proviene de una "
        "organización real. Los coeficientes reflejan ese proceso generador, por lo "
        "que las magnitudes concretas no son transferibles a otra empresa sin "
        "reentrenar con sus propios datos.",
    ),
    (
        "Datos",
        "Corte transversal sin dimensión temporal",
        "Cada empleado aparece una sola vez, en un único momento. El modelo estima "
        "si alguien abandona, no cuándo, y no puede capturar deterioros graduales "
        "(una satisfacción que baja tres puntos en un año).",
    ),
    (
        "Variables",
        "Variables relevantes ausentes",
        "Quedan fuera factores con peso reconocido en la literatura: calidad de la "
        "relación con el jefe directo, distancia al lugar de trabajo, ofertas "
        "externas recibidas, motivos personales o de salud, flexibilidad horaria.",
    ),
    (
        "Variables",
        "Satisfacción autodeclarada",
        "Es una medida subjetiva sujeta a sesgo de deseabilidad social: quien ya "
        "decidió irse puede responder de forma estratégica o simplemente no "
        "responder la encuesta.",
    ),
    (
        "Modelo",
        "Supuesto de linealidad en el logit",
        "La regresión logística asume un efecto lineal sobre el logaritmo de las "
        "odds. Relaciones no lineales o interacciones (por ejemplo, sobrecarga que "
        "solo pesa en salarios bajos) no se capturan sin especificarlas de forma "
        "explícita.",
    ),
    (
        "Modelo",
        "Precisión limitada por el desbalance",
        "Con `class_weight='balanced'` el modelo gana recall a costa de precisión: "
        "marca como riesgo a personas que no pensaban irse. Es un intercambio "
        "deliberado, pero implica un coste operativo en revisiones innecesarias.",
    ),
    (
        "Alcance",
        "Asociación, no causalidad",
        "El diseño es observacional. Un coeficiente alto no autoriza a concluir que "
        "modificar esa variable reduzca el abandono: haría falta un diseño "
        "experimental o cuasi-experimental para sostener esa afirmación.",
    ),
    (
        "Alcance",
        "Validez temporal limitada",
        "Los patrones de rotación cambian con el mercado laboral y con la propia "
        "organización. Un modelo entrenado hoy se degrada; requiere reentrenamiento "
        "y monitoreo periódico.",
    ),
    (
        "Ético",
        "Riesgo de uso indebido",
        "Una probabilidad alta no debe convertirse en una etiqueta que condicione "
        "promociones, asignaciones o renovaciones. Usada así, la herramienta "
        "produciría exactamente el daño que pretende evitar.",
    ),
]

MEJORAS = [
    "Incorporar histórico longitudinal para modelar la evolución del riesgo en el tiempo.",
    "Añadir variables de relación con el jefe directo y de calidad del equipo.",
    "Comparar contra modelos no lineales (bosques aleatorios, gradient boosting) "
    "usando SHAP para conservar la explicabilidad.",
    "Aplicar análisis de supervivencia (Cox, Kaplan-Meier) para estimar el tiempo hasta la salida.",
    "Calibrar el umbral de decisión con el coste real de cada tipo de error.",
    "Auditar sesgos por grupo (edad, género, área) antes de cualquier uso en producción.",
    "Implementar monitoreo de deriva de datos y reentrenamiento programado.",
]


def layout() -> html.Div:
    """Layout de la pestaña de limitaciones."""
    metricas = load_metrics()
    cm = metricas["matriz_confusion"]
    falsos_negativos = cm[1][0]
    falsos_positivos = cm[0][1]

    categorias = ["Datos", "Variables", "Modelo", "Alcance", "Ético"]
    colores = {cat: SERIES[i % len(SERIES)] for i, cat in enumerate(categorias)}

    return html.Div(
        [
            page_header(
                "Limitaciones",
                "Un modelo honesto declara sus fronteras: estas son las de este trabajo.",
                "⚠️",
            ),
            callout(
                f"En el conjunto de prueba el modelo dejó pasar {falsos_negativos} "
                f"abandonos reales y marcó {falsos_positivos} empleados que se quedaron. "
                "Ninguna de las dos cifras es un defecto oculto: son la consecuencia "
                "medida del umbral elegido y deben conocerse antes de usar la "
                "herramienta para tomar decisiones.",
                titulo="El error, con números",
                color=COLOR_ABANDONA,
            ),
            section_title("Limitaciones identificadas"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Span(
                                    categoria,
                                    className="etiqueta-categoria",
                                    style={"background": colores[categoria]},
                                ),
                                html.Div(titulo, className="guide-name"),
                                html.Div(texto, className="guide-text"),
                            ],
                            color=colores[categoria],
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for categoria, titulo, texto in LIMITACIONES
                ],
                className="g-3 mb-4",
            ),
            section_title("Qué haría falta para superarlas"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            bullet_list(MEJORAS, color=COLOR_PERMANECE),
                            titulo="Líneas de trabajo futuras",
                            color=COLOR_PERMANECE,
                        ),
                        lg=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card(
                            data_table(
                                ["Lo que el modelo sí hace", "Lo que no hace"],
                                [
                                    ["Estima una probabilidad de abandono", "Predice una fecha de salida"],
                                    ["Ordena casos por prioridad de atención", "Explica el motivo real de la renuncia"],
                                    ["Identifica variables asociadas", "Demuestra causalidad"],
                                    ["Sugiere dónde investigar", "Sustituye la conversación con la persona"],
                                    ["Apoya decisiones de retención", "Justifica decisiones disciplinarias"],
                                ],
                                resaltar_primera_columna=False,
                            ),
                            titulo="Frontera de uso",
                            color=COLOR_ABANDONA,
                        ),
                        lg=5,
                        className="mb-3",
                    ),
                ]
            ),
            card(
                paragraph(
                    "Declarar estas limitaciones no debilita el trabajo: lo hace "
                    "utilizable. Una herramienta cuyo margen de error se conoce puede "
                    "integrarse en un proceso de decisión con las salvaguardas "
                    "adecuadas; una que se presenta como infalible acaba retirada en "
                    "el primer caso en que falla."
                ),
                titulo="Nota final",
                color=SERIES[3],
            ),
        ],
        className="tab-content",
    )
