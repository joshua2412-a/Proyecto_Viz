"""
Pestaña 5 · Marco teórico
Conceptos, teorías de rotación laboral, fundamento del modelo estadístico y
tabla de operacionalización de variables.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    page_header,
    paragraph,
    section_title,
)
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# --------------------------------------------------------------------------- #
# Conceptos base
# --------------------------------------------------------------------------- #
CONCEPTOS = [
    (
        "Rotación laboral",
        "Movimiento de entrada y salida de personal en una organización durante "
        "un periodo. Se distingue entre rotación voluntaria (el empleado decide "
        "irse) e involuntaria (la empresa termina la relación). Este trabajo "
        "modela la primera, que es la accionable desde la gestión humana.",
    ),
    (
        "Tasa de abandono (attrition rate)",
        "Proporción de empleados que abandonan respecto al total de la plantilla "
        "en el periodo analizado. Es el indicador descriptivo de referencia, pero "
        "por sí solo no permite priorizar casos individuales.",
    ),
    (
        "Satisfacción laboral",
        "Valoración afectiva que el empleado hace de su trabajo. Es el "
        "antecedente más estudiado de la intención de renuncia: la literatura "
        "reporta de forma consistente una relación negativa entre satisfacción "
        "e intención de salida.",
    ),
    (
        "Intención de rotación",
        "Probabilidad autodeclarada de dejar la organización. Actúa como variable "
        "mediadora entre la satisfacción y el abandono efectivo, y es el "
        "constructo que este modelo aproxima con datos observables.",
    ),
]

TEORIAS = [
    (
        "Teoría de los dos factores (Herzberg, 1959)",
        "Separa factores higiénicos (salario, condiciones, carga de trabajo), cuya "
        "ausencia genera insatisfacción, de factores motivacionales (reconocimiento, "
        "crecimiento, logro), que generan satisfacción. En el modelo, el salario y "
        "las horas trabajadas representan los primeros; las promociones, los segundos.",
    ),
    (
        "Modelo de vínculos de March y Simon (1958)",
        "El empleado permanece mientras el equilibrio entre lo que aporta y lo que "
        "recibe le resulte favorable frente a sus alternativas externas. Justifica "
        "incluir el salario relativo y la antigüedad como variables predictoras.",
    ),
    (
        "Embeddedness organizacional (Mitchell et al., 2001)",
        "Cuanto más arraigado está alguien —vínculos, ajuste al puesto, coste de "
        "salir— menor es su probabilidad de irse. La antigüedad y las promociones "
        "operan como indicadores indirectos de ese arraigo.",
    ),
    (
        "Modelo de demandas y recursos laborales (JD-R, 2001)",
        "Demandas sostenidas sin recursos que las compensen producen desgaste y, "
        "finalmente, salida. Las horas trabajadas por semana son el indicador de "
        "demanda dentro de este modelo.",
    ),
]

# --------------------------------------------------------------------------- #
# Tabla de operacionalización de variables
# --------------------------------------------------------------------------- #
ENCABEZADOS_OPERACIONALIZACION = [
    "Variable",
    "Tipo",
    "Definición operacional",
    "Escala de medida",
    "Rango / categorías",
    "Indicador",
    "Rol en el modelo",
]

OPERACIONALIZACION = [
    [
        "edad",
        "Cuantitativa continua",
        "Años cumplidos del empleado en la fecha de corte del análisis.",
        "Razón",
        "21 – 60 años",
        "Fecha de nacimiento registrada en nómina",
        "Independiente",
    ],
    [
        "salario",
        "Cuantitativa continua",
        "Remuneración mensual bruta pactada en el contrato.",
        "Razón",
        "1.5M – 18M COP",
        "Valor de nómina mensual",
        "Independiente",
    ],
    [
        "anios_empresa",
        "Cuantitativa discreta",
        "Años completos transcurridos desde la fecha de vinculación.",
        "Razón",
        "0 – 35 años",
        "Antigüedad contractual",
        "Independiente",
    ],
    [
        "departamento",
        "Cualitativa nominal",
        "Área funcional a la que está adscrito el cargo.",
        "Nominal",
        "7 categorías (Ventas, Tecnología, Operaciones, Soporte, Marketing, "
        "Finanzas, Recursos Humanos)",
        "Estructura organizacional vigente",
        "Independiente (one-hot)",
    ],
    [
        "satisfaccion",
        "Cuantitativa continua",
        "Nivel de satisfacción laboral autodeclarado en la encuesta de clima.",
        "Intervalo (escala tipo Likert)",
        "1.0 (muy insatisfecho) – 5.0 (muy satisfecho)",
        "Promedio de ítems de la encuesta de clima",
        "Independiente",
    ],
    [
        "horas_trabajadas",
        "Cuantitativa discreta",
        "Promedio de horas efectivamente trabajadas por semana.",
        "Razón",
        "35 – 70 horas",
        "Registro de marcación / control horario",
        "Independiente",
    ],
    [
        "promociones",
        "Cuantitativa discreta",
        "Número de ascensos o cambios de nivel obtenidos en la empresa.",
        "Razón",
        "0 – 6 promociones",
        "Historial de movimientos de cargo",
        "Independiente",
    ],
    [
        "abandono",
        "Cualitativa dicotómica",
        "Indica si el empleado dejó la organización durante el periodo observado.",
        "Nominal binaria",
        "1 = abandonó · 0 = permanece",
        "Novedad de retiro en nómina",
        "Dependiente (objetivo)",
    ],
]

# Fórmula de la regresión logística en LaTeX (renderizada con dcc.Markdown)
FORMULA = r"""
$$P(\text{abandono}=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k)}}$$

$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{j=1}^{k} \beta_j x_j
\qquad\qquad OR_j = e^{\beta_j}$$
"""


def layout() -> html.Div:
    """Layout de la pestaña de marco teórico."""
    return html.Div(
        [
            page_header(
                "Marco teórico",
                "Conceptos, teorías de referencia y traducción de cada constructo "
                "a una variable medible.",
                "📚",
            ),
            section_title("Conceptos clave"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(definicion),
                            titulo=concepto,
                            color=SERIES[i % len(SERIES)],
                        ),
                        lg=6,
                        className="mb-3",
                    )
                    for i, (concepto, definicion) in enumerate(CONCEPTOS)
                ],
                className="g-3 mb-4",
            ),
            section_title("Teorías de referencia"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(descripcion),
                            titulo=teoria,
                            color=SERIES[(i + 2) % len(SERIES)],
                        ),
                        lg=6,
                        className="mb-3",
                    )
                    for i, (teoria, descripcion) in enumerate(TEORIAS)
                ],
                className="g-3 mb-4",
            ),
            section_title("Fundamento estadístico: regresión logística"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "La variable dependiente es dicotómica, por lo que la "
                                    "regresión lineal no aplica: produciría probabilidades "
                                    "fuera del intervalo [0, 1] y violaría el supuesto de "
                                    "homocedasticidad. La regresión logística resuelve el "
                                    "problema modelando el logaritmo de las odds como una "
                                    "combinación lineal de los predictores."
                                ),
                                dcc.Markdown(
                                    FORMULA,
                                    mathjax=True,
                                    className="formula-block",
                                ),
                                paragraph(
                                    "Al exponenciar un coeficiente se obtiene su razón de "
                                    "odds (OR): un OR mayor que 1 indica que la variable "
                                    "incrementa las probabilidades de abandono, y menor que "
                                    "1, que las reduce, manteniendo el resto constante."
                                ),
                            ],
                            titulo="Especificación del modelo",
                            color=COLOR_PERMANECE,
                        ),
                        lg=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        card(
                            [
                                bullet_list(
                                    [
                                        "Independencia de las observaciones.",
                                        "Relación lineal entre los predictores y el logit.",
                                        "Ausencia de multicolinealidad severa.",
                                        "Tamaño de muestra suficiente por categoría.",
                                        "Ausencia de valores atípicos con alta influencia.",
                                    ],
                                    color=COLOR_ABANDONA,
                                ),
                                callout(
                                    "Se eligió regresión logística por encima de modelos de "
                                    "mayor capacidad (bosques aleatorios, gradient boosting) "
                                    "porque en gestión humana la explicabilidad de la decisión "
                                    "es un requisito, no una preferencia.",
                                    titulo="Criterio de selección",
                                    color=COLOR_ABANDONA,
                                ),
                            ],
                            titulo="Supuestos del modelo",
                            color=COLOR_ABANDONA,
                        ),
                        lg=5,
                        className="mb-3",
                    ),
                ]
            ),
            section_title("Operacionalización de variables"),
            card(
                data_table(ENCABEZADOS_OPERACIONALIZACION, OPERACIONALIZACION),
                subtitulo=(
                    "Cada constructo teórico se traduce a una variable observable, "
                    "con su escala de medida, rango e indicador de recolección."
                ),
                color=SERIES[3],
            ),
        ],
        className="tab-content",
    )
