"""
Pestaña 10 · Conclusiones
Hallazgos, verificación de objetivos y recomendaciones accionables.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    kpi_row,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import get_dataframe, load_metrics, tasa_por_departamento
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# Recomendaciones operativas por horizonte temporal
RECOMENDACIONES = [
    (
        "Inmediato (0-30 días)",
        [
            "Calcular el riesgo de toda la plantilla y ordenarla por probabilidad.",
            "Abrir conversaciones de retención con el decil de mayor riesgo.",
            "Auditar la carga semanal en las áreas que superan la media global.",
        ],
    ),
    (
        "Corto plazo (1-3 meses)",
        [
            "Revisar la equidad salarial interna en los departamentos más críticos.",
            "Publicar rutas de promoción explícitas para los cargos con rotación alta.",
            "Incorporar la satisfacción medida al tablero de seguimiento mensual.",
        ],
    ),
    (
        "Mediano plazo (3-12 meses)",
        [
            "Integrar el modelo con el sistema de nómina para un cálculo automático.",
            "Registrar el histórico de encuestas para habilitar análisis longitudinal.",
            "Medir el efecto de las intervenciones sobre la tasa de abandono real.",
            "Reentrenar el modelo con datos propios y auditar sesgos por grupo.",
        ],
    ),
]


def _hallazgos() -> list[tuple[str, str, str]]:
    """Hallazgos redactados a partir de los datos y las métricas reales."""
    df = get_dataframe()
    resumen = tasa_por_departamento()
    metricas = load_metrics()

    peor = resumen.iloc[0]
    mejor = resumen.iloc[-1]
    sat_sale = df.loc[df["abandono"] == 1, "satisfaccion"].mean()
    sat_queda = df.loc[df["abandono"] == 0, "satisfaccion"].mean()
    horas_sale = df.loc[df["abandono"] == 1, "horas_trabajadas"].mean()
    horas_queda = df.loc[df["abandono"] == 0, "horas_trabajadas"].mean()
    sal_sale = df.loc[df["abandono"] == 1, "salario"].mean()
    sal_queda = df.loc[df["abandono"] == 0, "salario"].mean()
    ant_sale = df.loc[df["abandono"] == 1, "anios_empresa"].mean()
    ant_queda = df.loc[df["abandono"] == 0, "anios_empresa"].mean()

    return [
        (
            "1",
            "El abandono está concentrado, no repartido",
            f"{peor['departamento']} pierde {peor['tasa']:.1%} de su plantilla frente al "
            f"{mejor['tasa']:.1%} de {mejor['departamento']}. Una política única para toda "
            "la empresa gastaría presupuesto donde no hace falta y dejaría descubierta "
            "el área crítica.",
        ),
        (
            "2",
            "La satisfacción es el predictor dominante",
            f"Quienes abandonan declaran {sat_sale:.2f} sobre 5 frente a {sat_queda:.2f} de "
            "quienes permanecen. Es la variable con el coeficiente de mayor magnitud del "
            "modelo, y además es medible con instrumentos que la empresa ya aplica.",
        ),
        (
            "3",
            "La sobrecarga sostenida empuja la salida",
            f"El grupo que abandona trabaja {horas_sale:.1f} horas semanales frente a "
            f"{horas_queda:.1f} del que permanece. La diferencia es moderada en promedio, "
            "pero sistemática y acumulativa.",
        ),
        (
            "4",
            "El salario protege, pero no por sí solo",
            f"El salario promedio de quienes se van es de "
            f"${sal_sale / 1_000_000:,.2f}M frente a ${sal_queda / 1_000_000:,.2f}M. "
            "Su coeficiente es relevante, aunque menor que el de la satisfacción: subir "
            "sueldos sin corregir carga ni desarrollo rinde poco.",
        ),
        (
            "5",
            "La antigüedad y las promociones generan arraigo",
            f"Quienes permanecen acumulan {ant_queda:.1f} años frente a {ant_sale:.1f} de "
            "quienes salen. El primer tramo de la relación laboral es el más frágil: es "
            "donde el acompañamiento rinde más.",
        ),
        (
            "6",
            "El modelo es suficientemente bueno para priorizar",
            f"AUC de {metricas['roc_auc']:.3f} y recall de {metricas['recall']:.1%}: el "
            "modelo detecta la mayoría de las salidas reales. No es una bola de cristal, "
            "pero ordena la lista de a quién llamar primero mucho mejor que el azar.",
        ),
    ]


def _kpis_cierre() -> list[dict]:
    """Cifras de cierre del proyecto."""
    metricas = load_metrics()
    df = get_dataframe()
    return [
        {
            "valor": f"{df['abandono'].mean():.1%}",
            "etiqueta": "Tasa de abandono analizada",
            "detalle": f"{len(df):,} empleados",
            "color": SERIES[1],
        },
        {
            "valor": f"{metricas['roc_auc']:.3f}",
            "etiqueta": "AUC alcanzado",
            "detalle": "Capacidad de discriminación",
            "color": SERIES[0],
        },
        {
            "valor": f"{metricas['recall']:.1%}",
            "etiqueta": "Abandonos detectados",
            "detalle": "Recall en datos de prueba",
            "color": SERIES[2],
        },
        {
            "valor": "7",
            "etiqueta": "Variables utilizadas",
            "detalle": "Todas disponibles en nómina",
            "color": SERIES[3],
        },
    ]


def _tabla_objetivos() -> dbc.Table:
    """Verificación del cumplimiento de los objetivos específicos."""
    filas = [
        ["1. Construir la base de datos", "Cumplido", "data/generate_data.py · 2.000 registros"],
        ["2. Caracterizar el fenómeno", "Cumplido", "Pestaña Resultados · bloque exploratorio"],
        ["3. Entrenar el modelo", "Cumplido", "model/train_model.py · pipeline serializado"],
        ["4. Evaluar el desempeño", "Cumplido", "Accuracy, precisión, recall, F1, AUC y matriz"],
        ["5. Identificar factores", "Cumplido", "Coeficientes y razones de odds del modelo"],
        ["6. Entregar un simulador", "Cumplido", "Pestaña Predicción · formulario en tiempo real"],
    ]
    return data_table(["Objetivo específico", "Estado", "Evidencia"], filas)


def layout() -> html.Div:
    """Layout de la pestaña de conclusiones."""
    return html.Div(
        [
            page_header(
                "Conclusiones y recomendaciones",
                "Qué muestran los datos, qué puede hacerse con ello y en qué orden.",
                "✅",
            ),
            kpi_row(_kpis_cierre()),
            section_title("Hallazgos principales"),
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
                    for i, (numero, titulo, texto) in enumerate(_hallazgos())
                ],
                className="g-3 mb-4",
            ),
            section_title("Recomendaciones por horizonte"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            bullet_list(acciones, color=SERIES[i % len(SERIES)]),
                            titulo=horizonte,
                            color=SERIES[i % len(SERIES)],
                        ),
                        lg=4,
                        className="mb-3",
                    )
                    for i, (horizonte, acciones) in enumerate(RECOMENDACIONES)
                ],
                className="g-3 mb-4",
            ),
            section_title("Cumplimiento de objetivos"),
            card(
                _tabla_objetivos(),
                subtitulo="Cada objetivo específico con la evidencia que lo respalda "
                "dentro del proyecto.",
                color=COLOR_PERMANECE,
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(
                                "El proyecto demuestra que con siete variables que "
                                "cualquier área de gestión humana ya registra es posible "
                                "pasar de un indicador retrospectivo —cuántos se fueron— a "
                                "un instrumento de priorización —a quién conviene escuchar "
                                "esta semana—. El valor no está en la sofisticación del "
                                "modelo, sino en que su resultado es interpretable y "
                                "accionable por quien debe tomar la decisión."
                            ),
                            titulo="Conclusión general",
                            color=COLOR_PERMANECE,
                        ),
                        lg=7,
                        className="mb-3",
                    ),
                    dbc.Col(
                        callout(
                            "Una probabilidad alta es una invitación a conversar, no una "
                            "etiqueta sobre una persona. El modelo indica dónde mirar; la "
                            "decisión, el contexto y la responsabilidad siguen siendo "
                            "humanos.",
                            titulo="Principio de uso",
                            color=COLOR_ABANDONA,
                        ),
                        lg=5,
                        className="mb-3",
                    ),
                ]
            ),
        ],
        className="tab-content",
    )
