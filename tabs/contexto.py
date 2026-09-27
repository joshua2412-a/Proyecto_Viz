"""
Pestaña 2 · Contexto
Traduce la rotación laboral a impacto empresarial: coste, productividad y
consecuencias organizativas.
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
from utils.data_loader import get_dataframe, tasa_por_departamento
from utils.figures import fig_costo_departamento
from utils.theme import COLOR_ABANDONA, COLOR_PERMANECE, SERIES

# Supuesto de coste: reemplazar a alguien cuesta ~6 meses de su salario
MESES_REEMPLAZO = 6

# Dimensiones de impacto (cada una con su explicación)
IMPACTOS = [
    (
        "💸",
        "Coste directo de reemplazo",
        "Publicación de vacantes, tiempo de selección, contratación, formación "
        "inicial y curva de aprendizaje hasta alcanzar el rendimiento pleno.",
    ),
    (
        "📉",
        "Pérdida de productividad",
        "La vacante abierta redistribuye la carga en el equipo: aumentan las "
        "horas extra, se retrasan entregas y sube el riesgo de una segunda salida.",
    ),
    (
        "🧠",
        "Fuga de conocimiento",
        "El conocimiento operativo no documentado —clientes, criterios, atajos— "
        "sale por la puerta con la persona y rara vez se recupera.",
    ),
    (
        "🤝",
        "Impacto en clientes",
        "En áreas de cara al cliente, la rotación rompe la continuidad de la "
        "relación comercial y afecta directamente a la experiencia percibida.",
    ),
    (
        "🌡️",
        "Clima organizacional",
        "Las salidas frecuentes se leen como señal de alarma interna y erosionan "
        "el compromiso de quienes se quedan.",
    ),
    (
        "🏷️",
        "Marca empleadora",
        "Una reputación de alta rotación encarece y alarga cada nueva "
        "contratación en el mercado laboral.",
    ),
]


def _kpis_costo() -> list[dict]:
    """Indicadores económicos derivados del dataset."""
    df = get_dataframe()
    salidas = df[df["abandono"] == 1]
    costo_total = salidas["salario"].sum() * MESES_REEMPLAZO
    costo_promedio = salidas["salario"].mean() * MESES_REEMPLAZO
    nomina_anual = df["salario"].sum() * 12

    return [
        {
            "valor": f"{int(len(salidas)):,}",
            "etiqueta": "Salidas en el periodo",
            "detalle": f"de {len(df):,} empleados",
            "color": SERIES[1],
        },
        {
            "valor": f"${costo_total / 1_000_000_000:,.1f}MM",
            "etiqueta": "Coste estimado de la rotación",
            "detalle": f"{MESES_REEMPLAZO} meses de salario por salida",
            "color": SERIES[1],
        },
        {
            "valor": f"${costo_promedio / 1_000_000:,.1f}M",
            "etiqueta": "Coste por salida",
            "detalle": "Promedio en millones de COP",
            "color": SERIES[3],
        },
        {
            "valor": f"{costo_total / nomina_anual:.1%}",
            "etiqueta": "Equivalente de la nómina anual",
            "detalle": "Porción de nómina consumida por la rotación",
            "color": SERIES[0],
        },
    ]


def _tabla_departamentos() -> dbc.Table:
    """Tabla comparativa con el perfil económico de cada departamento."""
    resumen = tasa_por_departamento()
    filas = [
        [
            fila["departamento"],
            f"{int(fila['empleados']):,}",
            f"{int(fila['abandonos']):,}",
            f"{fila['tasa']:.1%}",
            f"${fila['salario_promedio'] / 1_000_000:,.2f}M",
            f"{fila['satisfaccion_promedio']:.2f}",
        ]
        for _, fila in resumen.iterrows()
    ]
    return data_table(
        [
            "Departamento",
            "Empleados",
            "Salidas",
            "Tasa de abandono",
            "Salario promedio",
            "Satisfacción promedio",
        ],
        filas,
    )


def layout() -> html.Div:
    """Layout de la pestaña de contexto."""
    return html.Div(
        [
            page_header(
                "Contexto e impacto empresarial",
                "La rotación no es una métrica de recursos humanos: es una línea "
                "del estado de resultados.",
                "🏢",
            ),
            kpi_row(_kpis_costo()),
            callout(
                f"Los importes asumen que reemplazar a una persona cuesta "
                f"{MESES_REEMPLAZO} meses de su salario, un supuesto conservador "
                "dentro del rango que suele citar la literatura de gestión humana "
                "(entre 6 y 12 meses según la complejidad del puesto). El objetivo "
                "no es la cifra exacta, sino el orden de magnitud del problema.",
                titulo="Supuesto de cálculo",
                color=SERIES[3],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_costo_departamento(MESES_REEMPLAZO),
                            "Coste estimado de la rotación por departamento",
                            "Suma de los salarios de quienes abandonaron, "
                            f"multiplicada por {MESES_REEMPLAZO} meses.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "El coste no se reparte por igual. Las áreas con "
                                    "salarios altos generan un impacto económico grande "
                                    "incluso con tasas moderadas, mientras que las áreas "
                                    "operativas lo generan por volumen de salidas."
                                ),
                                paragraph(
                                    "Esa distinción importa para decidir dónde intervenir: "
                                    "en un caso conviene retener perfiles concretos; en el "
                                    "otro, revisar condiciones estructurales del puesto."
                                ),
                                bullet_list(
                                    [
                                        "Volumen alto + salario bajo → revisar condiciones del puesto.",
                                        "Volumen bajo + salario alto → plan de retención individual.",
                                        "Volumen alto + salario alto → prioridad máxima.",
                                    ],
                                    color=COLOR_ABANDONA,
                                ),
                            ],
                            titulo="Cómo leer el gráfico",
                            color=COLOR_ABANDONA,
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Seis formas en que la rotación afecta al negocio"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(icono, className="guide-icon"),
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
                    for i, (icono, titulo, texto) in enumerate(IMPACTOS)
                ],
                className="g-3 mb-4",
            ),
            section_title("Radiografía por departamento"),
            card(
                _tabla_departamentos(),
                subtitulo="Ordenado de mayor a menor tasa de abandono.",
                color=COLOR_PERMANECE,
            ),
        ],
        className="tab-content",
    )
