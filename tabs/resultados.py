"""
Pestaña 7 · Resultados
Dos bloques: el análisis exploratorio (con selector de conjunto de datos) y el
desempeño del clasificador sobre el conjunto de prueba reservado.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

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
    AMBITO_NOMBRES,
    asociacion_con_grado,
    load_metrics,
    prueba_edad_por_grado,
    tamanos_particion,
)
from utils.figures import (
    fig_asociacion_grado,
    fig_boxplot_edad,
    fig_clinica_por_grado,
    fig_coeficientes,
    fig_distribucion_grado,
    fig_edad_por_grado,
    fig_matriz_confusion,
    fig_matriz_genes,
    fig_prevalencia_genes,
    fig_roc,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

ID_AMBITO = "res-ambito"
ID_EDA = "res-eda"


def _kpis() -> list[dict]:
    """Métricas de desempeño del modelo sobre el conjunto de prueba."""
    metricas = load_metrics()
    return [
        {
            "valor": f"{metricas['roc_auc']:.3f}",
            "etiqueta": "AUC-ROC",
            "detalle": f"CV {metricas['cv_auc_media']:.3f} ± {metricas['cv_auc_desviacion']:.3f}",
            "color": SERIES[0],
        },
        {
            "valor": f"{metricas['accuracy']:.1%}",
            "etiqueta": "Accuracy",
            "detalle": f"{metricas['n_test']} pacientes de prueba",
            "color": SERIES[2],
        },
        {
            "valor": f"{metricas['recall']:.1%}",
            "etiqueta": "Recall de GBM",
            "detalle": "Casos agresivos detectados",
            "color": SERIES[1],
        },
        {
            "valor": f"{metricas['precision']:.1%}",
            "etiqueta": "Precisión de GBM",
            "detalle": f"F1-score {metricas['f1']:.3f}",
            "color": SERIES[3],
        },
    ]


def _selector_ambito() -> html.Div:
    """Selector del conjunto de datos que describen los gráficos del EDA."""
    tamanos = tamanos_particion()
    return html.Div(
        [
            html.Span("Conjunto de datos:", className="control-label me-3"),
            dcc.RadioItems(
                id=ID_AMBITO,
                options=[
                    {
                        "label": f" Entrenamiento ({tamanos['train']})",
                        "value": "train",
                    },
                    {"label": f" Prueba ({tamanos['test']})", "value": "test"},
                    {"label": f" Completo ({tamanos['full']})", "value": "full"},
                ],
                value="train",
                inline=True,
                className="ambito-radio",
                inputClassName="me-1",
                labelClassName="me-4",
            ),
        ],
        className="filter-row",
    )


def _tabla_asociacion(ambito: str) -> dbc.Table:
    """Las diez variables más asociadas al grado, con su prueba estadística."""
    datos = asociacion_con_grado(ambito).head(10)
    filas = []
    for _, fila in datos.iterrows():
        v_cramer = "—" if np.isnan(fila["v_cramer"]) else f"{fila['v_cramer']:.3f}"
        p_valor = "< 0,0001" if fila["p_valor"] < 0.0001 else f"{fila['p_valor']:.4f}"
        filas.append(
            [
                fila["variable"],
                fila["tipo"],
                f"{fila['rho']:+.3f}",
                v_cramer,
                p_valor,
                "LGG" if fila["rho"] < 0 else "GBM",
            ]
        )
    return data_table(
        ["Variable", "Tipo", "Spearman r", "V de Cramér", "p-valor", "Empuja hacia"],
        filas,
    )


def _bloque_eda(ambito: str) -> html.Div:
    """Conjunto completo de figuras del EDA para el ámbito seleccionado."""
    prueba_edad = prueba_edad_por_grado(ambito)
    p_edad = (
        "p < 0,0001" if prueba_edad["p_valor"] < 0.0001
        else f"p = {prueba_edad['p_valor']:.4f}"
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_distribucion_grado(ambito),
                            "Distribución de la variable objetivo",
                            f"Composición del {AMBITO_NOMBRES[ambito]}.",
                        ),
                        lg=4,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_edad_por_grado(ambito),
                            "Edad al diagnóstico por grado",
                            f"Prueba U de Mann-Whitney: {p_edad}.",
                        ),
                        lg=8,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_boxplot_edad(ambito),
                            "Dispersión de la edad",
                            "Los puntos son atípicos dentro de cada grado, no en la "
                            "distribución global.",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_clinica_por_grado("Gender", ambito),
                            "Género dentro de cada grado",
                            "Predominio masculino en ambos grados, más marcado en GBM; la "
                            "prueba chi-cuadrado no encuentra asociación significativa.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_clinica_por_grado("Race", ambito),
                            "Grupo racial reportado dentro de cada grado",
                            "Más de nueve de cada diez pacientes son del grupo White: las "
                            "categorías minoritarias tienen muy pocas observaciones.",
                        ),
                        lg=12,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Mutaciones: prevalencia y capacidad discriminativa"),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_prevalencia_genes(ambito, top_n=12),
                            "Prevalencia de mutación por gen y grado",
                            "Los doce genes que más separan a los dos grupos.",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_asociacion_grado(ambito, top_n=14),
                            "Asociación de cada variable con el grado",
                            "Signo negativo = empuja hacia LGG. Las barras rayadas no "
                            "alcanzan significancia estadística (p ≥ 0,05 o |r| < 0,10).",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                ]
            ),
            card(
                _tabla_asociacion(ambito),
                titulo="Las diez variables más asociadas al grado",
                subtitulo=f"{AMBITO_NOMBRES[ambito].capitalize()} · Spearman, V de Cramér y "
                          "chi-cuadrado",
                color=SERIES[3],
                className="mb-4",
            ),
            section_title("Multicolinealidad entre mutaciones"),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_matriz_genes(ambito, top_n=12),
                            "V de Cramér entre pares de genes",
                            "La diagonal se omite (vale 1 por definición).",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "La matriz responde a una pregunta concreta: ¿hay pares "
                                    "de mutaciones tan redundantes que no convenga meterlas "
                                    "juntas en un modelo lineal?"
                                ),
                                bullet_list(
                                    [
                                        "ATRX-TP53 es la co-ocurrencia más fuerte "
                                        "(V ≈ 0,54): el eje clásico de co-mutación en la "
                                        "astrocitogénesis.",
                                        "FUBP1-CIC (V ≈ 0,46) y ATRX-IDH1 (V ≈ 0,46) "
                                        "reafirman las firmas del linaje oligodendroglial y "
                                        "de los gliomas de bajo grado.",
                                        "PTEN-IDH1 (V ≈ 0,40) refleja exclusión mutua: "
                                        "IDH1 mutado caracteriza LGG, PTEN alterado "
                                        "caracteriza GBM.",
                                    ],
                                    color=COLOR_LGG,
                                ),
                                callout(
                                    "Ningún par supera el umbral crítico de redundancia "
                                    "(V > 0,70), así que no hay multicolinealidad "
                                    "estructural: el subconjunto de marcadores puede "
                                    "retenerse completo sin inestabilidad en la estimación "
                                    "de los coeficientes.",
                                    titulo="Conclusión del diagnóstico",
                                    color=SERIES[2],
                                ),
                            ],
                            titulo="Cómo se lee la matriz",
                            color=SERIES[2],
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
        ]
    )


def _bloque_modelo() -> html.Div:
    """Desempeño del clasificador sobre el conjunto de prueba."""
    metricas = load_metrics()
    cm = metricas["matriz_confusion"]
    verdaderos_lgg, falsos_gbm = cm[0]
    falsos_lgg, verdaderos_gbm = cm[1]

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_matriz_confusion(),
                            "Matriz de confusión",
                            f"{falsos_lgg} falsos negativos (GBM no detectados) y "
                            f"{falsos_gbm} falsos positivos (LGG marcados como GBM).",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_roc(),
                            "Curva ROC",
                            "La curva se aproxima a la esquina superior izquierda: el "
                            "modelo discrimina bien en todo el rango de umbrales, no solo "
                            "en 0,50.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_coeficientes(top_n=12),
                            "Coeficientes del modelo y odds ratios",
                            f"Solo {metricas['n_variables_activas']} de "
                            f"{metricas['n_columnas_modelo']} columnas sobreviven a la "
                            "penalización L1; el resto tiene coeficiente exactamente cero.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    f"Con {verdaderos_gbm} de {verdaderos_gbm + falsos_lgg} "
                                    "glioblastomas detectados, el modelo prioriza no dejar "
                                    "pasar los casos agresivos, a costa de marcar de más "
                                    f"({falsos_gbm} pacientes con LGG clasificados como "
                                    "GBM). En un contexto clínico es el intercambio "
                                    "razonable: es preferible enviar un caso adicional a "
                                    "revisión que omitir un tumor agresivo."
                                ),
                                bullet_list(
                                    [
                                        "IDH1 mutado es el factor protector más fuerte: "
                                        "reduce drásticamente las probabilidades de GBM.",
                                        "IDH2 refuerza la misma señal biológica, al ser "
                                        "una isoforma de la misma enzima.",
                                        "TP53 y PTEN empujan hacia GBM, igual que la edad.",
                                        "La mayoría de los genes queda en cero: el modelo "
                                        "reduce por sí solo el panel necesario.",
                                    ],
                                    color=COLOR_GBM,
                                ),
                                callout(
                                    f"La diferencia entre el AUC de validación cruzada "
                                    f"({metricas['cv_auc_media']:.3f}) y el de prueba "
                                    f"({metricas['roc_auc']:.3f}) es pequeña: no hay "
                                    "indicios de sobreajuste.",
                                    titulo="Estabilidad",
                                    color=SERIES[2],
                                ),
                            ],
                            titulo="Lectura de los resultados",
                            color=COLOR_GBM,
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
        ]
    )


def layout() -> html.Div:
    """Layout de la pestaña de resultados."""
    return html.Div(
        [
            page_header(
                "Resultados",
                "Primero qué dicen los datos, después qué aprende el modelo de ellos.",
                "📊",
            ),
            kpi_row(_kpis()),
            section_title("Análisis exploratorio"),
            callout(
                "El EDA se describe por defecto sobre el conjunto de entrenamiento, igual "
                "que en el libro: mirar el conjunto de prueba antes de evaluar el modelo "
                "sería filtrar información. El selector permite comprobar que la "
                "estratificación mantuvo la misma estructura en todas las particiones.",
                titulo="Por qué el selector empieza en entrenamiento",
                color=COLOR_LGG,
            ),
            _selector_ambito(),
            html.Div(series_chips(), className="mb-3"),
            dcc.Loading(
                html.Div(_bloque_eda("train"), id=ID_EDA),
                type="dot",
                color=COLOR_LGG,
            ),
            section_title("Desempeño del clasificador"),
            paragraph(
                "Todas las cifras de esta sección provienen del conjunto de prueba "
                "reservado, que el modelo no vio durante el entrenamiento ni durante la "
                "búsqueda de hiperparámetros."
            ),
            _bloque_modelo(),
        ],
        className="tab-content",
    )


# --------------------------------------------------------------------------- #
# Callbacks propios de la pestaña
# --------------------------------------------------------------------------- #
@callback(Output(ID_EDA, "children"), Input(ID_AMBITO, "value"))
def actualizar_eda(ambito: str):
    """Reconstruye las figuras del EDA con el conjunto de datos seleccionado."""
    return _bloque_eda(ambito or "train")
