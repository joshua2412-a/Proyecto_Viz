"""
Pestaña 8 · Resultados
El análisis exploratorio completo, con un selector que permite recorrer el
mismo conjunto de figuras sobre entrenamiento, prueba o el dataset entero.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from utils.formato import num, pct
from utils.components import (
    bullet_list,
    lectura_guiada,
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
    lectura_asociacion,
    lectura_clinicas,
    lectura_cohorte,
    lectura_genes,
    lectura_multicolinealidad,
    estadisticas_edad,
    prevalencia_genes,
    proporcion_grado,
    prueba_edad_por_grado,
    tamanos_particion,
    vif_predictoras,
)
from utils.figures import (
    fig_asociacion_grado,
    fig_boxplot_edad,
    fig_clinica_por_grado,
    fig_distribucion_grado,
    fig_edad_por_grado,
    fig_matriz_genes,
    fig_prevalencia_genes,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

ID_AMBITO = "res-ambito"
ID_EDA = "res-eda"


def _kpis() -> list[dict]:
    """Los cuatro hallazgos que resumen el análisis, sobre entrenamiento."""
    asociacion = asociacion_con_grado("train")
    edad = estadisticas_edad("train").set_index("grade_label")
    idh1 = prevalencia_genes("train").set_index("gen").loc["IDH1"]
    proporciones = proporcion_grado("train").set_index("grado")

    mas_fuerte = asociacion.iloc[0]

    return [
        {
            "valor": f"{num(mas_fuerte['rho'], 2, signo=True)}",
            "etiqueta": f"Asociación de {mas_fuerte['variable']}",
            "detalle": "El marcador con mayor capacidad discriminativa",
        },
        {
            "valor": f"{num(edad.loc['GBM', 'media'] - edad.loc['LGG', 'media'], 1)} años",
            "etiqueta": "Diferencia de edad",
            "color": f"linear-gradient(180deg, {COLOR_LGG} 0 50%, {COLOR_GBM} 50% 100%)",
            "detalle": f"GBM {num(edad.loc['GBM', 'media'], 1)} frente a LGG "
                       f"{num(edad.loc['LGG', 'media'], 1)}",
        },
        {
            "valor": f"{pct(idh1['LGG'], 0)} vs {pct(idh1['GBM'], 0)}",
            "etiqueta": "IDH1 mutado: LGG vs GBM",
            "detalle": "Prevalencia de la mutación en cada grado",
            "color": f"linear-gradient(180deg, {COLOR_LGG} 0 50%, {COLOR_GBM} 50% 100%)",
        },
        {
            "valor": f"{int(asociacion['significativa'].sum())} de {len(asociacion)}",
            "etiqueta": "Variables con señal",
            "detalle": f"Sobre {int(proporciones['pacientes'].sum())} pacientes de entrenamiento",
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
    """Las diez variables más asociadas al grado, con su prueba estadística.

    A las diez primeras por |ρ| se les añaden las nominales, que no tienen ρ
    con el que ordenarse y si no se quedarían fuera de la tabla sin aparecer
    en ningún otro sitio de esta pestaña.
    """
    todas = asociacion_con_grado(ambito)
    con_signo = todas[todas["rho"].notna()].head(10)
    nominales = todas[todas["rho"].isna()]
    datos = pd.concat([con_signo, nominales])
    filas = []
    for _, fila in datos.iterrows():
        v_cramer = "—" if np.isnan(fila["v_cramer"]) else f"{num(fila['v_cramer'], 3)}"
        p_valor = "< 0,0001" if fila["p_valor"] < 0.0001 else f"{num(fila['p_valor'], 4)}"
        filas.append(
            [
                fila["variable"],
                fila["prueba"],
                f"{num(fila['rho'], 3, signo=True)}",
                v_cramer,
                p_valor,
                "Sí" if fila["significativa_bonferroni"] else "No",
                "—" if np.isnan(fila["rho"]) else ("LGG" if fila["rho"] < 0 else "GBM"),
            ]
        )
    return data_table(
        ["Variable", "Prueba", "Spearman r", "V de Cramér", "p-valor",
         "Aguanta Bonferroni", "Empuja hacia"],
        filas,
    )


def _bloque_eda(ambito: str) -> html.Div:
    """Conjunto completo de figuras del EDA para el ámbito seleccionado."""
    prueba_edad = prueba_edad_por_grado(ambito)
    p_edad = (
        "p < 0,0001" if prueba_edad["p_valor"] < 0.0001
        else f"p = {num(prueba_edad['p_valor'], 4)}"
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
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(
                            lectura_cohorte(ambito),
                            titulo="Lectura · la cohorte y la edad",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_clinica_por_grado("Gender", ambito),
                            "Género dentro de cada grado",
                            "Predominio masculino en ambos grados, más marcado en GBM.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(
                            lectura_clinicas(ambito),
                            titulo="Lectura · las variables demográficas",
                        ),
                        lg=5,
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
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(
                            lectura_genes(ambito),
                            titulo="Lectura · el panel de mutaciones",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_asociacion_grado(ambito, top_n=14),
                            "Asociación de cada variable con el grado",
                            "Signo negativo = empuja hacia LGG. Las barras rayadas no "
                            "alcanzan significancia estadística (p ≥ 0,05 o |r| < 0,10). "
                            "El grupo racial no aparece: es nominal y no tiene signo, "
                            "así que su asociación va por V de Cramér en la tabla.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(
                            lectura_asociacion(ambito),
                            titulo="Lectura · qué merece la pena medir",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            card(
                _tabla_asociacion(ambito),
                titulo="Las variables más asociadas al grado",
                subtitulo=f"{AMBITO_NOMBRES[ambito].capitalize()} · las diez de mayor "
                          "|ρ|, más el grupo racial, que es nominal y va por V de "
                          "Cramér",
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
                                    "juntas en un modelo lineal? Solo a esa: la "
                                    "multicolinealidad completa se mide más abajo, con el "
                                    "factor de inflación de la varianza."
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
                                ),
                                callout(
                                    "Ningún par supera el umbral de redundancia "
                                    "(V > 0,70). Eso descarta la redundancia por parejas, "
                                    "que no es lo mismo que descartar la "
                                    "multicolinealidad: una variable puede ser casi "
                                    "predecible a partir de varias a la vez sin parecerse "
                                    "a ninguna por separado.",
                                    titulo="Qué descarta esto, y qué no",
                                ),
                            ],
                            titulo="Cómo se lee la matriz",
                            tono="acento",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Multicolinealidad completa: factor de inflación de la varianza"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            _tabla_vif(ambito),
                            titulo="Las cinco predictoras con más VIF",
                            subtitulo="VIF = 1 / (1 − R²) de regresar cada predictora "
                                      "contra las otras 22",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(
                            lectura_multicolinealidad(ambito),
                            titulo="Lectura · se pisan o no las predictoras",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
        ]
    )


def _tabla_vif(ambito: str) -> dbc.Table:
    """Las predictoras con mayor inflación de varianza.

    Se enseñan solo las cinco primeras: si la peor está holgada, las demás
    también, y una tabla de 23 filas para decir «no hay problema» sobra.
    """
    datos = vif_predictoras(ambito).head(5)
    return data_table(
        ["Predictora", "R² contra las demás", "VIF", "Lectura"],
        [
            [
                fila["variable"],
                num(fila["r2"], 3),
                num(fila["vif"], 2),
                "Sin problema" if fila["vif"] < 5
                else ("Conviene mirarlo" if fila["vif"] < 10 else "Inestable"),
            ]
            for _, fila in datos.iterrows()
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de resultados."""
    return html.Div(
        [
            page_header(
                "Resultados",
                "Qué dicen los datos: distribuciones, contrastes y fuerza de cada "
                "asociación con el grado tumoral.",
                "bi-bar-chart-line",
            ),
            kpi_row(_kpis()),
            section_title("Análisis exploratorio"),
            callout(
                "El EDA se describe por defecto sobre el conjunto de entrenamiento, igual "
                "que en el libro: mirar el conjunto de prueba antes de evaluar el modelo "
                "sería filtrar información. El selector permite comprobar que la "
                "estratificación mantuvo la misma estructura en todas las particiones.",
                titulo="Por qué el selector empieza en entrenamiento",
            ),
            _selector_ambito(),
            html.Div(series_chips(), className="mb-3"),
            dcc.Loading(
                html.Div(_bloque_eda("train"), id=ID_EDA),
                type="dot",
            ),
            callout(
                "El modelado —la traducción de estos hallazgos en un clasificador— es el "
                "paso natural a partir de aquí, con el conjunto de prueba ya reservado "
                "para evaluarlo. Queda fuera del alcance de esta entrega.",
                titulo="Qué viene después",
            ),
        ],
        className="vista-pestana",
    )


# --------------------------------------------------------------------------- #
# Callbacks propios de la pestaña
# --------------------------------------------------------------------------- #
@callback(Output(ID_EDA, "children"), Input(ID_AMBITO, "value"))
def actualizar_eda(ambito: str):
    """Reconstruye las figuras del EDA con el conjunto de datos seleccionado."""
    return _bloque_eda(ambito or "train")
