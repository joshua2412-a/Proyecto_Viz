"""
Pestaña 8 · Resultados
El análisis exploratorio completo, siempre sobre el conjunto de entrenamiento:
cruzar las predictoras con el grado en el conjunto reservado gastaría la
reserva que después hace falta para evaluar un modelo. La comparación entre
particiones, que es descripción y no inferencia, vive en Exploración.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import html

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
    lectura_correlaciones,
    lectura_genes,
    lectura_multicolinealidad,
    matriz_asociacion_genes,
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
    fig_matriz_spearman,
    fig_prevalencia_genes,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES


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


def _tabla_asociacion(ambito: str) -> dbc.Table:
    """Las diez variables más asociadas al grado, con su prueba estadística.

    Las diez primeras por |ρ|. El grupo racial ya no necesita trato aparte: el
    libro lo correlaciona reagrupado en dos niveles, así que tiene ρ como el
    resto y aparece en el gráfico de barras con signo, que llega hasta la
    decimocuarta.
    """
    datos = asociacion_con_grado(ambito).head(10)
    filas = []
    for _, fila in datos.iterrows():
        v_cramer = "—" if np.isnan(fila["v_cramer"]) else f"{num(fila['v_cramer'], 3)}"
        p_valor = "< 0,0001" if fila["p_valor"] < 0.0001 else f"{num(fila['p_valor'], 4)}"
        q_valor = "< 0,0001" if fila["q_valor"] < 0.0001 else f"{num(fila['q_valor'], 4)}"
        filas.append(
            [
                fila["variable"],
                fila["prueba"],
                f"{num(fila['rho'], 3, signo=True)}",
                v_cramer,
                p_valor,
                q_valor,
                "—" if np.isnan(fila["rho"]) else ("LGG" if fila["rho"] < 0 else "GBM"),
            ]
        )
    return data_table(
        ["Variable", "Prueba", "Spearman r", "V de Cramér", "p-valor",
         "q (BH)", "Empuja hacia"],
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
                            "El grupo racial aparece reagrupado en dos niveles, que es "
                            "como el libro lo correlaciona.",
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
                          "|ρ|",
                className="mb-4",
            ),
            section_title("Correlación entre todas las variables"),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_matriz_spearman(ambito),
                            "Matriz de Spearman del grado y las 23 predictoras",
                            "Solo el triángulo inferior: el superior es su reflejo. El "
                            "grupo racial entra reagrupado en dos niveles, igual que en "
                            "el libro.",
                        ),
                        lg=8,
                        className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(
                            lectura_correlaciones(ambito),
                            titulo="Lectura · el mapa completo",
                        ),
                        lg=4,
                        className="mb-4",
                    ),
                ]
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
                                        f"ATRX-TP53 es la co-ocurrencia más fuerte "
                                        f"(V = {_v_par(ambito, 'ATRX', 'TP53')}): el eje "
                                        "clásico de co-mutación en la astrocitogénesis.",
                                        f"FUBP1-CIC (V = {_v_par(ambito, 'FUBP1', 'CIC')}) "
                                        f"y ATRX-IDH1 (V = {_v_par(ambito, 'ATRX', 'IDH1')}) "
                                        "reafirman las firmas del linaje oligodendroglial y "
                                        "de los gliomas de bajo grado.",
                                        f"PTEN-IDH1 (V = {_v_par(ambito, 'PTEN', 'IDH1')}) "
                                        "refleja exclusión mutua: IDH1 mutado caracteriza "
                                        "LGG, PTEN alterado caracteriza GBM.",
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
            callout(
                "Las dos figuras de arriba miran pares, y la multicolinealidad no "
                "es un fenómeno por pares: una variable puede ser casi predecible "
                "a partir de una combinación de otras sin parecerse a ninguna por "
                "separado. El VIF responde a esa pregunta, y es el mismo cálculo "
                "que cierra la sección de multicolinealidad del libro.",
                titulo="Por qué no basta con mirar pares",
            ),
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


def _v_par(ambito: str, gen_a: str, gen_b: str) -> str:
    """V de Cramér de un par de genes, leída de la misma matriz que el mapa.

    Se calcula en vez de escribirse a mano porque ya nos pasó: al alinear la
    V con la del libro (sin corrección de Yates) las cifras se movieron en el
    segundo decimal y el texto se quedó diciendo las de antes.
    """
    matriz = matriz_asociacion_genes(ambito)
    return num(float(matriz.loc[gen_a, gen_b]), 2)


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
                f"Todo lo que sigue se calcula sobre el conjunto de entrenamiento "
                f"({tamanos_particion()['train']} pacientes), igual que en el libro. El resto se "
                "apartó antes de "
                "mirar nada y aquí no se cruza con el grado: si estas pruebas se "
                "corrieran también sobre él, la evaluación del modelo que venga después "
                "dejaría de ser independiente, porque las variables se habrían elegido "
                "sabiendo lo que ese conjunto decía. Para comprobar que la partición "
                "quedó repartida de forma parecida, la pestaña Exploración permite "
                "comparar la distribución de cada variable entre los tres conjuntos, "
                "sin cruzarla con el grado.",
                titulo="Sobre qué datos está hecho este análisis",
            ),
            html.Div(series_chips(), className="mb-3"),
            _bloque_eda("train"),
            callout(
                "El modelado —la traducción de estos hallazgos en un clasificador— es el "
                "paso natural a partir de aquí, con el conjunto de prueba ya reservado "
                "para evaluarlo. Queda fuera del alcance de esta entrega.",
                titulo="Qué viene después",
            ),
        ],
        className="vista-pestana",
    )
