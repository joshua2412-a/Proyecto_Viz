"""
Pestaña 3 · Problema
Formula el problema analítico: qué se quiere decidir, con qué información y
por qué secuenciar el panel completo no siempre es viable.
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
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import asociacion_con_grado
from utils.figures import fig_prevalencia_genes
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

AMBITO = "train"


def _tabla_señal() -> dbc.Table:
    """Reparto de las 23 predictoras entre las que aportan señal y las que no."""
    asociacion = asociacion_con_grado(AMBITO)
    significativas = asociacion[asociacion["significativa"]]
    descartables = asociacion[~asociacion["significativa"]]

    return data_table(
        ["Grupo", "Variables", "Cuáles"],
        [
            [
                "Con señal (p < 0,05 y |r| ≥ 0,10)",
                f"{len(significativas)}",
                ", ".join(significativas["variable"].tolist()),
            ],
            [
                "Sin señal apreciable",
                f"{len(descartables)}",
                ", ".join(descartables["variable"].tolist()),
            ],
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de problema."""
    return html.Div(
        [
            page_header(
                "El problema: decidir el grado con la menor información posible",
                "Clasificar LGG frente a GBM es un problema de clasificación binaria, "
                "pero con una restricción práctica: cada gen que se secuencia cuesta.",
                "bi-exclamation-circle",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                section_title("Situación", nivel=5),
                                paragraph(
                                    "La caracterización biomolecular es hoy imprescindible "
                                    "para determinar el pronóstico y el tratamiento de un "
                                    "glioma. Pero las pruebas de secuenciación genética "
                                    "completa suponen un coste económico elevado para los "
                                    "sistemas de salud y para los pacientes, y no todos los "
                                    "centros tienen acceso a ellas en el momento en que hay "
                                    "que decidir un tratamiento."
                                ),
                                section_title("Tensión", nivel=5),
                                paragraph(
                                    "Se secuencia un panel de 20 genes, pero no todos "
                                    "informan por igual sobre el grado del tumor. Si una "
                                    "parte de esas mutaciones no ayuda a distinguir LGG de "
                                    "GBM, se está pagando por información que no cambia la "
                                    "decisión clínica."
                                ),
                                section_title("Pregunta analítica", nivel=5),
                                paragraph(
                                    "¿Cuál es el subconjunto mínimo de variables clínicas y "
                                    "mutaciones que permite clasificar el grado del glioma "
                                    "con una precisión aceptable, y cuánta precisión se "
                                    "pierde al reducir el panel?"
                                ),
                                callout(
                                    "El problema no es solo maximizar la precisión: es "
                                    "identificar cuánta precisión se consigue con cuánta "
                                    "información, para que la reducción de costes sea una "
                                    "decisión informada y no un recorte a ciegas.",
                                    titulo="Cómo se plantea aquí",
                                ),
                            ],
                            titulo="Planteamiento",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            fig_prevalencia_genes(AMBITO, top_n=12),
                            "Prevalencia de mutación por gen y grado tumoral",
                            "Ordenado por la diferencia entre grados: arriba, los genes "
                            "que más separan LGG de GBM. IDH1 es el caso extremo.",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("No todas las variables informan del grado"),
            card(
                [
                    paragraph(
                        "Cruzando cada predictora con el grado tumoral en el conjunto de "
                        "entrenamiento, las variables se reparten en dos grupos muy "
                        "desiguales. Las que no alcanzan significancia estadística ni una "
                        "magnitud mínima de asociación aportan una varianza explicativa "
                        "insignificante, y son las primeras candidatas a salir del panel."
                    ),
                    _tabla_señal(),
                    bullet_list(
                        [
                            "IDH1 es el marcador con mayor capacidad discriminativa y de "
                            "signo inverso a la malignidad (r ≈ -0,70): su mutación es "
                            "prácticamente una firma de glioma de bajo grado.",
                            "La edad al diagnóstico es el principal correlato clínico "
                            "directo con la severidad (r ≈ +0,53).",
                            "PTEN es el marcador de riesgo genómico más sólido "
                            "(V de Cramér ≈ 0,36), seguido de EGFR y RB1.",
                            "Gender, PDGFRA, NF1, CSMD3, BCOR, PIK3CA y FAT4 no muestran "
                            "asociación apreciable con el grado en esta muestra.",
                        ],
                    ),
                ],
                titulo="Reparto de las 23 predictoras",
                subtitulo="Conjunto de entrenamiento · Spearman y chi-cuadrado",
            ),
            callout(
                "Que una variable no discrimine el grado en esta muestra no significa que "
                "sea irrelevante en oncología: puede importar para el pronóstico, la "
                "respuesta al tratamiento o la supervivencia, que no son lo que este "
                "análisis mide.",
                titulo="Alcance de la afirmación",
            ),
        ],
        className="vista-pestana",
    )
