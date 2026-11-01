"""
Pestaña 3 · Problema
Formula el problema analítico: qué se quiere decidir, con qué información y
por qué secuenciar el panel completo no siempre es viable.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.config import nombre_corto
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
from utils.formato import num
from utils.figures import fig_prevalencia_genes
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

AMBITO = "train"


def _v_de(variable: str) -> str:
    """V de Cramér de una predictora, calculada en vez de escrita a mano."""
    tabla = asociacion_con_grado(AMBITO).set_index("variable")
    return num(float(tabla.loc[variable, "v_cramer"]), 2)


def _tabla_señal() -> dbc.Table:
    """Reparto de las 23 predictoras entre las que aportan señal y las que no."""
    asociacion = asociacion_con_grado(AMBITO)
    significativas = asociacion[asociacion["significativa"]]
    descartables = asociacion[~asociacion["significativa"]]

    # Las tres clínicas llegan con el nombre de su columna (Age_at_diagnosis,
    # Gender, Race); en la tabla se leen en español, como en el resto del tablero.
    def nombres(filas) -> str:
        return ", ".join(nombre_corto(v) for v in filas["variable"])

    return data_table(
        ["Grupo", "Variables", "Cuáles"],
        [
            [
                r"Con señal ($p < 0{,}05$ y $|r| \geq 0{,}10$)",
                f"{len(significativas)}",
                nombres(significativas),
            ],
            [
                "Sin señal apreciable",
                f"{len(descartables)}",
                nombres(descartables),
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
                "error",
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
                        "desiguales. Las que no alcanzan significancia estadística o una "
                        "magnitud mínima de asociación aportan una varianza explicativa "
                        "insignificante, y son las primeras candidatas a salir del panel."
                    ),
                    _tabla_señal(),
                    bullet_list(
                        [
                            "IDH1 es el marcador con mayor capacidad discriminativa y de "
                            r"signo inverso a la malignidad ($r \approx -0{,}70$): su mutación es "
                            "prácticamente una firma de glioma de bajo grado.",
                            "La edad al diagnóstico es el principal correlato clínico "
                            r"directo con la severidad ($r \approx +0{,}53$).",
                            f"PTEN es el marcador de riesgo genómico más sólido "
                            f"(V de Cramér = {_v_de('PTEN')}), seguido de EGFR y RB1.",
                            # La tabla de arriba lista nueve; como en el libro,
                            # se nombran las más claras sin darlas por todas.
                            "Género, PDGFRA, NF1, CSMD3, BCOR, PIK3CA y FAT4, entre "
                            "otras, no muestran asociación apreciable con el grado en "
                            "esta muestra.",
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
