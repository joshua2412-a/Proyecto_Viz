"""
Pestaña 5 · Marco teórico
Los conceptos que hacen falta para leer el resto del tablero: qué mide cada
variable, cómo se cuantifica una asociación y qué representa cada gen del
panel.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    enlace_externo,
    page_header,
    paragraph,
    section_title,
)
from utils.config import (
    GENDER_LABELS,
    GENE_DESCRIPCION,
    GENE_FEATURES,
    RACE_LABELS,
    URL_LIBRO_EDA,
)
from utils.data_loader import asociacion_con_grado, prevalencia_genes
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

AMBITO = "train"

# Fórmulas en LaTeX (dcc.Markdown con mathjax=True las renderiza)
FORMULA_CRAMER = r"""
$$
V \;=\; \sqrt{\frac{\chi^{2}}{n \,\cdot\, \min(k-1,\; r-1)}}
$$
"""

FORMULA_SPEARMAN = r"""
$$
\rho \;=\; 1 - \frac{6 \sum d_i^{2}}{n\,(n^{2}-1)}
$$
"""

ESTADISTICOS = [
    (
        "p-valor",
        "Probabilidad de observar una diferencia así de grande si en realidad no "
        "hubiera ninguna.",
        "Responde a si la asociación es distinguible del azar, no a si es "
        "importante. Con 671 pacientes, diferencias minúsculas pueden salir "
        "significativas.",
    ),
    (
        "Spearman (ρ)",
        "Correlación entre los rangos de dos variables: va de -1 a +1.",
        "Aporta el signo, que es lo que permite decir si una variable empuja hacia "
        "LGG (negativo) o hacia GBM (positivo). No exige normalidad ni linealidad.",
    ),
    (
        "V de Cramér",
        "Fuerza de la asociación entre dos variables categóricas: va de 0 a 1.",
        "Es el complemento del chi-cuadrado: la prueba dice si hay asociación, la V "
        "dice cuánta. Sirve tanto contra el grado como entre pares de genes.",
    ),
    (
        "Rango intercuartílico (IQR)",
        "Distancia entre el percentil 25 y el 75 de una variable continua.",
        "Describe la dispersión sin que la arrastren los extremos, y define el "
        "criterio con el que se identifican valores atípicos.",
    ),
    (
        "Prevalencia",
        "Proporción de pacientes que presentan una mutación.",
        "Una mutación muy rara puede tener una asociación fuerte y aun así aportar "
        "poco en la práctica, simplemente porque casi nadie la tiene.",
    ),
]


def _tabla_variables() -> dbc.Table:
    """Operacionalización de las variables del dataset."""
    return data_table(
        ["Variable", "Tipo", "Codificación", "Papel en el análisis"],
        [
            [
                "Grade",
                "Binaria",
                "0 = LGG · 1 = GBM",
                "Variable objetivo",
            ],
            [
                "Age_at_diagnosis",
                "Continua (años)",
                "Decimales incluidos: recogen los días exactos",
                "Predictora · se contrasta con Mann-Whitney",
            ],
            [
                "Gender",
                "Binaria",
                " · ".join(f"{clave} = {valor}" for clave, valor in GENDER_LABELS.items()),
                "Predictora · chi-cuadrado y V de Cramér",
            ],
            [
                "Race",
                "Categórica (4 niveles)",
                " · ".join(f"{clave} = {valor}" for clave, valor in RACE_LABELS.items()),
                "Predictora · chi-cuadrado (categorías agrupadas)",
            ],
            [
                f"{len(GENE_FEATURES)} genes",
                "Binarias",
                "0 = no mutado (wildtype) · 1 = mutado",
                "Predictoras · chi-cuadrado, V de Cramér y Spearman",
            ],
        ],
    )


def _tabla_genes() -> dbc.Table:
    """Panel de genes con su función biológica y su prevalencia observada."""
    prevalencia = prevalencia_genes(AMBITO).set_index("gen")
    asociacion = asociacion_con_grado(AMBITO).set_index("variable")

    filas = []
    for gen in prevalencia.index:  # ya viene ordenado por prevalencia
        fila_prev = prevalencia.loc[gen]
        rho = asociacion.loc[gen, "rho"]
        hacia = "LGG" if rho < 0 else "GBM"
        marca = "" if asociacion.loc[gen, "significativa"] else " (no significativa)"
        filas.append(
            [
                gen,
                GENE_DESCRIPCION.get(gen, ""),
                f"{fila_prev['prevalencia']:.1f}%",
                f"{rho:+.2f} → {hacia}{marca}",
            ]
        )
    return data_table(
        ["Gen", "Función biológica", "Prevalencia", "Asociación con el grado"], filas
    )


def layout() -> html.Div:
    """Layout de la pestaña de marco teórico."""
    return html.Div(
        [
            page_header(
                "Marco teórico",
                "Qué mide cada variable, qué prueba estadística le corresponde y "
                "cómo se interpretan sus resultados.",
                "bi-journal-text",
            ),
            section_title("Operacionalización de las variables"),
            card(
                [
                    paragraph(
                        "El dataset llega ya codificado numéricamente. Conocer esa "
                        "codificación es imprescindible para leer los coeficientes: un "
                        "coeficiente positivo en Gender no significa 'ser hombre aumenta "
                        "el riesgo', sino que la categoría codificada como 1 (femenino) lo "
                        "hace respecto a la de referencia."
                    ),
                    _tabla_variables(),
                ],
                titulo="Las 23 predictoras y la variable objetivo",
            ),
            section_title("Cómo se mide una asociación"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Decir que una mutación «está relacionada» con el "
                                    "grado del tumor no significa nada hasta que se "
                                    "cuantifica. Dos preguntas distintas, dos números "
                                    "distintos: ¿es distinguible del azar?, y ¿de qué "
                                    "tamaño es el efecto?"
                                ),
                                paragraph(
                                    "Para variables categóricas, la prueba de "
                                    "chi-cuadrado responde a la primera y la V de Cramér "
                                    "a la segunda. La V normaliza el estadístico por el "
                                    "tamaño de la muestra y por las dimensiones de la "
                                    "tabla, de modo que queda acotada entre 0 y 1 y se "
                                    "puede comparar entre pares de variables:"
                                ),
                                dcc.Markdown(
                                    FORMULA_CRAMER,
                                    mathjax=True,
                                    className="formula-block",
                                ),
                                callout(
                                    "Esta distinción es la que evita la trampa más común "
                                    "del análisis exploratorio: con muestras grandes, "
                                    "casi todo sale significativo. Por eso aquí una "
                                    "variable solo se considera con señal si además "
                                    "alcanza una magnitud mínima.",
                                    titulo="Significancia no es magnitud",
                                ),
                            ],
                            titulo="Significancia y tamaño del efecto",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "La correlación de Spearman aporta lo que la V de "
                                    "Cramér no puede dar: la dirección. Se calcula sobre "
                                    "los rangos de los datos, no sobre sus valores, de "
                                    "modo que no exige normalidad ni una relación lineal."
                                ),
                                dcc.Markdown(
                                    FORMULA_SPEARMAN,
                                    mathjax=True,
                                    className="formula-block",
                                ),
                                bullet_list(
                                    [
                                        "Signo negativo: la variable acompaña a los "
                                        "gliomas de bajo grado.",
                                        "Signo positivo: acompaña a los glioblastomas.",
                                        "Cerca de cero: no distingue entre los dos.",
                                    ],
                                ),
                                paragraph(
                                    "Codificando el grado como 0 = LGG y 1 = GBM, el "
                                    "signo de ρ se lee directamente como la dirección "
                                    "hacia la que empuja cada marcador."
                                ),
                            ],
                            titulo="Dirección del efecto",
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                ]
            ),
            section_title("Los estadísticos que verás en Resultados"),
            card(
                data_table(
                    ["Estadístico", "Qué mide", "Cómo leerlo"],
                    [[nombre, definicion, lectura] for nombre, definicion, lectura in ESTADISTICOS],
                ),
                titulo="Cinco números y su interpretación",
                subtitulo="Aparecen en las tablas y los tooltips de la pestaña de resultados",
                className="mb-4",
            ),
            section_title("El panel de genes"),
            card(
                [
                    paragraph(
                        "Los 20 genes del panel son los de mayor frecuencia de mutación en "
                        "los proyectos TCGA-LGG y TCGA-GBM. La prevalencia y la asociación "
                        "se calculan sobre el conjunto de entrenamiento, así que cambian "
                        "solas si se reemplaza el dataset."
                    ),
                    _tabla_genes(),
                    html.Div(
                        enlace_externo(
                            "Ver el desarrollo completo del EDA en el libro",
                            URL_LIBRO_EDA,
                        ),
                        className="mt-3",
                    ),
                ],
                titulo="Función biológica, prevalencia y dirección de cada gen",
                subtitulo="Ordenado por prevalencia de mutación en la cohorte de entrenamiento",
            ),
        ],
        className="vista-pestana",
    )
