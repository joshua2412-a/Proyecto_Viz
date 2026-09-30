"""
Pestaña 4 · Objetivos
Objetivo general, objetivos específicos y criterios con los que se considera
que el análisis exploratorio cumplió su papel.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import html

from utils.formato import num, pct
from utils.components import (
    badge_numero,
    card,
    data_table,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import (
    asociacion_con_grado,
    get_dataframe,
    matriz_asociacion_genes,
    proporcion_grado,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

OBJETIVOS_ESPECIFICOS = [
    (
        "1",
        "Caracterizar la cohorte",
        "Describir el perfil clínico (edad, género, grupo racial) y mutacional de "
        "los 839 pacientes, y verificar la calidad del dataset: duplicados, valores "
        "faltantes, valores atípicos y supuestos distribucionales.",
    ),
    (
        "2",
        "Reservar un conjunto de prueba antes de mirar nada",
        "Separar el 20 % de los pacientes con una partición estratificada antes de "
        "calcular el primer estadístico, para que ninguna decisión tomada durante el "
        "análisis esté contaminada por los datos con los que se evaluará después.",
    ),
    (
        "3",
        "Medir la asociación de cada variable con el grado",
        "Contrastar cada predictora contra el grado tumoral con la prueba adecuada a "
        "su naturaleza (Mann-Whitney para la edad, chi-cuadrado y V de Cramér para "
        "las binarias y categóricas) y ordenar las variables por capacidad "
        "discriminativa.",
    ),
    (
        "4",
        "Descartar redundancia entre marcadores",
        "Evaluar la co-ocurrencia entre mutaciones para detectar multicolinealidad y "
        "confirmar que el conjunto retenido puede entrar completo en un modelo "
        "posterior sin inestabilidad en la estimación de los parámetros.",
    ),
    (
        "5",
        "Delimitar el panel que merece la pena secuenciar",
        "Separar las variables que aportan señal sobre el grado de las que no, para "
        "sostener con evidencia la hipótesis de que un panel más pequeño bastaría "
        "para la decisión clínica.",
    ),
    (
        "6",
        "Poner el análisis a disposición de otros",
        "Publicar el desarrollo completo como Jupyter Book y una capa interactiva "
        "—este dashboard— que permita explorar el EDA sin tocar el código.",
    ),
]


def _tabla_criterios() -> dbc.Table:
    """Criterios de cumplimiento contrastados con los datos, en vivo."""
    df = get_dataframe("full")
    asociacion = asociacion_con_grado("train")
    matriz = matriz_asociacion_genes("train", top_n=12)
    significativas = int(asociacion["significativa"].sum())

    nulos = int(df.isnull().sum().sum())
    rho_maximo = float(asociacion["rho"].abs().max())

    # Diferencia de composición entre entrenamiento y prueba (en puntos)
    gbm_train = proporcion_grado("train").set_index("grado").loc["GBM", "porcentaje"]
    gbm_test = proporcion_grado("test").set_index("grado").loc["GBM", "porcentaje"]
    brecha = abs(gbm_train - gbm_test)

    # V de Cramér máxima entre pares distintos de genes
    valores = matriz.values.copy()
    np.fill_diagonal(valores, 0.0)
    v_maxima = float(valores.max())

    filas = [
        [
            "Dataset sin valores faltantes",
            f"{nulos} nulos en {df.shape[0]} × {df.shape[1] - 3} celdas",
            "Cumple" if nulos == 0 else "Revisar",
        ],
        [
            "La estratificación conserva la proporción de clases",
            f"GBM: {pct(gbm_train, 1)} en entrenamiento y {pct(gbm_test, 1)} en prueba "
            f"({num(brecha, 1)} puntos de diferencia)",
            "Cumple" if brecha < 2 else "Revisar",
        ],
        [
            "Al menos un marcador con asociación fuerte (|r| ≥ 0,50)",
            f"El máximo observado es |r| = {num(rho_maximo, 2)}",
            "Cumple" if rho_maximo >= 0.50 else "No cumple",
        ],
        [
            "Sin multicolinealidad estructural (V de Cramér < 0,70)",
            f"El par de genes más asociado llega a V = {num(v_maxima, 2)}",
            "Cumple" if v_maxima < 0.70 else "Revisar",
        ],
        [
            "El panel se puede reducir: no todas las variables aportan",
            f"{significativas} de {len(asociacion)} variables muestran señal",
            "Cumple" if significativas < len(asociacion) else "No cumple",
        ],
    ]
    return data_table(["Criterio", "Resultado observado", "Estado"], filas)


def layout() -> html.Div:
    """Layout de la pestaña de objetivos."""
    return html.Div(
        [
            page_header(
                "Objetivos del proyecto",
                "Qué se propone construir, con qué pasos y con qué criterio se "
                "considera suficiente el resultado.",
                "bi-bullseye",
            ),
            card(
                [
                    paragraph(
                        "Construir una solución analítica integral que identifique el "
                        "subconjunto óptimo de factores clínicos y mutaciones genéticas "
                        "para clasificar con precisión la severidad del glioma (LGG frente "
                        "a GBM), reduciendo los costos asociados a pruebas moleculares "
                        "innecesarias."
                    ),
                    paragraph(
                        "El énfasis está en la palabra óptimo: no se busca la mayor "
                        "cantidad de información posible, sino la mínima que sostenga la "
                        "decisión clínica. Este tablero cubre la fase que responde esa "
                        "pregunta: cuáles de las 23 variables disponibles se relacionan de "
                        "verdad con el grado del tumor, y con qué fuerza."
                    ),
                    paragraph(
                        "La fase de modelado, que traduciría esos hallazgos en un "
                        "clasificador, es el paso siguiente y queda fuera del alcance "
                        "de esta entrega."
                    ),
                ],
                titulo="Objetivo general",
                tono="acento",
            ),
            section_title("Objetivos específicos"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(badge_numero(numero), className="guide-badge"),
                                html.Div(titulo, className="guide-name"),
                                html.Div(descripcion, className="guide-text"),
                            ],
                            className="guide-card",
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (numero, titulo, descripcion) in enumerate(
                        OBJETIVOS_ESPECIFICOS
                    )
                ],
                className="g-3",
            ),
            section_title("Criterios de cumplimiento"),
            card(
                [
                    paragraph(
                        "Cada criterio se comprueba contra los datos cada vez que se abre "
                        "esta pestaña, no contra una cifra escrita a mano. Si mañana se "
                        "reemplaza el dataset, la tabla se recalcula y delata cualquier "
                        "supuesto que haya dejado de cumplirse."
                    ),
                    _tabla_criterios(),
                ],
                titulo="Estado frente a los criterios definidos",
                subtitulo="Calculado en vivo sobre el dataset del proyecto",
            ),
        ],
        className="vista-pestana",
    )
