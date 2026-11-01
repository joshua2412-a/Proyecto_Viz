"""
Pestaña 4 · Objetivos
Objetivo general, objetivos específicos y criterios con los que se considera
que el análisis exploratorio cumplió su papel.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import html

from utils.formato import num, pct, tnum
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
from utils.modelo import cargar_resumen, panel_minimo
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
        # Antes eran dos objetivos (redundancia y panel). Se unieron para dejar
        # sitio al de modelado sin romper la cuadrícula de 3 + 3 tarjetas.
        "4",
        "Delimitar un panel reducido y sin redundancia",
        "Separar las variables que aportan señal sobre el grado de las que no y "
        "evaluar la co-ocurrencia entre mutaciones, para sostener con evidencia que "
        "un panel más pequeño basta y que puede entrar en un modelo sin "
        "multicolinealidad.",
    ),
    (
        "5",
        "Entrenar y comparar modelos de clasificación",
        "Ajustar regresión logística, Random Forest y XGBoost con validación cruzada "
        "anidada, comparar técnicas de balanceo y de optimización de "
        "hiperparámetros, evaluar discriminación y calibración en el conjunto de "
        "prueba y medir cuánto se pierde con solo las variables que más aportan.",
    ),
    (
        "6",
        "Poner el análisis a disposición de otros",
        "Publicar el desarrollo completo en un Jupyter Book y acompañarlo de un "
        "tablero interactivo que permita explorar los datos, comparar los modelos y "
        "obtener predicciones sin necesidad de ejecutar código.",
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
    # Con las cifras ya redondeadas: restando antes de redondear salía
    # «42,0 % y 41,7 % (0,4 puntos)», que el lector toma por una cuenta mal hecha.
    gbm_train, gbm_test = round(gbm_train, 1), round(gbm_test, 1)
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
            r"Al menos un marcador con asociación fuerte ($|r| \geq 0{,}50$)",
            f"El máximo observado es $|r| = {tnum(rho_maximo, 2)}$",
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

    # Criterios del modelado, con las cifras que exporta el libro
    # (modelos/resumen_modelado.json): AUC de prueba de los tres finales y
    # AUC de validación cruzada del panel de 4 frente a las 23 variables.
    resumen = cargar_resumen()
    auc_prueba = {m: d["auc_test"] for m, d in resumen["modelos"].items()}
    auc_min = min(auc_prueba.values())
    # El panel, con la regla de un error estándar que fija el notebook (el
    # criterio anterior, «el de 4 pierde menos de 0,01», no venía del libro)
    pm = panel_minimo(resumen)
    filas += [
        [
            "Los tres modelos discriminan bien en prueba (AUC ≥ 0,90)",
            f"AUC de {num(auc_min, 3)} a {num(max(auc_prueba.values()), 3)} "
            f"en los {resumen['meta']['n_test']} pacientes de prueba",
            "Cumple" if auc_min >= 0.90 else "No cumple",
        ],
        [
            "Un panel reducido no se distingue del completo (regla de un error estándar)",
            f"{pm['k']} de 23 variables: AUC {num(pm['auc_k'], 3)} frente a "
            f"{num(pm['auc_todas'], 3)} en validación cruzada. Con 4, {num(pm['auc_4'], 3)}",
            "Cumple" if pm["k"] < 23 else "No cumple",
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
                "track_changes",
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
                        "decisión clínica."
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
                        "Cada criterio se evalúa directamente sobre los datos: las cifras "
                        "de la tabla salen del mismo cálculo que el resto del tablero, de "
                        "modo que el estado refleja lo que los datos muestran y no una "
                        "afirmación previa."
                    ),
                    _tabla_criterios(),
                ],
                titulo="Estado frente a los criterios definidos",
                subtitulo="Calculado sobre el dataset del proyecto, no escrito a mano",
            ),
        ],
        className="vista-pestana",
    )
