"""
Pestaña 9 · Limitaciones
Qué NO se puede concluir con este trabajo. Se declara de forma explícita para
que ninguna cifra del tablero se lea con más alcance del que tiene.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.formato import pct
from utils.components import (
    callout,
    card,
    data_table,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import (
    asociacion_con_grado,
    distribucion_clinica,
    tamanos_particion,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

LIMITACIONES = [
    (
        "Cohorte única y no local",
        "Todos los pacientes provienen de los proyectos TCGA-LGG y TCGA-GBM, "
        "recogidos mayoritariamente en centros de Estados Unidos. No hay validación "
        "externa con una cohorte independiente, y menos aún con población "
        "latinoamericana, así que el desempeño observado no es trasladable sin más a "
        "otro contexto asistencial.",
    ),
    (
        "Fuerte desbalance en el grupo racial",
        "Más de nueve de cada diez pacientes pertenecen al grupo White. Las "
        "categorías minoritarias tienen tan pocas observaciones que su coeficiente "
        "no es estimable con precisión: la asociación significativa entre grupo "
        "racial y grado debe leerse con máxima cautela y no como un efecto biológico.",
    ),
    (
        "Asociación, no causalidad",
        "Las pruebas de independencia y las correlaciones detectan asociación "
        "estadística. Que IDH1 mutado acompañe a los gliomas de bajo grado no "
        "establece un mecanismo causal: la dirección del efecto y su explicación "
        "biológica están fuera del alcance de estos contrastes.",
    ),
    (
        "El análisis describe el grado, no el pronóstico",
        "La variable objetivo es la clasificación histológica LGG/GBM. Nada de lo "
        "que se observa aquí habla de supervivencia, de respuesta al tratamiento ni "
        "de progresión: son preguntas distintas que requieren otros datos y otro "
        "diseño.",
    ),
    (
        "Panel de genes fijo y binario",
        "Las 20 mutaciones entran como indicadores 0/1: no se distingue el tipo de "
        "variante, su carga alélica ni su localización. Dos pacientes con 'IDH1 "
        "mutado' pueden tener alteraciones biológicamente distintas.",
    ),
    (
        "Es un análisis exploratorio, no una capacidad predictiva medida",
        "Saber que una variable se asocia al grado no dice cuánta precisión "
        "aportaría en una clasificación real. Esa pregunta pertenece a la fase de "
        "modelado, que queda fuera del alcance de este trabajo.",
    ),
    (
        "La reducción de costes es una hipótesis, no un resultado clínico",
        "Que la mayoría de los genes no muestre asociación con el grado sugiere que un "
        "panel más pequeño bastaría para clasificar el grado. Validar eso exige un "
        "estudio prospectivo con el panel reducido, no solo un modelo entrenado "
        "sobre datos históricos.",
    ),
    (
        "Contrastes múltiples: cuáles no aguantan la corrección",
        "Se contrastan 23 variables contra el grado por separado, así que alguna "
        "significancia aparece por azar. El criterio principal del tablero no "
        "corrige por ello, pero la tabla de Resultados marca cuáles sobreviven a "
        "la corrección de Bonferroni, que es conservadora. Las asociaciones "
        "fuertes no se mueven; las que rozaban el umbral, sí.",
    ),
]


def _tabla_representatividad() -> dbc.Table:
    """Composición por grupo racial: la evidencia del sesgo de la muestra."""
    tabla = distribucion_clinica("Race", "train")
    agregado = (
        tabla.groupby("categoria", as_index=False)["pacientes"].sum().sort_values(
            "pacientes", ascending=False
        )
    )
    total = agregado["pacientes"].sum()
    return data_table(
        ["Grupo racial reportado", "Pacientes", "% de la cohorte de entrenamiento"],
        [
            [
                fila["categoria"],
                f"{int(fila['pacientes'])}",
                f"{pct(fila['pacientes'] / total * 100, 2)}",
            ]
            for _, fila in agregado.iterrows()
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de limitaciones."""
    tamanos = tamanos_particion()
    asociacion = asociacion_con_grado("train")

    return html.Div(
        [
            page_header(
                "Limitaciones",
                "Ocho fronteras del trabajo, declaradas antes de que alguien las "
                "encuentre leyendo las cifras con demasiada confianza.",
                "bi-exclamation-triangle",
            ),
            callout(
                f"Todo lo que se afirma aquí se sostiene sobre {tamanos['full']} pacientes "
                f"de una sola fuente, y los {len(asociacion)} contrastes se calculan "
                f"sobre los {tamanos['train']} del conjunto de entrenamiento. Es una muestra "
                "suficiente para orientar decisiones de análisis y pequeña para una "
                "afirmación clínica.",
                titulo="El tamaño manda",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(descripcion),
                            titulo=titulo,
                        ),
                        lg=6,
                        className="mb-3",
                    )
                    for i, (titulo, descripcion) in enumerate(LIMITACIONES)
                ],
                className="g-3",
            ),
            section_title("La limitación más medible: representatividad"),
            card(
                [
                    paragraph(
                        "El sesgo de composición no es una sospecha, se puede contar. Esta "
                        "es la distribución real por grupo racial en la cohorte de "
                        "entrenamiento, y explica por qué cualquier conclusión sobre "
                        "grupos minoritarios queda fuera del alcance del trabajo."
                    ),
                    _tabla_representatividad(),
                ],
                titulo="Composición de la cohorte por grupo racial",
            ),
            callout(
                "Un dashboard que estima probabilidades sobre personas tiene que decir con "
                "la misma claridad qué no sabe. Esa es la función de esta pestaña: no es un "
                "trámite académico, es parte del resultado.",
                titulo="Por qué esta pestaña existe",
            ),
        ],
        className="vista-pestana",
    )
