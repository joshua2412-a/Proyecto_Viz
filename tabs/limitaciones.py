"""
Pestaña 9 · Limitaciones
Qué NO se puede concluir con este trabajo. Se declara de forma explícita para
que ninguna cifra del tablero se lea con más alcance del que tiene.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.components import (
    callout,
    card,
    data_table,
    page_header,
    paragraph,
    section_title,
)
from utils.data_loader import distribucion_clinica, load_metrics, tamanos_particion
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
        "establece un mecanismo causal, y el modelo tampoco lo estima.",
    ),
    (
        "El modelo predice el grado, no el pronóstico",
        "La variable objetivo es la clasificación histológica LGG/GBM. Nada de lo "
        "que estima este modelo habla de supervivencia, de respuesta al tratamiento "
        "ni de progresión: son preguntas distintas que requieren otros datos y otro "
        "diseño.",
    ),
    (
        "Panel de genes fijo y binario",
        "Las 20 mutaciones entran como indicadores 0/1: no se distingue el tipo de "
        "variante, su carga alélica ni su localización. Dos pacientes con 'IDH1 "
        "mutado' pueden tener alteraciones biológicamente distintas.",
    ),
    (
        "Es un baseline, no el modelo final",
        "La regresión logística se eligió por interpretabilidad y como referencia "
        "comparable con la literatura. El proyecto contempla contrastarla con "
        "modelos más complejos; hasta que esa comparación esté hecha, estas cifras "
        "son la cota de referencia, no el techo alcanzable.",
    ),
    (
        "La reducción de costes es una hipótesis, no un resultado clínico",
        "Que la penalización L1 deje en cero la mayoría de los genes sugiere que un "
        "panel más pequeño bastaría para clasificar el grado. Validar eso exige un "
        "estudio prospectivo con el panel reducido, no solo un modelo entrenado "
        "sobre datos históricos.",
    ),
    (
        "Ausencia de calibración evaluada",
        "Se reportan métricas de discriminación (AUC, recall, precisión), pero no se "
        "evaluó la calibración de las probabilidades. Una probabilidad del 70 % no "
        "está verificada como equivalente a un 70 % de casos reales de GBM.",
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
                f"{fila['pacientes'] / total * 100:.2f}%",
            ]
            for _, fila in agregado.iterrows()
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de limitaciones."""
    metricas = load_metrics()
    tamanos = tamanos_particion()

    return html.Div(
        [
            page_header(
                "Limitaciones",
                "Ocho fronteras del trabajo, declaradas antes de que alguien las "
                "encuentre leyendo las cifras con demasiada confianza.",
                "⚠️",
            ),
            callout(
                f"Todo lo que se afirma aquí se sostiene sobre {tamanos['full']} pacientes "
                f"de una sola fuente, con {metricas['n_test']} de ellos usados para medir "
                "el desempeño. Es una muestra suficiente para un baseline y pequeña para "
                "una afirmación clínica.",
                titulo="El tamaño manda",
                color=COLOR_GBM,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            paragraph(descripcion),
                            titulo=titulo,
                            color=SERIES[i % len(SERIES)],
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
                color=COLOR_LGG,
            ),
            callout(
                "Un dashboard que estima probabilidades sobre personas tiene que decir con "
                "la misma claridad qué no sabe. Esa es la función de esta pestaña: no es un "
                "trámite académico, es parte del resultado.",
                titulo="Por qué esta pestaña existe",
                color=SERIES[2],
            ),
        ],
        className="tab-content",
    )
