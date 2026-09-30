"""
Pestaña 10 · Conclusiones
Qué quedó establecido con el análisis exploratorio, qué significa para el
objetivo de reducir el panel de secuenciación y qué falta por hacer.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import html

from utils.formato import num, pct
from utils.components import (
    bullet_list,
    callout,
    card,
    data_table,
    enlace_externo,
    kpi_row,
    page_header,
    paragraph,
    section_title,
)
from utils.config import URL_LIBRO, URL_REPO_LIBRO
from utils.data_loader import (
    asociacion_con_grado,
    estadisticas_edad,
    matriz_asociacion_genes,
    prevalencia_genes,
    tamanos_particion,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

SIGUIENTES_PASOS = [
    (
        "Abrir la fase de modelado",
        "El proyecto cierra en el análisis exploratorio. El paso siguiente es "
        "entrenar y contrastar clasificadores —regresión logística, árboles, "
        "ensembles— sobre esta misma partición 80/20 y con las mismas métricas, "
        "para después traer el resultado a este tablero.",
    ),
    (
        "Cuantificar el panel mínimo",
        "Medir cuánta capacidad predictiva se pierde al recortar el panel: solo IDH1; "
        "IDH1 más la edad; el subconjunto de marcadores con señal. Es la respuesta "
        "directa al objetivo del proyecto y todavía no está cuantificada.",
    ),
    (
        "Validar con una cohorte externa",
        "Todo lo observado proviene de TCGA. Repetir los contrastes sobre una cohorte "
        "independiente diría si estas asociaciones se sostienen fuera de esta muestra "
        "o son particularidades suyas.",
    ),
]


def _kpis() -> list[dict]:
    """Las cuatro cifras que resumen lo que encontró el análisis."""
    asociacion = asociacion_con_grado("train")
    edad = estadisticas_edad("train").set_index("grade_label")
    tamanos = tamanos_particion()

    matriz = matriz_asociacion_genes("train", top_n=12).values.copy()
    np.fill_diagonal(matriz, 0.0)

    idh1 = float(asociacion.set_index("variable").loc["IDH1", "rho"])
    significativas = int(asociacion["significativa"].sum())

    return [
        {
            "valor": f"{num(idh1, 2, signo=True)}",
            "etiqueta": "Spearman de IDH1",
            "detalle": "El marcador más discriminante, y de signo protector",
        },
        {
            "valor": f"{num(edad.loc['GBM', 'media'] - edad.loc['LGG', 'media'], 1)} años",
            "etiqueta": "Brecha de edad",
            "detalle": "Entre el diagnóstico de GBM y el de LGG",
        },
        {
            "valor": f"{significativas} de {len(asociacion)}",
            "etiqueta": "Variables con señal",
            "detalle": f"Sobre {tamanos['train']} pacientes de entrenamiento",
        },
        {
            "valor": f"{num(matriz.max(), 2)}",
            "etiqueta": "Redundancia máxima",
            "detalle": "V de Cramér del par de genes más asociado (umbral: 0,70)",
        },
    ]


def _tabla_hallazgos() -> dbc.Table:
    """Hallazgos principales con su evidencia cuantitativa, calculada en vivo."""
    asociacion = asociacion_con_grado("train").set_index("variable")
    prevalencia = prevalencia_genes("train").set_index("gen")
    edad = estadisticas_edad("train").set_index("grade_label")

    idh1 = prevalencia.loc["IDH1"]
    no_significativas = asociacion[~asociacion["significativa"]]

    return data_table(
        ["Hallazgo", "Evidencia", "Consecuencia"],
        [
            [
                "IDH1 es el marcador dominante",
                f"r = {num(asociacion.loc['IDH1', 'rho'], 2, signo=True)} · mutado en "
                f"{pct(idh1['LGG'], 1)} de LGG frente a {pct(idh1['GBM'], 1)} de GBM",
                "Su presencia es casi por sí sola una firma de glioma de bajo grado.",
            ],
            [
                "La edad aporta señal clínica independiente",
                f"r = {num(asociacion.loc['Age_at_diagnosis', 'rho'], 2, signo=True)} · "
                f"{num(edad.loc['GBM', 'media'], 1)} años de media en GBM frente a "
                f"{num(edad.loc['LGG', 'media'], 1)} en LGG",
                "Una variable que ya está en la historia clínica y no cuesta nada "
                "medir discrimina casi tanto como una mutación.",
            ],
            [
                "La mayoría del panel no discrimina el grado",
                f"{len(no_significativas)} de {len(asociacion)} variables sin "
                "asociación apreciable (p ≥ 0,05 o |r| < 0,10)",
                "Sostiene la hipótesis central: un panel reducido podría bastar para "
                "decidir el grado.",
            ],
            [
                "No hay multicolinealidad estructural",
                "Ningún par de variables con V de Cramér > 0,70",
                "Los marcadores retenidos pueden entrar juntos en un modelo sin "
                "inestabilizar la estimación de sus parámetros.",
            ],
            [
                "El grupo racial no es interpretable aquí",
                "Más del 90 % de la cohorte pertenece a una sola categoría",
                "La asociación que aparece es un artefacto de composición de la "
                "muestra, no un hallazgo biológico.",
            ],
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de conclusiones."""
    asociacion = asociacion_con_grado("train")
    significativas = int(asociacion["significativa"].sum())

    return html.Div(
        [
            page_header(
                "Conclusiones",
                "Un puñado de marcadores concentra casi toda la información sobre el "
                "grado del tumor, y el análisis dice cuáles son.",
                "bi-check2-circle",
            ),
            kpi_row(_kpis()),
            card(
                [
                    paragraph(
                        f"De las {len(asociacion)} variables disponibles, solo "
                        f"{significativas} muestran una asociación con el grado que sea "
                        "a la vez estadísticamente significativa y de magnitud "
                        "apreciable. El resto aporta una varianza explicativa "
                        "insignificante: no distingue a un glioma de bajo grado de un "
                        "glioblastoma."
                    ),
                    paragraph(
                        "Ese es el resultado que importa para el objetivo del proyecto. "
                        "No dice todavía cuánta precisión se conseguiría con un panel "
                        "reducido —eso exige entrenar y comparar modelos— pero sí "
                        "establece qué variables merecen entrar en esa comparación y "
                        "cuáles se pueden descartar desde ya."
                    ),
                    callout(
                        "La señal que encuentran los contrastes coincide con la "
                        "literatura clínica: IDH1 e IDH2 hacia LGG, y la edad, TP53 y "
                        "PTEN hacia GBM. Que un análisis hecho a ciegas sobre los datos "
                        "reproduzca el conocimiento médico establecido es la mejor "
                        "validación disponible sin una cohorte externa.",
                        titulo="Coherencia con el conocimiento clínico",
                    ),
                ],
                titulo="Resultado principal",
            ),
            section_title("Hallazgos y su evidencia"),
            card(
                _tabla_hallazgos(),
                titulo="Cinco hallazgos con respaldo cuantitativo",
                subtitulo="Cifras calculadas sobre el conjunto de entrenamiento",
                className="mb-4",
            ),
            section_title("Qué falta por hacer"),
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
                    for i, (titulo, descripcion) in enumerate(SIGUIENTES_PASOS)
                ],
                className="g-3",
            ),
            card(
                [
                    paragraph("El proyecto se entrega en tres piezas complementarias:"),
                    bullet_list(
                        [
                            html.Span(
                                [
                                    html.B("Jupyter Book: "),
                                    "el análisis exploratorio completo con su desarrollo "
                                    "estadístico. ",
                                    enlace_externo("Abrirlo", URL_LIBRO),
                                ]
                            ),
                            html.Span(
                                [
                                    html.B("Dashboard: "),
                                    "esta capa interactiva, con el EDA navegable y el "
                                    "selector de conjunto de datos.",
                                ]
                            ),
                            html.Span(
                                [
                                    html.B("Código: "),
                                    "pipeline reproducible con una semilla fija, del CSV "
                                    "a los contrastes. ",
                                    enlace_externo("Repositorio", URL_REPO_LIBRO),
                                ]
                            ),
                        ],
                    ),
                ],
                titulo="Entregables",
            ),
        ],
        className="vista-pestana",
    )
