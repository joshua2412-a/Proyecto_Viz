"""
Pestaña 1 · Introducción
Portada del tablero: presenta el problema de la caracterización del grado
tumoral en gliomas y reparte la lectura del resto de pestañas en tres bloques.

Cada pestaña expone una única función pública `layout()` y es totalmente
autónoma: no importa nada de las demás pestañas. Lo único que esta comparte
con `app.py` son los identificadores de pestaña que llevan los botones de la
guía (`tab-contexto`, `tab-problema`...), que el callback `ir_a_pestana` usa
para cambiar de pestaña sin recargar nada.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import get_asset_url, html

from utils.formato import lista_y, num, pct
from utils.components import (
    bullet_list,
    callout,
    card,
    enlace_externo,
    kpi_row,
    paragraph,
    section_title,
    series_chips,
)
from utils.config import AUTORES, URL_DATASET, URL_LIBRO
from utils.data_loader import (
    asociacion_con_grado,
    prevalencia_genes,
    proporcion_grado,
    tamanos_particion,
)
from utils.navegacion import BLOQUES, DESCRIPCIONES, ETIQUETAS, ICONOS
from utils.theme import ACCENT_NEUTRAL, COLOR_GBM, COLOR_LGG


# --------------------------------------------------------------------------- #
# Piezas de la pestaña
# --------------------------------------------------------------------------- #
def _portada() -> html.Div:
    """Cabecera ilustrada: el titular a la izquierda, la ilustración a la derecha.

    La ilustración vive en assets/portada.svg y está dibujada con la misma
    paleta que el tema, así que la portada y los gráficos se leen como una
    sola pieza.
    """
    return html.Div(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            "Análisis exploratorio · TCGA-LGG y TCGA-GBM",
                            className="hero-eyebrow",
                        ),
                        html.H2(
                            "Gliomas: qué separa a un LGG de un GBM",
                            className="hero-titulo",
                        ),
                        html.P(
                            "839 pacientes, tres variables clínicas y veinte "
                            "mutaciones. Este tablero mide cuáles de ellas "
                            "separan de verdad los dos grados, y con qué fuerza.",
                            className="hero-subtitulo",
                        ),
                        series_chips(),
                        html.Div(
                            "Por "
                            + lista_y(autor["nombre"] for autor in AUTORES),
                            className="hero-firma",
                        ),
                    ],
                    lg=6,
                    className="hero-texto",
                ),
                dbc.Col(
                    html.Img(
                        src=get_asset_url("portada.svg"),
                        alt="Corte axial esquemático de un cerebro con una lesión "
                            "en un hemisferio, enlazada al panel de veinte genes "
                            "del estudio, con la casilla de IDH1 resaltada.",
                        className="hero-imagen",
                    ),
                    lg=6,
                ),
            ],
            className="align-items-center g-4",
        ),
        className="hero",
    )


def _kpis() -> list[dict]:
    """Indicadores de portada, todos calculados a partir del propio dataset.

    El color de la franja es semántico: neutro cuando la cifra no habla de un
    grado en concreto, coral cuando habla de GBM, y partido azul/coral cuando
    la cifra es justamente la distancia entre los dos.
    """
    tamanos = tamanos_particion()
    proporciones = proporcion_grado("full").set_index("grado")
    asociacion = asociacion_con_grado("train")
    idh1 = prevalencia_genes("train").set_index("gen").loc["IDH1"]

    return [
        {
            "valor": f"{tamanos['full']}",
            "etiqueta": "Pacientes analizados",
            "detalle": f"{tamanos['train']} entrenamiento · {tamanos['test']} prueba",
            "color": ACCENT_NEUTRAL,
        },
        {
            "valor": f"{pct(proporciones.loc['GBM', 'porcentaje'], 1)}",
            "etiqueta": "Casos de GBM",
            "detalle": "El resto son gliomas de bajo grado (LGG)",
            "color": COLOR_GBM,
        },
        {
            "valor": f"{int(asociacion['significativa'].sum())} de {len(asociacion)}",
            "etiqueta": "Variables con señal",
            "detalle": "Asociadas al grado con p < 0,05 y |r| ≥ 0,10",
            "color": ACCENT_NEUTRAL,
        },
        {
            "valor": f"{num(idh1['LGG'] - idh1['GBM'], 0)} pp",
            "etiqueta": "Brecha de IDH1",
            "detalle": f"Mutado en {pct(idh1['LGG'], 1)} de LGG frente a {pct(idh1['GBM'], 1)} de GBM",
            # La franja parte los dos colores porque la cifra es la distancia
            # entre ellos, no una propiedad de uno solo.
            "color": f"linear-gradient(180deg, {COLOR_LGG} 0 50%, {COLOR_GBM} 50% 100%)",
        },
    ]


def _enlace_pestana(
    tab_id: str, nombre: str | None = None, descripcion: str | None = None
) -> html.Button:
    """Fila clicable que lleva a otra pestaña.

    Es un `<button>` de verdad, no una tarjeta decorativa: antes estas fichas
    se levantaban al pasar el ratón, invitaban al clic y no hacían nada. El id
    es de tipo diccionario para que un solo callback (`ir_a_pestana` en app.py)
    atienda a todas mediante pattern matching.

    El nombre y la descripción salen de utils/navegacion.py salvo que se pasen
    a mano, que es lo que hace la tira de documentación con su «Ir a».
    """
    nombre = nombre or ETIQUETAS[tab_id]
    descripcion = descripcion or DESCRIPCIONES[tab_id]
    return html.Button(
        [
            html.I(className=f"bi {ICONOS[tab_id]} bloque-icono"),
            html.Span(
                [
                    html.Span(nombre, className="bloque-enlace-nombre"),
                    html.Span(descripcion, className="bloque-enlace-texto"),
                ],
                className="bloque-enlace-cuerpo",
            ),
            html.I(className="bi bi-arrow-right bloque-flecha"),
        ],
        id={"type": "ir-a-pestana", "index": tab_id},
        n_clicks=0,
        className="bloque-enlace",
    )


def _bloque(bloque: dict) -> dbc.Card:
    """Una de las tres etapas de la guía, con sus pestañas dentro."""
    return card(
        [
            html.Div(
                [
                    html.Span(bloque["numero"], className="bloque-numero"),
                    html.Span(bloque["titulo"], className="bloque-titulo"),
                ],
                className="bloque-encabezado",
            ),
            html.P(bloque["resumen"], className="bloque-resumen"),
            html.Div(
                [_enlace_pestana(tab_id) for tab_id in bloque["pestanas"]],
                className="bloque-enlaces",
            ),
        ],
        className="bloque-card",
    )


def _tira_documentacion() -> dbc.Card:
    """Documentación va aparte: no es una etapa del trabajo, es su respaldo."""
    return card(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.I(className="bi bi-book bloque-icono"),
                                html.Span("Documentación", className="bloque-titulo"),
                            ],
                            className="bloque-encabezado",
                        ),
                        html.P(
                            "El desarrollo estadístico completo vive en el Jupyter "
                            "Book del proyecto, con el detalle de cada prueba y el "
                            "código que la produce.",
                            className="bloque-resumen",
                        ),
                    ],
                    lg=8,
                ),
                dbc.Col(
                    html.Div(
                        _enlace_pestana(
                            "tab-documentacion",
                            nombre="Ir a Documentación",
                            descripcion="El Jupyter Book y el repositorio del "
                                        "proyecto.",
                        ),
                        className="bloque-enlaces",
                    ),
                    lg=4,
                ),
            ],
            className="align-items-center g-3",
        ),
        className="bloque-card tira-documentacion",
    )


def _ficha_dataset() -> dbc.Card:
    """Ficha técnica del dataset."""
    return card(
        [
            bullet_list(
                [
                    html.Span(
                        [html.B("Nombre: "),
                         "Glioma Grading Clinical and Mutation Features"]
                    ),
                    html.Span(
                        [html.B("Fuente: "),
                         enlace_externo("UCI Machine Learning Repository", URL_DATASET)]
                    ),
                    html.Span([html.B("Muestra: "), "839 pacientes"]),
                    html.Span(
                        [html.B("Atributos: "),
                         "23 (20 genes con alta frecuencia de mutación y 3 "
                         "variables clínicas)"]
                    ),
                    html.Span([html.B("Objetivo: "), "clasificación binaria LGG vs GBM"]),
                    html.Span(
                        [html.B("Naturaleza: "),
                         "tabular, multivariada (numérica y categórica), sin "
                         "valores faltantes"]
                    ),
                ],
            ),
            callout(
                "La variable objetivo es binaria: 1 si el paciente presenta "
                "glioblastoma multiforme (GBM), 0 si el glioma es de bajo "
                "grado (LGG).",
                titulo="Variable a predecir",
            ),
        ],
        titulo="Ficha técnica del dataset",
        tono="acento",
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def layout() -> html.Div:
    """Layout de la pestaña de introducción."""
    return html.Div(
        [
            _portada(),
            kpi_row(_kpis()),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Los gliomas son el tumor cerebral primario más "
                                    "común en adultos y se clasifican principalmente "
                                    "en gliomas de bajo grado (LGG) y glioblastoma "
                                    "multiforme (GBM). La diferencia no es de matiz: "
                                    "cambia el pronóstico, la agresividad del "
                                    "tratamiento y el seguimiento."
                                ),
                                paragraph(
                                    "Los criterios histológicos e imagenológicos han "
                                    "sido la base del diagnóstico, pero la "
                                    "caracterización biomolecular se ha vuelto "
                                    "imprescindible para decidir el tratamiento idóneo."
                                ),
                                callout(
                                    "La secuenciación genética completa tiene un coste "
                                    "elevado para los sistemas de salud y para los "
                                    "pacientes. Ese gasto es lo que vuelve valiosa "
                                    "cualquier respuesta a la pregunta de qué merece "
                                    "la pena medir.",
                                    titulo="El cuello de botella",
                                ),
                                paragraph(
                                    "Este tablero aborda la primera mitad del desafío: "
                                    "identificar qué variables clínicas y qué "
                                    "mutaciones separan realmente a los dos grupos, y "
                                    "con qué fuerza. Es el paso que decide qué merece "
                                    "la pena medir antes de construir cualquier modelo."
                                ),
                            ],
                            titulo="El problema en una página",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(_ficha_dataset(), lg=5, className="mb-4"),
                ]
            ),
            section_title("Cómo recorrer el tablero"),
            dbc.Row(
                [
                    dbc.Col(_bloque(bloque), lg=4, md=6, xs=12, className="mb-3")
                    for bloque in BLOQUES
                    if bloque["en_guia"]
                ],
                className="g-3",
            ),
            _tira_documentacion(),
        ],
        className="vista-pestana",
    )
