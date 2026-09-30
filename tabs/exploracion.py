"""
Pestaña 7 · Exploración
El módulo interactivo del tablero: el visitante elige cualquiera de las 23
predictoras y la ve sola (univariado) o enfrentada al grado (bivariado).

El resto del tablero cuenta un recorrido fijo y narrado; esta pestaña es la
única donde la ruta la decide quien mira. Por eso aquí no se repiten los
hallazgos de Resultados: se da el instrumento para encontrarlos.

Toda la estadística vive en utils/data_loader.py y todas las figuras en
utils/figures.py. Este archivo solo decide qué se enseña y dónde.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from utils.formato import num, p_valor
from utils.components import (
    callout,
    card,
    chip_tipo,
    banda_oscura,
    desplegable,
    graph_card,
    lectura_guiada,
    page_header,
    paragraph,
    section_title,
    series_chips,
    stat_list,
)
from utils.data_loader import (
    ALFA,
    AMBITO_NOMBRES,
    EFECTO_MINIMO,
    ficha_variable,
    lectura_bivariada,
    lectura_univariada,
    opciones_variables,
    prueba_bivariada,
    resumen_univariado,
    tamanos_particion,
)
from utils.figures import fig_bivariada, fig_univariada

ID_VARIABLE = "exp-variable"
ID_AMBITO = "exp-ambito"
ID_SUBTAB = "exp-subtab"
ID_CONTENIDO = "exp-contenido"

TAB_UNIVARIADO = "exp-tab-univariado"
TAB_BIVARIADO = "exp-tab-bivariado"

VARIABLE_INICIAL = "Age_at_diagnosis"


# --------------------------------------------------------------------------- #
# Controles
# --------------------------------------------------------------------------- #
def _controles() -> dbc.Card:
    """Selector de variable y de conjunto de datos, en una sola tarjeta."""
    tamanos = tamanos_particion()
    return card(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("Variable a explorar", className="control-label"),
                        dcc.Dropdown(
                            id=ID_VARIABLE,
                            options=opciones_variables(),
                            value=VARIABLE_INICIAL,
                            clearable=False,
                            searchable=True,
                            className="selector-variable",
                        ),
                    ],
                    lg=7,
                    className="mb-3 mb-lg-0",
                ),
                dbc.Col(
                    [
                        html.Div("Conjunto de datos", className="control-label"),
                        dcc.RadioItems(
                            id=ID_AMBITO,
                            options=[
                                {"label": f" Entrenamiento ({tamanos['train']})",
                                 "value": "train"},
                                {"label": f" Prueba ({tamanos['test']})", "value": "test"},
                                {"label": f" Completo ({tamanos['full']})", "value": "full"},
                            ],
                            value="train",
                            inline=True,
                            className="ambito-radio",
                            inputClassName="me-1",
                            labelClassName="me-4",
                        ),
                    ],
                    lg=5,
                ),
            ],
            className="align-items-end g-3",
        ),
        className="controles-card mb-4",
    )


# --------------------------------------------------------------------------- #
# Univariado
# --------------------------------------------------------------------------- #
def _ficha(variable: str) -> dbc.Card:
    """Diccionario de datos recortado a la variable elegida.

    El marco teórico sigue mostrando la tabla de operacionalización completa.
    Esto no la sustituye: la trae aquí en la dosis que hace falta para leer el
    gráfico que está al lado, sin cambiar de pestaña.
    """
    ficha = ficha_variable(variable)
    return card(
        [
            html.Div(chip_tipo(ficha["tipo_nombre"]), className="mb-3"),
            paragraph(ficha["descripcion"]),
            stat_list(
                [
                    ("Nombre en el dataset", ficha["variable"]),
                    ("Codificación", ficha["codificacion"]),
                ]
            ),
            html.Div(ficha["papel"], className="ficha-papel"),
        ],
        titulo=f"Ficha de {ficha['nombre']}",
        className="mb-4",
        tono="acento",
    )


def _panel_univariado(variable: str, ambito: str) -> html.Div:
    """Distribución de la variable, sus descriptivos y su lectura."""
    ficha = ficha_variable(variable)
    resumen = resumen_univariado(variable, ambito)

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_univariada(variable, ambito),
                            f"Distribución de {ficha['nombre']}",
                            f"Sobre el {AMBITO_NOMBRES[ambito]}, sin cruzar con el grado.",
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        card(
                            stat_list(resumen["filas"]),
                            titulo="Resumen estadístico",
                            subtitulo=f"{resumen['n']} pacientes en la selección actual",
                            tono="acento",
                        ),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        lectura_guiada(lectura_univariada(variable, ambito)),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(_ficha(variable), lg=5),
                ]
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Bivariado
# --------------------------------------------------------------------------- #
def _panel_pruebas(variable: str, ambito: str) -> dbc.Accordion:
    """Las pruebas aplicadas, plegadas como en los desplegables del libro."""
    prueba = prueba_bivariada(variable, ambito)
    significativa = prueba["significativa"]
    p_texto = p_valor(prueba["p_valor"])

    if significativa:
        veredicto = (
            f"Se rechaza la hipótesis nula ({p_texto}) y la magnitud supera el "
            f"umbral del proyecto ({prueba['magnitud_nombre']} = "
            f"{num(prueba['magnitud'], 2)} ≥ {num(EFECTO_MINIMO, 2)}): la "
            "asociación es distinguible del azar y además tiene tamaño "
            "suficiente para que merezca la pena mirarla."
        )
    elif prueba["p_valor"] < ALFA:
        veredicto = (
            f"Se rechaza la hipótesis nula ({p_texto}), pero la magnitud queda por "
            f"debajo del umbral del proyecto ({prueba['magnitud_nombre']} = "
            f"{num(prueba['magnitud'], 2)}, frente a {num(EFECTO_MINIMO, 2)}). "
            "Significancia no es magnitud: con esta muestra, una diferencia "
            "pequeña basta para salir significativa."
        )
    else:
        veredicto = (
            f"No se rechaza la hipótesis nula ({p_texto}): los datos son "
            "compatibles con que esta variable y el grado sean independientes."
        )

    contenido = [
        paragraph(prueba["motivo"]),
        stat_list([("Prueba aplicada", prueba["prueba"]), *prueba["detalle"]]),
        callout(veredicto, titulo="Veredicto"),
    ]
    if prueba["nota_metodo"]:
        contenido.append(
            callout(prueba["nota_metodo"], titulo="Cómo se calculó, y por qué así")
        )

    return desplegable(
        f"Prueba estadística aplicada · {prueba['prueba']}", contenido
    )


def _panel_bivariado(variable: str, ambito: str) -> html.Div:
    """La variable frente al grado, con su lectura y sus pruebas."""
    ficha = ficha_variable(variable)
    es_numerica = ficha["tipo"] == "numerica"

    nota = (
        "Cada grado se normaliza a su propio 100 %, porque los dos grupos no "
        "tienen el mismo tamaño."
        if es_numerica
        else "El porcentaje se calcula dentro de cada categoría: la pregunta es "
             "qué proporción de ese grupo acaba siendo GBM."
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            fig_bivariada(variable, ambito),
                            f"{ficha['nombre']} frente al grado tumoral",
                            nota,
                        ),
                        lg=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(lectura_bivariada(variable, ambito)),
                        lg=5,
                        className="mb-4",
                    ),
                ]
            ),
            _panel_pruebas(variable, ambito),
        ]
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def layout() -> html.Div:
    """Layout de la pestaña de exploración."""
    return html.Div(
        [
            page_header(
                "Exploración",
                "Elige una variable y míralas por dentro: su distribución, sus "
                "descriptivos y la prueba que decide si separa los dos grados.",
                "bi-sliders",
            ),
            banda_oscura(
                "Exploración de variables",
                "Alterna entre la lectura univariada y la comparación contra el "
                "grado tumoral sin salir del mismo módulo. La forma del gráfico "
                "la decide el tipo de dato: una variable continua pide una "
                "distribución y una categórica pide una comparación de "
                "proporciones.",
                eyebrow="Módulo interactivo",
            ),
            _controles(),
            html.Div(series_chips(), className="mb-3"),
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Análisis univariado",
                        tab_id=TAB_UNIVARIADO,
                        label_class_name="sub-tab",
                        active_label_class_name="sub-tab-active",
                    ),
                    dbc.Tab(
                        label="Análisis bivariado",
                        tab_id=TAB_BIVARIADO,
                        label_class_name="sub-tab",
                        active_label_class_name="sub-tab-active",
                    ),
                ],
                id=ID_SUBTAB,
                active_tab=TAB_UNIVARIADO,
                className="sub-tabs",
            ),
            dcc.Loading(
                html.Div(
                    _panel_univariado(VARIABLE_INICIAL, "train"),
                    id=ID_CONTENIDO,
                ),
                type="dot",
            ),
            section_title("Dónde seguir"),
            callout(
                "Este módulo responde de una en una. El recorrido completo, con "
                "las variables comparadas entre sí y la matriz de "
                "multicolinealidad, está en Resultados; la tabla de "
                "operacionalización de las 23 predictoras y el detalle de cada "
                "prueba, en Marco teórico.",
                titulo="Cómo se relaciona con el resto del tablero",
            ),
        ],
        className="vista-pestana",
    )


# --------------------------------------------------------------------------- #
# Callbacks propios de la pestaña
# --------------------------------------------------------------------------- #
@callback(
    Output(ID_CONTENIDO, "children"),
    Input(ID_VARIABLE, "value"),
    Input(ID_AMBITO, "value"),
    Input(ID_SUBTAB, "active_tab"),
)
def actualizar_exploracion(variable: str, ambito: str, subtab: str):
    """Reconstruye el panel con la variable, el conjunto y la vista elegidos."""
    variable = variable or VARIABLE_INICIAL
    ambito = ambito or "train"
    if subtab == TAB_BIVARIADO:
        return _panel_bivariado(variable, ambito)
    return _panel_univariado(variable, ambito)
