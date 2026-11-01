"""
Pestaña 10 · Predicción
El visitante describe un paciente —edad, género, grupo racial y estado de los
20 genes— y ve qué probabilidad de glioblastoma le asigna cada uno de los tres
modelos finales y por qué.

Las predicciones y sus explicaciones (valores SHAP exactos) salen de
utils/modelo.py, que reproduce los modelos del libro con numpy a partir de
modelos/modelos_finales.json. Aquí no se entrena nada.

Es una herramienta docente: muestra cómo razona un modelo entrenado con 671
pacientes de TCGA, no sustituye a ningún diagnóstico.
"""

from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from utils.components import (
    altura_grafico,
    icono,
    banda_oscura,
    callout,
    card,
    graph_card,
    lectura_guiada,
    page_header,
    paragraph,
    section_title,
    series_chips,
)
from utils.config import GENE_DESCRIPCION, GENE_FEATURES, GRADE_LABELS, RACE_LABELS
from utils.data_loader import get_dataframe
from utils.figures_modelo import (
    fig_arbol,
    fig_explicacion,
    fig_curva_logistica,
    fig_suma_xgb,
    fig_votacion,
    pregunta_nodo,
    respuesta_nodo,
)
from utils.formato import lista_y, num, pct
from utils.modelo import (
    NOMBRE_MODELO,
    ORDEN_MODELOS,
    UMBRAL,
    aportes_arboles,
    arbol_mas_influyente,
    arbol_representativo,
    logodds_logistica,
    camino_paciente,
    cargar_modelos,
    cargar_resumen,
    explicar,
    obtener_arbol,
    predecir,
    probas_arboles,
    umbral_en_unidades,
)
from utils.theme import COLOR_GBM, COLOR_LGG, STATUS_WARNING

ID_PERFIL = "pred-perfil"
ID_EDAD = "pred-edad"
ID_GENERO = "pred-genero"
ID_RAZA = "pred-raza"
ID_GENES_PANEL = "pred-genes-panel"
ID_GENES_RESTO = "pred-genes-resto"
ID_MODELO = "pred-modelo"
ID_CARGADO = "pred-cargado"
ID_PROBAS = "pred-probas"
ID_ANIMACION = "pred-animacion"
ID_VEREDICTO = "pred-veredicto"
ID_REAL = "pred-real"
ID_EXPLICACION = "pred-explicacion"
ID_LECTURA = "pred-lectura"
ID_CONTEXTO = "pred-contexto"
ID_CONTEXTO_LECTURA = "pred-contexto-lectura"
ID_BOTON = "pred-boton"
ID_PREDICHO = "pred-predicho"
ID_PENDIENTE = "pred-pendiente"
ID_EDAD_VALOR = "pred-edad-valor"
ID_GENES_CUENTA = "pred-genes-cuenta"
ID_ARBOL = "pred-arbol"
ID_ARBOL_REP = "pred-arbol-rep"
ID_VOTACION = "pred-votacion"
ID_ARBOL_FIG = "pred-arbol-fig"
ID_CAMINO = "pred-camino"
ID_SEC_RF = "pred-sec-rf"
ID_SEC_XGB = "pred-sec-xgb"
ID_SEC_LOG = "pred-sec-log"
ID_XGB_SUMA = "pred-xgb-suma"
ID_XGB_ARBOL = "pred-xgb-arbol"
ID_XGB_REP = "pred-xgb-rep"
ID_XGB_ARBOL_FIG = "pred-xgb-arbol-fig"
ID_XGB_CAMINO = "pred-xgb-camino"
ID_LOG_CURVA = "pred-log-curva"
ID_LOG_RESUMEN = "pred-log-resumen"

# Los tres genes del panel reducido (con la edad, conservan casi toda la
# capacidad del modelo completo; ver la pestaña Modelos).
GENES_PANEL = ["IDH1", "TP53", "PTEN"]

# Perfiles de ejemplo. Los dos primeros son prototipos de cada grado; el
# tercero es el peor error del modelo en prueba (paciente 576), un GBM con el
# perfil mutacional de un astrocitoma IDH-mutado.
PERFILES = {
    "lgg": {"nombre": "Perfil típico de LGG (oligodendroglioma)",
            "Age_at_diagnosis": 38, "Gender": 0, "Race": 0,
            "genes": ["IDH1", "CIC", "FUBP1"]},
    "gbm": {"nombre": "Perfil típico de GBM (IDH sin mutar)",
            "Age_at_diagnosis": 64, "Gender": 0, "Race": 0,
            "genes": ["PTEN", "TP53", "NF1"]},
    "vacio": {"nombre": "Sin mutaciones, 50 años",
              "Age_at_diagnosis": 50, "Gender": 0, "Race": 0, "genes": []},
}
PACIENTE_DIFICIL = 576
PERFIL_INICIAL = "gbm"


def _genes_ordenados() -> list[str]:
    """Genes fuera del panel, de más a menos importantes según SHAP."""
    orden = [f["variable"] for f in cargar_resumen()["importancia"]]
    resto = [g for g in orden if g in GENE_FEATURES and g not in GENES_PANEL]
    return resto + [g for g in GENE_FEATURES if g not in resto and g not in GENES_PANEL]


def _opciones_perfil() -> list[dict]:
    opciones = [{"label": p["nombre"], "value": clave} for clave, p in PERFILES.items()]
    opciones.append({"label": f"Caso difícil: paciente {PACIENTE_DIFICIL} de prueba (GBM con IDH1 mutado)",
                     "value": f"test-{PACIENTE_DIFICIL}"})
    prueba = get_dataframe("test")
    for idx, fila in prueba.sort_index().iterrows():
        opciones.append({
            "label": f"Paciente {idx} de prueba · {GRADE_LABELS[int(fila['Grade'])]} · "
                     f"{num(fila['Age_at_diagnosis'], 0)} años",
            "value": f"test-{idx}",
        })
    return opciones


def _perfil(valor: str) -> dict:
    """Paciente como diccionario de variables del dataset (y su grado real si lo hay)."""
    if valor and valor.startswith("test-"):
        idx = int(valor.split("-")[1])
        fila = get_dataframe("test").loc[idx]
        paciente = {"Age_at_diagnosis": float(fila["Age_at_diagnosis"]),
                    "Gender": int(fila["Gender"]), "Race": int(fila["Race"]),
                    **{g: int(fila[g]) for g in GENE_FEATURES}}
        return {"paciente": paciente, "grado": int(fila["Grade"]), "indice": idx}
    p = PERFILES.get(valor, PERFILES[PERFIL_INICIAL])
    paciente = {"Age_at_diagnosis": p["Age_at_diagnosis"], "Gender": p["Gender"],
                "Race": p["Race"], **{g: int(g in p["genes"]) for g in GENE_FEATURES}}
    return {"paciente": paciente, "grado": None, "indice": None}


def _paciente_de_formulario(edad, genero, raza, panel, resto) -> dict:
    mutados = set(panel or []) | set(resto or [])
    return {"Age_at_diagnosis": float(edad if edad is not None else 50),
            "Gender": int(genero or 0), "Race": int(raza or 0),
            **{g: int(g in mutados) for g in GENE_FEATURES}}


# --------------------------------------------------------------------------- #
# Formulario
# --------------------------------------------------------------------------- #
def _chips_genes(id_, genes, marcados) -> dbc.Checklist:
    return dbc.Checklist(
        id=id_,
        options=[{"label": html.Span(g, title=GENE_DESCRIPCION.get(g, "")), "value": g}
                 for g in genes],
        value=[g for g in genes if g in marcados],
        inline=True,
        className="chips-genes",
        input_class_name="chip-gen-input",
        label_class_name="chip-gen",
        label_checked_class_name="chip-gen-activo",
    )


def _campo(etiqueta, hijos, extra=None, className="") -> html.Div:
    """Un campo del formulario: etiqueta arriba (con un dato a la derecha) y el control."""
    cabecera = [html.Span(etiqueta, className="campo-label")]
    if extra is not None:
        cabecera.append(extra)
    return html.Div([html.Div(cabecera, className="campo-fila"), *hijos],
                    className=f"form-campo {className}".strip())


def _formulario() -> dbc.Card:
    inicial = _perfil(PERFIL_INICIAL)["paciente"]
    marcados = [g for g in GENE_FEATURES if inicial[g] == 1]
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Perfil del paciente", className="form-titulo"),
                _campo("Punto de partida", [
                    dcc.Dropdown(id=ID_PERFIL, options=_opciones_perfil(), value=PERFIL_INICIAL,
                                 clearable=False, searchable=True,
                                 className="selector-variable selector-app"),
                ]),
                _campo("Edad al diagnóstico", [
                    dcc.Slider(id=ID_EDAD, min=10, max=90, step=1,
                               value=inicial["Age_at_diagnosis"],
                               marks={v: str(v) for v in range(10, 91, 20)},
                               className="slider-app"),
                ], extra=html.Span(f"{int(inicial['Age_at_diagnosis'])} años", id=ID_EDAD_VALOR,
                                   className="campo-valor")),
                html.Div(
                    [
                        _campo("Género", [
                            dbc.RadioItems(
                                id=ID_GENERO,
                                options=[{"label": "Hombre", "value": 0},
                                         {"label": "Mujer", "value": 1}],
                                value=inicial["Gender"], inline=True,
                                className="segmentado",
                                input_class_name="segmentado-input",
                                label_class_name="segmentado-op",
                                label_checked_class_name="segmentado-activo",
                            ),
                        ]),
                        _campo("Grupo racial", [
                            dcc.Dropdown(id=ID_RAZA,
                                         options=[{"label": v, "value": k}
                                                  for k, v in RACE_LABELS.items()],
                                         value=inicial["Race"], clearable=False,
                                         searchable=False,
                                         className="selector-variable selector-app"),
                        ], className="form-campo-ancho"),
                    ],
                    className="form-fila",
                ),
                _campo("Genes mutados", [
                    html.Div("Panel reducido: con la edad, concentran casi toda la capacidad "
                             "del modelo.", className="control-ayuda"),
                    _chips_genes(ID_GENES_PANEL, GENES_PANEL, marcados),
                    html.Div("Resto del panel, de más a menos peso. Pasa el cursor para ver "
                             "qué hace cada gen.", className="control-ayuda mt-3"),
                    _chips_genes(ID_GENES_RESTO, _genes_ordenados(), marcados),
                ], extra=html.Span(f"{len(marcados)} marcados", id=ID_GENES_CUENTA,
                                   className="form-bloque-cuenta")),
                html.Div(
                    [
                        html.Div(id=ID_PENDIENTE, className="pred-pendiente"),
                        dbc.Button(
                            [html.Span("Predecir", className="boton-predecir-texto"),
                             html.Span(icono("arrow_forward"), className="boton-predecir-icono")],
                            id=ID_BOTON, n_clicks=0, className="boton-predecir",
                        ),
                    ],
                    className="pred-acciones",
                ),
            ],
            className="form-cuerpo",
        ),
        className="soft-card formulario-prediccion",
    )


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
def _veredicto(probas: dict) -> html.Div:
    gbm = [m for m in ORDEN_MODELOS if probas[m] >= UMBRAL]
    if len(gbm) == 3:
        titulo, color, fondo = "Los tres modelos apuntan a glioblastoma (GBM)", COLOR_GBM, "#FCEEE9"
    elif not gbm:
        titulo, color, fondo = ("Los tres modelos apuntan a glioma de bajo grado (LGG)",
                                COLOR_LGG, "#E8F0FB")
    else:
        titulo = ("Los modelos no coinciden: " + lista_y(NOMBRE_MODELO[m] for m in gbm)
                  + (" dice" if len(gbm) == 1 else " dicen") + " GBM")
        color, fondo = STATUS_WARNING, "#FBF3E0"
    minimo, maximo = min(probas.values()), max(probas.values())
    if maximo < 0.35 or minimo > 0.65:
        detalle = "Predicción firme: las tres probabilidades están lejos del umbral."
    elif minimo >= 0.35 and maximo <= 0.65:
        detalle = ("Zona de incertidumbre: las tres probabilidades rondan el 50 %. Un perfil "
                   "así no se puede clasificar con confianza solo con estas variables.")
    else:
        detalle = (f"Las probabilidades van del {pct(minimo * 100, 0)} al {pct(maximo * 100, 0)}: "
                   "alguna está cerca del umbral, así que conviene leer la explicación.")
    return html.Div(
        [html.Div([html.Span(className="veredicto-punto", style={"background": color}),
                   html.Span(titulo, className="veredicto-titulo")],
                  className="veredicto", style={"background": fondo}),
         html.Div(detalle, className="veredicto-texto")],
    )


def _barras(probas: dict) -> html.Div:
    """Probabilidad de GBM de cada modelo: cifra grande y barra, en mosaicos.

    Son HTML y no Plotly para poder animarlos: la barra crece desde cero (CSS,
    `prob-barra`) y la cifra cuenta hasta su valor (assets/contador.js, clase
    `cifra-animada`) cada vez que se pulsa Predecir.
    """
    mosaicos = []
    for m in ORDEN_MODELOS:
        p = probas[m]
        es_gbm = p >= UMBRAL
        color = COLOR_GBM if es_gbm else COLOR_LGG
        mosaicos.append(html.Div(
            [
                html.Div(NOMBRE_MODELO[m], className="prob-mosaico-nombre"),
                html.Div(
                    [html.Span([html.Span(f"{round(p * 100)}", className="cifra-animada"), " %"],
                               className="prob-mosaico-cifra"),
                     html.Span("GBM" if es_gbm else "LGG", className="prob-grado",
                               style={"color": "#B4512F" if es_gbm else "#2F6DB8"})],
                    className="prob-mosaico-fila",
                ),
                html.Div(
                    [html.Div(className="prob-barra",
                              style={"width": f"{p * 100:.1f}%", "background": color}),
                     html.Div(className="prob-umbral")],
                    className="prob-riel prob-riel-fino",
                ),
            ],
            className="prob-mosaico",
        ))
    return html.Div(mosaicos, className="prob-mosaicos")


def _real(cargado: dict | None, paciente: dict, probas: dict):
    """Grado real si el formulario sigue siendo exactamente un paciente de prueba."""
    if not cargado or cargado.get("grado") is None or cargado["paciente"] != paciente:
        return None
    grado = cargado["grado"]
    aciertos = [m for m in ORDEN_MODELOS if int(probas[m] >= UMBRAL) == grado]
    texto = ("los tres modelos aciertan" if len(aciertos) == 3
             else "ningún modelo acierta" if not aciertos
             else f"acierta{'n' if len(aciertos) > 1 else ''} {lista_y(NOMBRE_MODELO[m] for m in aciertos)}")
    color = COLOR_GBM if grado == 1 else COLOR_LGG
    return html.Div(
        [html.Span(className="chip-dot", style={"background": color}),
         html.Span(f"Diagnóstico real en TCGA del paciente {cargado['indice']}: "
                   f"{GRADE_LABELS[grado]} · {texto}")],
        className="real-badge",
    )


def _nombre(v: str, paciente: dict) -> str:
    if v == "Age_at_diagnosis":
        return f"la edad, {num(paciente['Age_at_diagnosis'], 0)} años"
    if v == "Gender":
        return "el género"
    if v == "Race_binary":
        return "el grupo racial"
    return f"{v} {'mutado' if paciente.get(v) else 'sin mutar'}"


def _lectura(explicacion: dict, paciente: dict, modelo: str, proba: float) -> list:
    c = explicacion["contribuciones"]
    escala = explicacion["escala"]
    fmt = (lambda v: num(v, 2, signo=True)) if escala == "log-odds" else \
          (lambda v: num(v * 100, 1, signo=True) + " puntos")
    # Se ignoran las contribuciones mínimas: nombrar "MUC16 sin mutar (−0,02)"
    # como algo que empuja hacia LGG solo distrae.
    minimo = 0.05 if escala == "log-odds" else 0.01
    hacia_gbm = sorted([v for v in c if c[v] >= minimo], key=lambda v: -c[v])[:3]
    hacia_lgg = sorted([v for v in c if c[v] <= -minimo], key=lambda v: c[v])[:3]
    lineas = [
        f"{NOMBRE_MODELO[modelo]} asigna a este perfil una probabilidad de GBM del "
        f"{pct(proba * 100, 0)}."
    ]
    for lista, grado in [(hacia_gbm, "GBM"), (hacia_lgg, "LGG")]:
        if lista:
            lineas.append(f"Empuja hacia {grado}: " + lista_y(
                f"{_nombre(v, paciente)} ({fmt(c[v])})" for v in lista) + ".")
        else:
            lineas.append(f"Ninguna variable empuja de forma apreciable hacia {grado}.")
    unidad = ("en log-odds: +1 multiplica por 2,7 las probabilidades a favor de GBM"
              if escala == "log-odds" else "en puntos porcentuales de probabilidad")
    lineas.append(f"Las contribuciones están {unidad}. Sumadas al valor base dan "
                  "exactamente la predicción del modelo.")
    return lineas


# --------------------------------------------------------------------------- #
# Dentro del Random Forest
# --------------------------------------------------------------------------- #
def _n_arboles() -> int:
    return len(cargar_modelos()["modelos"]["random_forest"]["arboles"])


def _lista_camino(camino: dict) -> html.Ol:
    """El camino del paciente en el árbol, paso a paso."""
    menor = camino.get("menor", False)
    pasos = [
        html.Li([html.Span(pregunta_nodo(p["variable"], p["umbral"], menor)),
                 html.B(respuesta_nodo(p["variable"], p["izquierda"]),
                        className="camino-respuesta")])
        for p in camino["pasos"]
    ]
    if camino.get("escala") == "log-odds":
        hacia_gbm = camino["proba"] > 0
        texto = (f"Hoja: suma {num(camino['proba'], 3, signo=True)} a los log-odds, "
                 f"un pequeño empujón hacia {'GBM' if hacia_gbm else 'LGG'}.")
    else:
        hacia_gbm = camino["proba"] >= UMBRAL
        texto = (f"Hoja con {num(camino['cobertura'], 0)} casos de entrenamiento: "
                 f"{pct(camino['proba'] * 100, 0)} GBM. Este árbol dice "
                 f"{'GBM' if hacia_gbm else 'LGG'}.")
    color = COLOR_GBM if hacia_gbm else COLOR_LGG
    pasos.append(html.Li(
        [html.Span(className="chip-dot", style={"background": color}), html.Span(texto)],
        className="camino-hoja",
    ))
    return html.Ol(pasos, className="camino-lista")


def _vistas_bosque(paciente: dict, numero: int):
    """Histograma de los árboles, dibujo del árbol elegido y su camino."""
    indice = min(max(int(numero or 1), 1), _n_arboles()) - 1
    camino = camino_paciente(paciente, indice)
    return (
        fig_votacion(probas_arboles(paciente), indice),
        fig_arbol(obtener_arbol(indice), camino, cargar_modelos()["predictoras"],
                  umbral_en_unidades),
        _lista_camino(camino),
    )


def _seccion_bosque(paciente: dict) -> html.Div:
    n = _n_arboles()
    numero = arbol_representativo(paciente) + 1
    votacion, arbol, lista = _vistas_bosque(paciente, numero)
    return html.Div(
        [
            section_title("Cómo vota el Random Forest"),
            paragraph(
                f"El Random Forest reúne {n} árboles de decisión. Cada uno se entrenó con "
                "una muestra distinta de pacientes y de variables, así que cada uno da su "
                "propia probabilidad de GBM. La predicción del bosque es la media de todas."
            ),
            graph_card(votacion, f"Lo que dice cada uno de los {n} árboles",
                       "La línea negra es la media, es decir, la predicción del Random "
                       "Forest. La discontinua marca el árbol que se dibuja abajo.",
                       graph_id=ID_VOTACION),
            card(
                [
                    html.Div("Árbol que se dibuja", className="control-label"),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Slider(
                                id=ID_ARBOL, min=1, max=n, step=1, value=numero,
                                marks={v: str(v) for v in [1, *range(50, n + 1, 50)]},
                                tooltip={"placement": "bottom", "always_visible": True,
                                         "template": "árbol {value}"},
                            ), md=9),
                            dbc.Col(dbc.Button("Árbol más representativo", id=ID_ARBOL_REP,
                                               n_clicks=0, className="boton-secundario"),
                                    md=3, className="text-md-end mt-3 mt-md-0"),
                        ],
                        className="align-items-center",
                    ),
                    html.Div("Al predecir se elige el árbol cuya probabilidad está más cerca "
                             "de la media del bosque.", className="control-ayuda mt-2"),
                ],
                className="controles-card my-4",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card(arbol, "El árbol completo y el camino del paciente",
                                       "Cada punto es una pregunta; los extremos son hojas, "
                                       "coloreadas por el grado que predicen. Pasa el cursor "
                                       "por un nodo para ver cuántos casos llegaron a él.",
                                       graph_id=ID_ARBOL_FIG),
                            lg=8, className="mb-4"),
                    dbc.Col(card(html.Div(lista, id=ID_CAMINO), titulo="Paso a paso",
                                 subtitulo="Las preguntas que responde el paciente en este "
                                           "árbol, de la raíz a la hoja."),
                            lg=4, className="mb-4"),
                ]
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Dentro de XGBoost
# --------------------------------------------------------------------------- #
def _n_arboles_xgb() -> int:
    return len(cargar_modelos()["modelos"]["xgboost"]["arboles"])


def _vistas_xgb(paciente: dict, numero: int):
    """Suma árbol a árbol, dibujo del árbol elegido y su camino."""
    indice = min(max(int(numero or 1), 1), _n_arboles_xgb()) - 1
    base, aportes = aportes_arboles(paciente)
    camino = camino_paciente(paciente, indice, "xgboost")
    arbol = fig_arbol(obtener_arbol(indice, "xgboost"), camino,
                      cargar_modelos()["predictoras"], umbral_en_unidades)
    return fig_suma_xgb(base, aportes, indice), arbol, _lista_camino(camino)


def _seccion_xgb(paciente: dict) -> html.Div:
    n = _n_arboles_xgb()
    numero = arbol_mas_influyente(paciente) + 1
    suma, arbol, lista = _vistas_xgb(paciente, numero)
    return html.Div(
        [
            section_title("Cómo suma XGBoost"),
            paragraph(
                f"XGBoost no promedia árboles: los suma. Parte de un valor base y cada uno "
                f"de sus {n} árboles, pequeños y de dos preguntas, corrige un poco lo que "
                "dejaron los anteriores. La línea muestra cómo llega a la predicción."
            ),
            graph_card(suma, "La predicción, árbol a árbol",
                       "Cada árbol suma o resta en log-odds; el eje lo traduce a "
                       "probabilidad. La discontinua marca el árbol que se dibuja abajo.",
                       graph_id=ID_XGB_SUMA),
            card(
                [
                    html.Div("Árbol que se dibuja", className="control-label"),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Slider(
                                id=ID_XGB_ARBOL, min=1, max=n, step=1, value=numero,
                                marks={v: str(v) for v in [1, *range(100, n + 1, 100), n]},
                                tooltip={"placement": "bottom", "always_visible": True,
                                         "template": "árbol {value}"},
                            ), md=9),
                            dbc.Col(dbc.Button("Árbol que más pesa", id=ID_XGB_REP,
                                               n_clicks=0, className="boton-secundario"),
                                    md=3, className="text-md-end mt-3 mt-md-0"),
                        ],
                        className="align-items-center",
                    ),
                    html.Div("Al predecir se elige el árbol que más mueve la predicción de "
                             "este paciente.", className="control-ayuda mt-2"),
                ],
                className="controles-card my-4",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card(arbol, "Un árbol de XGBoost y el camino del paciente",
                                       "Las hojas no dan una probabilidad: suman un valor "
                                       "pequeño a los log-odds (coral, hacia GBM; azul, "
                                       "hacia LGG).", graph_id=ID_XGB_ARBOL_FIG),
                            lg=8, className="mb-4"),
                    dbc.Col(card(html.Div(lista, id=ID_XGB_CAMINO), titulo="Paso a paso",
                                 subtitulo="Las preguntas de este árbol y lo que suma."),
                            lg=4, className="mb-4"),
                ]
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Dentro de la regresión logística
# --------------------------------------------------------------------------- #
def _resumen_logistica(z: float) -> html.Div:
    p = 1 / (1 + np.exp(-z))
    return html.Div(
        [
            html.Div([html.Div("Suma de efectos", className="prob-mosaico-nombre"),
                      html.Div(num(z, 2, signo=True), className="prob-mosaico-cifra")],
                     className="prob-mosaico"),
            html.Div([html.Div("Probabilidad de GBM", className="prob-mosaico-nombre"),
                      html.Div(pct(p * 100, 0), className="prob-mosaico-cifra")],
                     className="prob-mosaico"),
            html.Div("Por encima de 0 la suma da más del 50 %. Cerca de 0 la curva es "
                     "empinada y un solo cambio mueve mucho la probabilidad; en los "
                     "extremos se aplana y casi no la mueve.", className="prob-nota"),
        ],
        className="log-resumen",
    )


def _seccion_logistica(paciente: dict) -> html.Div:
    z = logodds_logistica(paciente)
    return html.Div(
        [
            section_title("Cómo pesa la regresión logística"),
            paragraph(
                "La regresión logística suma un efecto por variable y convierte esa suma "
                "en probabilidad con una curva en S. Los efectos son los de la cascada de "
                "arriba; aquí se ve dónde cae el perfil sobre la curva."
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_curva_logistica(z), "La curva en S",
                                       "Puntos de fondo: los 168 pacientes de prueba en el "
                                       "color de su grado real.", graph_id=ID_LOG_CURVA),
                            lg=8, className="mb-4"),
                    dbc.Col(html.Div(_resumen_logistica(z), id=ID_LOG_RESUMEN),
                            lg=4, className="mb-4"),
                ]
            ),
        ]
    )


def _seccion_como_decide(paciente: dict, modelo: str = "logistica") -> html.Div:
    """Las tres secciones; solo se ve la del modelo elegido arriba."""
    def visible(m):
        return {} if m == modelo else {"display": "none"}
    return html.Div(
        [
            html.Div(_seccion_logistica(paciente), id=ID_SEC_LOG, style=visible("logistica")),
            html.Div(_seccion_bosque(paciente), id=ID_SEC_RF, style=visible("random_forest")),
            html.Div(_seccion_xgb(paciente), id=ID_SEC_XGB, style=visible("xgboost")),
        ]
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _tarjeta_enjambre(proba: float) -> dbc.Card:
    """«Dónde cae este perfil» dibujado con partículas (assets/enjambre.js).

    Sustituye al gráfico de puntos de Plotly. Cada punto es uno de los 168
    pacientes de prueba, en la probabilidad de GBM que le da el modelo y en la
    fila de su grado real. Al cambiar de modelo, los puntos se desplazan a su
    nueva probabilidad (se ve quién cruza el umbral); al predecir, el perfil
    viaja a su sitio y se encienden los pacientes con una probabilidad
    parecida. La lectura escrita de debajo dice lo mismo en palabras.

    Los datos de prueba van en el propio HTML (`data-prueba`); el modelo y la
    probabilidad del perfil, en el Store `ID_CONTEXTO`, que actualiza el
    callback de predicción y lee un callback de cliente.
    """
    test = cargar_resumen()["test"]
    prueba = {"y": [int(v) for v in test["y"]]}
    for m in ORDEN_MODELOS:
        prueba[m] = [round(float(v), 4) for v in test[m]]
    n_lgg = prueba["y"].count(0)
    n_gbm = prueba["y"].count(1)
    return dbc.Card(
        dbc.CardBody([
            html.H5("Dónde cae este perfil", className="card-title-custom"),
            html.Div(
                [
                    html.Canvas(**{"role": "img",
                                   "aria-label": "Los 168 pacientes de prueba según la "
                                                 "probabilidad de GBM que les da el modelo"}),
                    html.Div(className="enjambre-punta", **{"aria-hidden": "true"}),
                ],
                id="pred-enjambre",
                className="enjambre-lienzo",
                **{"data-prueba": json.dumps(prueba, separators=(",", ":"))},
            ),
            html.Div(
                [
                    html.Span([html.I(className="enjambre-punto lgg"), f"LGG real ({n_lgg})"]),
                    html.Span([html.I(className="enjambre-punto gbm"), f"GBM real ({n_gbm})"]),
                    html.Span([html.I(className="enjambre-punto perfil"), "este perfil"]),
                ],
                className="enjambre-leyenda", **{"aria-hidden": "true"},
            ),
            html.Div(id=ID_CONTEXTO_LECTURA, className="enjambre-lectura", **{"aria-live": "polite"}),
            dcc.Store(id=ID_CONTEXTO, data={"modelo": "logistica", "p": proba}),
            html.Div("Probabilidades que da el modelo elegido arriba a los 168 pacientes de "
                     "prueba, separados por su grado real. Pasa el ratón por un punto para ver "
                     "su valor.", className="graph-note"),
        ]),
        className="soft-card tarjeta-relleno tarjeta-enjambre",
    )


def layout() -> html.Div:
    """Layout de la pestaña de predicción."""
    inicial = _perfil(PERFIL_INICIAL)["paciente"]
    probas = predecir(inicial)
    exp_ = explicar(inicial, "logistica")
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.Div("03 Los resultados · Predicción", className="titular-eyebrow"),
                         html.H2(["Describe un paciente", html.Br(),
                                  html.Strong("y mira qué predicen")], className="titular")],
                    ),
                    html.Div(
                        [html.Div("Modelo que se explica", className="titular-ayuda"),
                         dcc.RadioItems(id=ID_MODELO,
                                        options=[{"label": NOMBRE_MODELO[m], "value": m}
                                                 for m in ORDEN_MODELOS],
                                        value="logistica", inline=True,
                                        className="ambito-radio segmentado-oscuro")],
                        className="titular-selector",
                    ),
                ],
                className="titular-fila",
            ),
            callout(
                "Esta herramienta muestra cómo razonan modelos entrenados con 671 pacientes "
                "de TCGA, con un perfil clínico y mutacional muy limitado. No es un "
                "instrumento diagnóstico ni sustituye a la valoración histológica y "
                "molecular de un especialista.",
                titulo="Uso docente, no clínico",
                color=STATUS_WARNING,
            ),
            dcc.Store(id=ID_CARGADO, data=_perfil(PERFIL_INICIAL)),
            # Último paciente enviado con el botón: los resultados solo se
            # recalculan cuando cambia este, no con cada clic en el formulario.
            dcc.Store(id=ID_PREDICHO, data=inicial),
            dcc.Store(id=ID_ANIMACION),
            dbc.Row(
                [
                    dbc.Col(_formulario(), lg=5, className="mb-4"),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(_veredicto(probas), id=ID_VEREDICTO),
                                    html.Div(id=ID_REAL),
                                    html.Div(_barras(probas), id=ID_PROBAS),
                                    html.Div("Probabilidad de GBM según cada modelo. Por encima "
                                             "del 50 % (la marca de la barra) clasifica como GBM.",
                                             className="prob-nota"),
                                ],
                                className="bloque-resultados mb-4",
                            ),
                            _tarjeta_enjambre(probas["logistica"]),
                        ],
                        lg=7,
                        className="mb-4 columna-apilada",
                    ),
                ],
                className="g-4",
            ),
            section_title("Por qué lo predice"),
            html.Div(series_chips(), className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_explicacion(exp_, inicial), "Contribución de cada variable",
                                       "Valores SHAP exactos. Coral: empuja hacia GBM; azul: hacia "
                                       "LGG. El valor base es la predicción media de la cohorte.",
                                       graph_id=ID_EXPLICACION),
                            lg=7, className="mb-4"),
                    dbc.Col(html.Div(lectura_guiada(_lectura(exp_, inicial, "logistica",
                                                             probas["logistica"]),
                                                    titulo="Lectura de esta predicción"),
                                     id=ID_LECTURA),
                            lg=5, className="mb-4"),
                ]
            ),
            _seccion_como_decide(inicial),
            callout(
                [
                    paragraph(
                        "Prueba a cargar el caso difícil (paciente 576): un glioblastoma de "
                        "45 años con IDH1, TP53 y ATRX mutados. Los tres modelos lo clasifican "
                        "como LGG porque la mutación de IDH1 pesa más que todo lo demás. Es el "
                        "perfil de un astrocitoma IDH-mutado de grado 4, que la clasificación "
                        "de la OMS de 2021 ya no llama glioblastoma; distinguirlo exigiría "
                        "variables que el dataset no tiene."
                    ),
                ],
                titulo="Dónde se equivocan los modelos",
            ),
        ],
        className="vista-pestana",
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@callback(
    Output(ID_EDAD, "value"),
    Output(ID_GENERO, "value"),
    Output(ID_RAZA, "value"),
    Output(ID_GENES_PANEL, "value"),
    Output(ID_GENES_RESTO, "value"),
    Output(ID_CARGADO, "data"),
    Input(ID_PERFIL, "value"),
    prevent_initial_call=True,
)
def cargar_perfil(valor: str):
    """Rellena el formulario con el perfil elegido."""
    if not valor:
        return (no_update,) * 6
    perfil = _perfil(valor)
    p = perfil["paciente"]
    mutados = [g for g in GENE_FEATURES if p[g] == 1]
    # El slider trabaja con años enteros; el paciente guardado también, para que
    # el formulario y el paciente cargado sigan siendo "el mismo" al compararlos.
    p["Age_at_diagnosis"] = float(round(p["Age_at_diagnosis"]))
    return (p["Age_at_diagnosis"], p["Gender"], p["Race"],
            [g for g in GENES_PANEL if g in mutados],
            [g for g in mutados if g not in GENES_PANEL], perfil)


@callback(
    Output(ID_PREDICHO, "data"),
    Input(ID_BOTON, "n_clicks"),
    State(ID_EDAD, "value"),
    State(ID_GENERO, "value"),
    State(ID_RAZA, "value"),
    State(ID_GENES_PANEL, "value"),
    State(ID_GENES_RESTO, "value"),
    prevent_initial_call=True,
)
def enviar_perfil(_clics, edad, genero, raza, panel, resto):
    """Fija el perfil del formulario como el paciente a predecir."""
    return _paciente_de_formulario(edad, genero, raza, panel, resto)


@callback(
    Output(ID_PENDIENTE, "children"),
    Input(ID_EDAD, "value"),
    Input(ID_GENERO, "value"),
    Input(ID_RAZA, "value"),
    Input(ID_GENES_PANEL, "value"),
    Input(ID_GENES_RESTO, "value"),
    Input(ID_PREDICHO, "data"),
    prevent_initial_call=True,
)
def avisar_cambios(edad, genero, raza, panel, resto, predicho):
    """Avisa cuando el formulario ya no coincide con lo que se está mostrando."""
    actual = _paciente_de_formulario(edad, genero, raza, panel, resto)
    return "" if actual == predicho else "Hay cambios sin aplicar"


@callback(
    Output(ID_EDAD_VALOR, "children"),
    Input(ID_EDAD, "value"),
    prevent_initial_call=True,
)
def mostrar_edad(edad):
    """Lectura grande de la edad junto a su etiqueta."""
    return f"{int(edad if edad is not None else 50)} años"


@callback(
    Output(ID_GENES_CUENTA, "children"),
    Input(ID_GENES_PANEL, "value"),
    Input(ID_GENES_RESTO, "value"),
    prevent_initial_call=True,
)
def contar_genes(panel, resto):
    """Cuántos genes hay marcados como mutados."""
    n = len(set(panel or []) | set(resto or []))
    return f"{n} marcado" if n == 1 else f"{n} marcados"


@callback(
    Output(ID_VEREDICTO, "children"),
    Output(ID_REAL, "children"),
    Output(ID_PROBAS, "children"),
    Output(ID_EXPLICACION, "figure"),
    Output(ID_LECTURA, "children"),
    Output(ID_CONTEXTO, "data"),
    Input(ID_PREDICHO, "data"),
    Input(ID_MODELO, "value"),
    State(ID_CARGADO, "data"),
    # El layout ya trae la predicción del perfil inicial: sin esto se
    # recalculaba y redibujaba todo al abrir la pestaña (parpadeo).
    prevent_initial_call=True,
)
def predecir_perfil(paciente, modelo, cargado):
    """Predice, explica y sitúa el último perfil enviado con el botón."""
    modelo = modelo or "logistica"
    probas = predecir(paciente)
    explicacion = explicar(paciente, modelo)
    return (
        _veredicto(probas),
        _real(cargado, paciente, probas),
        _barras(probas),
        fig_explicacion(explicacion, paciente),
        lectura_guiada(_lectura(explicacion, paciente, modelo, probas[modelo]),
                       titulo="Lectura de esta predicción"),
        {"modelo": modelo, "p": probas[modelo]},
    )


@callback(
    Output(ID_ARBOL, "value"),
    Input(ID_PREDICHO, "data"),
    Input(ID_ARBOL_REP, "n_clicks"),
    prevent_initial_call=True,
)
def elegir_arbol(paciente, _clics):
    """Con cada predicción nueva, o al pedirlo, vuelve al árbol representativo."""
    return arbol_representativo(paciente) + 1


@callback(
    Output(ID_VOTACION, "figure"),
    Output(ID_ARBOL_FIG, "figure"),
    Output(ID_ARBOL_FIG, "style"),
    Output(ID_CAMINO, "children"),
    Input(ID_ARBOL, "value"),
    Input(ID_PREDICHO, "data"),
    prevent_initial_call=True,
)
def dibujar_bosque(numero, paciente):
    """Redibuja el histograma de árboles, el árbol elegido y su camino."""
    votacion, arbol, lista = _vistas_bosque(paciente, numero)
    return votacion, arbol, altura_grafico(arbol), lista


# Al cargar un perfil, los genes se encienden uno tras otro: se marca el
# formulario un momento y el CSS escalona la animación de cada ficha.
clientside_callback(
    """
    function (valor) {
        var f = document.querySelector(".formulario-prediccion");
        if (f) {
            f.classList.remove("cargando-perfil");
            void f.offsetWidth;
            f.classList.add("cargando-perfil");
            clearTimeout(window.__temporizadorPerfil);
            window.__temporizadorPerfil = setTimeout(function () {
                f.classList.remove("cargando-perfil");
            }, 1800);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output(ID_ANIMACION, "data"),
    Input(ID_PERFIL, "value"),
    prevent_initial_call=True,
)


@callback(
    Output(ID_SEC_LOG, "style"),
    Output(ID_SEC_RF, "style"),
    Output(ID_SEC_XGB, "style"),
    Input(ID_MODELO, "value"),
    prevent_initial_call=True,
)
def mostrar_como_decide(modelo):
    """Muestra solo la sección del modelo elegido en el selector de arriba."""
    return tuple({} if m == modelo else {"display": "none"}
                 for m in ("logistica", "random_forest", "xgboost"))


@callback(
    Output(ID_XGB_ARBOL, "value"),
    Input(ID_PREDICHO, "data"),
    Input(ID_XGB_REP, "n_clicks"),
    prevent_initial_call=True,
)
def elegir_arbol_xgb(paciente, _clics):
    """Con cada predicción nueva, o al pedirlo, vuelve al árbol que más pesa."""
    return arbol_mas_influyente(paciente) + 1


@callback(
    Output(ID_XGB_SUMA, "figure"),
    Output(ID_XGB_ARBOL_FIG, "figure"),
    Output(ID_XGB_ARBOL_FIG, "style"),
    Output(ID_XGB_CAMINO, "children"),
    Input(ID_XGB_ARBOL, "value"),
    Input(ID_PREDICHO, "data"),
    prevent_initial_call=True,
)
def dibujar_xgb(numero, paciente):
    suma, arbol, lista = _vistas_xgb(paciente, numero)
    return suma, arbol, altura_grafico(arbol), lista


@callback(
    Output(ID_LOG_CURVA, "figure"),
    Output(ID_LOG_RESUMEN, "children"),
    Input(ID_PREDICHO, "data"),
    prevent_initial_call=True,
)
def dibujar_logistica(paciente):
    z = logodds_logistica(paciente)
    return fig_curva_logistica(z), _resumen_logistica(z)


# El asistente virtual lee el perfil desde cualquier pestaña, así que se le
# copia a su propio Store (fuera de esta pestaña) cada vez que cambia el
# perfil enviado o el modelo. Sin prevent_initial_call: al abrir la pestaña
# ya le pasa el perfil inicial. `cargado` sirve para decir el grado real
# cuando el perfil es un paciente de prueba.
clientside_callback(
    """
    function (paciente, modelo, cargado) {
        return { paciente: paciente, modelo: modelo || "logistica", cargado: cargado };
    }
    """,
    Output("asistente-prediccion", "data"),
    Input(ID_PREDICHO, "data"),
    Input(ID_MODELO, "value"),
    State(ID_CARGADO, "data"),
)


# El gráfico de partículas lo dibuja assets/enjambre.js; este callback le pasa
# el modelo y la probabilidad del perfil, y pone la lectura escrita que
# devuelve (también al montar la pestaña, para que el gráfico se inicie).
clientside_callback(
    """
    function (estado) {
        var e = window.enjambrePrediccion;
        if (!e || !estado) return window.dash_clientside.no_update;
        return e.actualizar(estado);
    }
    """,
    Output(ID_CONTEXTO_LECTURA, "children"),
    Input(ID_CONTEXTO, "data"),
)
