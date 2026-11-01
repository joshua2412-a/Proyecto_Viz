"""
Pestaña 9 · Modelos
El capítulo 2 del Jupyter Book en versión navegable: cómo se entrenaron los
tres clasificadores, cómo rinden en los pacientes que nadie miró, qué variables
usan y si sus diferencias son reales.

No entrena nada: todas las cifras salen de modelos/resumen_modelado.json, que
escribe `python -m modelado.exportar` a partir de los resultados del libro. Así
el tablero y el libro no pueden contar números distintos.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from utils.components import (
    titular,
    banda_oscura,
    callout,
    card,
    data_table,
    graph_card,
    kpi_row,
    lectura_guiada,
    page_header,
    paragraph,
    section_title,
    stat_list,
)
from utils.config import URL_LIBRO
from utils.figures_modelo import (
    METRICAS,
    fig_anytime,
    fig_calibracion,
    fig_combinaciones,
    fig_importancia,
    fig_matriz_confusion,
    fig_panel_reducido,
    fig_pliegues,
    fig_roc,
    matriz_confusion,
)
from utils.formato import miles, num, p_tex, p_valor, pct, tnum
from utils.modelo import (
    panel_minimo,
    NOMBRE_BALANCEO,
    NOMBRE_MODELO,
    NOMBRE_OPTIMIZADOR,
    ORDEN_MODELOS,
    cargar_resumen,
)
from utils.theme import COLOR_GBM

URL_CAPITULO = f"{URL_LIBRO}02_Modelado.html"

ID_SUBTAB = "mod-subtab"
ID_CONTENIDO = "mod-contenido"
ID_METRICA = "mod-metrica"
ID_MAPA = "mod-mapa"
ID_MODELO = "mod-modelo"
ID_UMBRAL = "mod-umbral"
ID_MATRIZ = "mod-matriz"
ID_MATRIZ_STATS = "mod-matriz-stats"
ID_CALIBRACION = "mod-calibracion"
ID_ANYTIME_MODELO = "mod-anytime-modelo"
ID_ANYTIME = "mod-anytime"

TAB_DISENO = "mod-tab-diseno"
TAB_PRUEBA = "mod-tab-prueba"
TAB_INTERPRETACION = "mod-tab-interpretacion"
TAB_COMPARACION = "mod-tab-comparacion"


def _radio_modelo(id_, valor="logistica"):
    return dcc.RadioItems(
        id=id_,
        options=[{"label": f" {NOMBRE_MODELO[m]}", "value": m} for m in ORDEN_MODELOS],
        value=valor, inline=True, className="ambito-radio",
        inputClassName="me-1", labelClassName="me-4",
    )


# --------------------------------------------------------------------------- #
# Indicadores de cabecera
# --------------------------------------------------------------------------- #
def _kpis() -> dbc.Row:
    r = cargar_resumen()
    meta, modelos = r["meta"], r["modelos"]
    aucs = [modelos[m]["auc_test"] for m in ORDEN_MODELOS]
    mc = {m: matriz_confusion(m) for m in ORDEN_MODELOS}
    pm = panel_minimo(r)
    return kpi_row([
        {"valor": str(meta["combinaciones"]), "etiqueta": "Combinaciones evaluadas",
         "detalle": f"3 modelos × 4 balanceos × 4 optimizadores · "
                    f"{miles(meta['configuraciones'])} configuraciones probadas"},
        {"valor": f"{num(min(aucs), 2)}–{num(max(aucs), 2)}", "etiqueta": "AUC en prueba",
         "detalle": f"Los tres modelos, en {meta['n_test']} pacientes que no se usaron para entrenar"},
        {"valor": f"{min(v['vp'] for v in mc.values())}–{max(v['vp'] for v in mc.values())} de {meta['gbm_test']}",
         "etiqueta": "Glioblastomas detectados",
         "detalle": "Con el umbral de 0,5: recall de GBM entre el 89 % y el 91 %",
         "color": COLOR_GBM},
        {"valor": str(pm["k"]), "etiqueta": "Variables que bastan",
         "detalle": f"AUC {num(pm['auc_k'], 3)} frente a {num(pm['auc_todas'], 3)} con las 23 "
                    f"(regla de un error estándar). Con IDH1, edad, TP53 y PTEN, "
                    f"{num(pm['auc_4'], 3)}"},
    ])


# --------------------------------------------------------------------------- #
# Subpestaña 1 · Diseño y resultados de la validación anidada
# --------------------------------------------------------------------------- #
def _etapa(numero: str, titulo: str, items: list[str], nota: str) -> dbc.Col:
    return dbc.Col(
        card(
            [
                html.Ul([html.Li(i) for i in items], className="etapa-lista"),
                html.Div(nota, className="etapa-nota"),
            ],
            titulo=f"{numero} · {titulo}",
            className="etapa-card h-100",
        ),
        lg=4, className="mb-3",
    )


def _panel_diseno() -> html.Div:
    meta = cargar_resumen()["meta"]
    return html.Div(
        [
            section_title("El experimento: un producto, no una suma"),
            paragraph(
                "No se entrenó un modelo sino todas las combinaciones de tres "
                "decisiones. Cada una de las 48 se optimizó y se evaluó por separado, "
                "con el preprocesamiento y el balanceo siempre dentro del pipeline: "
                "ningún paciente de validación influye en el escalado de la edad ni "
                "sirve de base para crear pacientes sintéticos."
            ),
            dbc.Row(
                [
                    _etapa("1", "Modelo", ["Regresión logística (L1/L2)", "Random Forest", "XGBoost"],
                           "Un modelo lineal y dos de árboles con supuestos opuestos."),
                    _etapa("2", "Balanceo", ["Sin balanceo", "SMOTE-NC", "ADASYN", "class_weight='balanced'"],
                           "Desbalance leve: 58 % LGG y 42 % GBM."),
                    _etapa("3", "Optimizador", ["Grid Search", "Random Search", "Bayesiano (Optuna, TPE)",
                                                "Genético (DEAP)"],
                           f"{meta['presupuesto']} evaluaciones cada uno: comparación a presupuesto fijo."),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            stat_list([
                                ("Bucle externo (solo mide)", f"{meta['k_externo']} pliegues estratificados"),
                                ("Bucle interno (busca)", f"{meta['k_interno']} pliegues dentro de cada uno"),
                                ("Configuraciones probadas", miles(meta["configuraciones"])),
                                ("Ajustes internos", miles(meta["ajustes_internos"])),
                                ("Métrica que guía la búsqueda", "AUC"),
                                ("Semilla global", str(meta["semilla"])),
                                ("Cómputo total", f"{num(meta['minutos'], 0)} min con 2 procesos"),
                            ]),
                            titulo="Validación cruzada anidada",
                            subtitulo="Elegir hiperparámetros y medir con los mismos datos "
                                      "sobreestima el desempeño; por eso se separan.",
                            tono="acento",
                        ),
                        lg=5, className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada(
                            [
                                "La AUC se eligió como criterio porque no depende del umbral "
                                "ni se distorsiona con el desbalance: mide si el modelo ordena "
                                "bien a los pacientes por riesgo de GBM.",
                                "El recall de GBM es la métrica clínica: un glioblastoma "
                                "clasificado como LGG retrasa un tratamiento urgente, mientras "
                                "que el error contrario lleva a pruebas adicionales.",
                                "Los 168 pacientes de prueba que reservó el EDA solo se usan "
                                "al final, una vez elegidos los tres modelos.",
                            ],
                            titulo="Por qué así",
                        ),
                        lg=7, className="mb-4",
                    ),
                ],
                className="mt-2",
            ),
            section_title("Las 48 combinaciones"),
            card(
                [
                    html.Div("Métrica del mapa", className="control-label"),
                    dcc.RadioItems(
                        id=ID_METRICA,
                        options=[{"label": f" {v[0]}", "value": k} for k, v in METRICAS.items()],
                        value="auc", inline=True, className="ambito-radio",
                        inputClassName="me-1", labelClassName="me-4",
                    ),
                ],
                className="controles-card mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(fig_combinaciones("auc"), "Media del bucle externo (10 pliegues)",
                                   "Pasa el cursor por una celda para ver la desviación entre "
                                   "pliegues, el recall y el tiempo de cómputo.", graph_id=ID_MAPA),
                        lg=8, className="mb-4",
                    ),
                    dbc.Col(
                        lectura_guiada([
                            "Las 48 combinaciones caben en una franja de 0,009 de AUC "
                            "(de 0,911 a 0,921), mientras que cada una varía unos 0,04 de un "
                            "pliegue a otro.",
                            "La mejor es Random Forest con ADASYN y optimización bayesiana "
                            "(0,921); la regresión logística sin balanceo queda a 0,001 con "
                            "15 segundos de cómputo.",
                            "El balanceo casi no mueve la AUC; cambia el recall: más GBM "
                            "detectados a cambio de algo de precisión.",
                            "El algoritmo genético ocupa la última columna en las cuatro filas de "
                            "Random Forest, porque converge demasiado pronto; en XGBoost el peor "
                            "método cambia de una fila a otra.",
                        ]),
                        lg=4, className="mb-4",
                    ),
                ]
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Subpestaña 2 · Evaluación en el conjunto de prueba
# --------------------------------------------------------------------------- #
def _tabla_prueba() -> dbc.Table:
    r = cargar_resumen()["modelos"]
    filas = []
    for m in ORDEN_MODELOS:
        mc = matriz_confusion(m)
        d = r[m]
        filas.append([
            NOMBRE_MODELO[m],
            f"{NOMBRE_BALANCEO[d['balanceo']]} · {NOMBRE_OPTIMIZADOR[d['metodo']]}",
            f"{num(d['auc_test'], 3)} [{num(d['auc_test_ic'][0], 2)}; {num(d['auc_test_ic'][1], 2)}]",
            pct(mc["recall"] * 100, 1), pct(mc["especificidad"] * 100, 1),
            pct(mc["precision"] * 100, 1), num(d["brier_test"], 3),
        ])
    return data_table(
        ["Modelo", "Balanceo · optimizador", "AUC [IC 95 %]", "Recall GBM",
         "Especificidad", "Precisión", "Brier"], filas)


def _stats_matriz(modelo: str, umbral: float) -> html.Div:
    mc = matriz_confusion(modelo, umbral)
    return stat_list([
        ("GBM detectados (recall)", f"{mc['vp']} de {mc['vp'] + mc['fn']} · {pct(mc['recall'] * 100, 1)}"),
        ("LGG bien clasificados (especificidad)",
         f"{mc['vn']} de {mc['vn'] + mc['fp']} · {pct(mc['especificidad'] * 100, 1)}"),
        ("Precisión de la alerta GBM", pct(mc["precision"] * 100, 1)),
        ("Exactitud global", pct(mc["accuracy"] * 100, 1)),
    ])


def _panel_prueba() -> html.Div:
    modelos = cargar_resumen()["modelos"]
    return html.Div(
        [
            section_title("Los tres modelos finales en 168 pacientes nuevos"),
            paragraph(
                "Para cada familia se eligió la combinación con mejor AUC en el bucle "
                "externo, se volvió a optimizar con los 671 pacientes de entrenamiento "
                "y se aplicó una sola vez al conjunto de prueba."
            ),
            card(_tabla_prueba(), className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_roc(), "Curvas ROC en prueba",
                                       "Cada punto de la curva es un umbral distinto."),
                            lg=6, className="mb-4"),
                    dbc.Col(
                        lectura_guiada([
                            "Las tres curvas se superponen: AUC de 0,923, 0,926 y 0,930, con "
                            "intervalos de confianza de unos ±0,04 que se solapan casi por "
                            "completo.",
                            "La AUC en prueba coincide con la estimación de la validación "
                            "anidada (0,918–0,921): el procedimiento no sobreestimó el "
                            "desempeño.",
                            # "Más del doble" no valía para XGBoost (15 frente a 8)
                            "Con el umbral de 0,5 hay unas dos veces más falsos positivos que "
                            "falsos negativos (15-18 frente a 6-8), que es el desequilibrio "
                            "deseable aquí.",
                            "Llevar el recall al 95 % exige aceptar entre un 21 % y un 33 % de "
                            "falsos positivos: pruébalo con el deslizador de abajo.",
                        ]),
                        lg=6, className="mb-4",
                    ),
                ]
            ),
            section_title("Elige el umbral de decisión"),
            card(
                dbc.Row(
                    [
                        dbc.Col([html.Div("Modelo", className="control-label"),
                                 _radio_modelo(ID_MODELO)], lg=6, className="mb-3 mb-lg-0"),
                        dbc.Col(
                            [
                                html.Div("Umbral de probabilidad para decir GBM",
                                         className="control-label"),
                                dcc.Slider(id=ID_UMBRAL, min=0.05, max=0.95, step=0.05, value=0.5,
                                           marks={v / 100: f"{v} %" for v in range(10, 100, 20)},
                                           tooltip={"placement": "bottom",
                                                    "always_visible": False}),
                            ],
                            lg=6,
                        ),
                    ],
                    className="align-items-end g-3",
                ),
                className="controles-card mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_matriz_confusion("logistica"), "Matriz de confusión en prueba",
                                       "Cada fila suma su propio 100 %.", graph_id=ID_MATRIZ),
                            lg=6, className="mb-4"),
                    dbc.Col(card(html.Div(_stats_matriz("logistica", 0.5), id=ID_MATRIZ_STATS),
                                 titulo="Con este umbral", tono="acento",
                                 subtitulo="Bajar el umbral detecta más GBM y genera más falsas "
                                           "alarmas; subirlo hace lo contrario."),
                            lg=6, className="mb-4"),
                ]
            ),
            section_title("¿Se puede creer la probabilidad?"),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_calibracion("logistica"), "Diagrama de confiabilidad",
                                       "El modelo es el elegido arriba. El tamaño del punto es el "
                                       "número de pacientes del tramo.", graph_id=ID_CALIBRACION),
                            lg=7, className="mb-4"),
                    dbc.Col(
                        card(
                            [
                                stat_list([
                                    (NOMBRE_MODELO[m],
                                     f"ECE {num(modelos[m]['ece_oof'], 3)} · Brier {num(modelos[m]['brier_test'], 3)}")
                                    for m in ORDEN_MODELOS
                                ]),
                                callout(
                                    "Una AUC alta solo garantiza que el modelo ordena bien. La "
                                    "regresión logística y XGBoost están razonablemente "
                                    "calibrados; Random Forest, entrenado con ADASYN, sobrestima "
                                    "el riesgo de GBM entre el 30 % y el 50 %. La regresión "
                                    "isotónica reduce su error de calibración de 0,069 a 0,031 "
                                    "sin cambiar la AUC.",
                                    titulo="Calibración",
                                ),
                            ],
                            titulo="Error de calibración esperado (ECE)",
                            subtitulo="Fuera de pliegue, 671 pacientes. Más bajo es mejor.",
                        ),
                        lg=5, className="mb-4",
                    ),
                ]
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Subpestaña 3 · Interpretación
# --------------------------------------------------------------------------- #
def _panel_interpretacion() -> html.Div:
    pm = panel_minimo()
    return html.Div(
        [
            section_title("Qué variables usan los modelos"),
            paragraph(
                "La importancia se mide con valores SHAP: cuánto aporta cada variable, "
                "en promedio y en valor absoluto, a las predicciones de los 839 pacientes. "
                "Se compara un modelo lineal con uno de árboles porque, si dos modelos "
                "construidos con supuestos opuestos coinciden, el hallazgo no es un "
                "artefacto de uno de ellos."
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_importancia(), "Importancia global (SHAP)",
                                       "Porcentaje del total de |SHAP| medio. Las 10 variables "
                                       "más importantes según XGBoost."),
                            lg=7, className="mb-4"),
                    dbc.Col(
                        lectura_guiada([
                            "IDH1 explica por sí solo entre el 42 % y el 49 % de la importancia; "
                            "con la edad, TP53 y PTEN se llega al 74 % en XGBoost y al 83 % en la logística.",
                            "TP53 sube al tercer puesto aunque su asociación individual con el "
                            "grado era moderada: su efecto aparece una vez conocido IDH1.",
                            "ATRX casi desaparece: su información ya está en TP53 e IDH1, con "
                            "los que coocurre. La regularización L1 de la logística la anula, "
                            "junto con otras 9 variables.",
                            "EGFR cambia de signo: asociada a GBM por sí sola, empuja levemente "
                            "hacia LGG una vez descontadas la edad y la ausencia de IDH1.",
                            "NF1 no mostraba asociación con el grado por sí sola, pero es la "
                            "quinta variable en XGBoost y entra en el panel de ocho: su "
                            "efecto aparece en combinación con las demás.",
                        ]),
                        lg=5, className="mb-4",
                    ),
                ]
            ),
            section_title("¿Cuántas variables hacen falta?"),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_panel_reducido(), "AUC según el tamaño del panel",
                                       "El ranking de variables se recalcula dentro de cada pliegue "
                                       "externo, sin ver sus datos de validación."),
                            lg=7, className="mb-4"),
                    dbc.Col(
                        card(
                            [
                                paragraph(
                                    "Es la respuesta al objetivo general del proyecto. Con la "
                                    "edad y el estado de tres genes el modelo conserva casi "
                                    "toda la capacidad del panel de 20 genes; según la regla "
                                    "de un error estándar, el panel más pequeño que no se "
                                    f"distingue del completo tiene {pm['k']} variables."
                                ),
                                stat_list([
                                    ("Solo IDH1", f"AUC {num(pm['auc'][1], 3)}"),
                                    ("IDH1 + edad", f"AUC {num(pm['auc'][2], 3)}"),
                                    ("IDH1 + edad + TP53 + PTEN", f"AUC {num(pm['auc_4'], 3)}"),
                                    (f"Las {pm['k']} primeras", f"AUC {num(pm['auc_k'], 3)}"),
                                    ("Las 23 variables", f"AUC {num(pm['auc_todas'], 3)}"),
                                ]),
                                # Las 8 no son una lista fija (el ranking se rehace en
                                # cada pliegue): son las que entran en el top 8 de
                                # forma más estable (libro, sección 2.12)
                                paragraph(
                                    "Las ocho del panel mínimo: IDH1, edad, TP53, PTEN y "
                                    "NF1, que entran entre las ocho primeras en los 10 "
                                    "pliegues, y CIC, ATRX e IDH2, en 9, 8 y 7."
                                ),
                                callout(
                                    "IDH1 encabeza el ranking en los 10 pliegues, y la edad, "
                                    "TP53 y PTEN ocupan de media los puestos 2 a 4. Además, la "
                                    "variante más común "
                                    "de IDH1, R132H (cerca del 90 % de los gliomas con IDH "
                                    "mutado), puede detectarse por inmunohistoquímica, sin "
                                    "secuenciar.",
                                    titulo="Un panel estable",
                                ),
                            ],
                            titulo="Panel reducido", tono="acento",
                        ),
                        lg=5, className="mb-4",
                    ),
                ]
            ),
            callout(
                "En la pestaña Predicción puedes ver estas contribuciones paciente a "
                "paciente: cambia una mutación y mira cómo se mueve la probabilidad.",
                titulo="De la importancia global a la explicación local",
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Subpestaña 4 · Comparación estadística y optimizadores
# --------------------------------------------------------------------------- #
def _panel_comparacion() -> html.Div:
    r = cargar_resumen()
    fr, fac, dl = r["friedman"], r["factores"], r["delong"]
    nombres_factor = {"modelo": "Modelo", "balanceo": "Balanceo", "metodo": "Optimizador"}
    # Solo "sin balanceo" lleva mayúscula: capitalize() dejaba "Smote-nc"
    nombre_nivel = {**NOMBRE_MODELO, **NOMBRE_BALANCEO, "ninguno": "Sin balanceo",
                    **NOMBRE_OPTIMIZADOR}
    filas_factor = [
        [nombres_factor[f], str(d["bloques"]), f"$\\chi^2 = {tnum(d['chi2'], 1)}$", p_valor(d["p"]),
         ", ".join(f"{nombre_nivel[k]} {num(v, 2)}" for k, v in d["rangos"].items())]
        for f, d in fac.items()
    ]
    filas_delong = [
        [f"{NOMBRE_MODELO[d['a']]} vs {NOMBRE_MODELO[d['b']]}", num(d["diferencia"], 3, signo=True),
         f"[{num(d['ic'][0], 3)}; {num(d['ic'][1], 3)}]", p_valor(d["p"]), p_valor(d["p_holm"]),
         num(d["cliff"], 2, signo=True)]
        for d in dl
    ]
    opt = r["optimizadores"]
    filas_opt = [
        [NOMBRE_OPTIMIZADOR[o], num(opt[o]["auc_t6"], 4), num(opt[o]["auc_t24"], 4),
         pct(opt[o]["al_mejor"] * 100, 0), num(opt[o]["evals_al_mejor"], 0),
         num(opt[o]["auc_externa"], 4), num(opt[o]["optimismo"], 4, signo=True)]
        for o in ["grid", "random", "bayesiano", "genetico"]
    ]
    return html.Div(
        [
            section_title("¿Son distintos de verdad?"),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_pliegues(), "AUC de cada pliegue externo",
                                       "Modelos finales. La raya es la media de los 10 pliegues."),
                            lg=7, className="mb-4"),
                    dbc.Col(
                        card(
                            stat_list([
                                ("DE de un procedimiento entre pliegues", num(fr["variacion_dentro"], 3)),
                                ("DE entre los 48 procedimientos", num(fr["variacion_entre"], 3)),
                                ("Rango de las medias (mejor − peor)", num(fr["rango_medias"], 3)),
                                ("Friedman, 48 combinaciones",
                                 f"$\\chi^2({fr['gl']}) = {tnum(fr['chi2'], 1)}$, {p_tex(fr['p'])}"),
                            ]),
                            titulo="Variación entre pacientes frente a variación entre métodos",
                            subtitulo="La primera es diecisiete veces mayor que la segunda.",
                            tono="acento",
                        ),
                        lg=5, className="mb-4",
                    ),
                ]
            ),
            callout(
                "La prueba de Friedman no encuentra diferencias entre las 48 combinaciones "
                f"({p_valor(fr['p'])}), DeLong no distingue entre los tres modelos finales en "
                "prueba y todos los tamaños de efecto (delta de Cliff) son despreciables. Con "
                "estos datos los tres modelos son equivalentes en discriminación; la elección "
                "entre ellos depende de la interpretabilidad, la calibración y el costo.",
                titulo="Conclusión estadística",
            ),
            section_title("Comparación dirigida entre los modelos finales"),
            card(data_table(["Comparación", "Δ AUC (prueba)", "IC 95 % bootstrap", "p DeLong",
                             "p ajustado (Holm)", "Tamaño del efecto (Cliff)"], filas_delong),
                 subtitulo="Con 3 comparaciones, los p-valores se ajustan por Holm-Bonferroni.",
                 className="mb-4"),
            section_title("Análisis por factor"),
            card(data_table(["Factor", "Bloques", "Friedman", "p", "Rango promedio (1 = mejor)"],
                            filas_factor),
                 subtitulo="Cada combinación de los otros dos factores con cada pliegue es un "
                           "bloque. El modelo y el balanceo salen significativos, con efectos "
                           "pequeños: sus medias difieren menos de 0,003 de AUC. Sin balanceo "
                           "obtiene el mejor rango; el optimizador no influye.",
                 className="mb-4"),
            section_title("Los cuatro optimizadores"),
            card(
                [html.Div("Modelo", className="control-label"),
                 _radio_modelo(ID_ANYTIME_MODELO, "xgboost")],
                className="controles-card mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card(fig_anytime("xgboost"), "Curvas anytime",
                                       "Cuánto le falta a cada método, tras t evaluaciones, para la "
                                       "mejor AUC interna que encontró cualquiera de los cuatro. "
                                       "Más abajo es mejor; media de 40 búsquedas.",
                                       graph_id=ID_ANYTIME),
                            lg=7, className="mb-4"),
                    dbc.Col(
                        lectura_guiada([
                            "El bayesiano (TPE) es el más eficiente: tras 6 evaluaciones ya "
                            "supera lo que Random Search logra tras 12.",
                            "Grid Search depende del orden de la rejilla: empieza por la "
                            "regularización más fuerte y tarda en llegar a la zona buena.",
                            "Ninguno sobreajusta su búsqueda de forma apreciable: lo que la "
                            "AUC interna supera a la externa no pasa de 0,002.",
                        ]),
                        lg=5, className="mb-4",
                    ),
                ]
            ),
            card(data_table(["Optimizador", "AUC interna t = 6", "t = 24", "Bloques al mejor",
                             "Evaluaciones al mejor", "AUC externa", "Optimismo"], filas_opt),
                 subtitulo="Promedios de 120 búsquedas (3 modelos × 4 balanceos × 10 pliegues). "
                           "Optimismo = AUC interna − AUC externa.",
                 className="mb-4"),
        ]
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def layout() -> html.Div:
    """Layout de la pestaña de modelos."""
    sub = [
        ("Diseño y resultados", TAB_DISENO),
        ("Evaluación en prueba", TAB_PRUEBA),
        ("Interpretación", TAB_INTERPRETACION),
        ("Comparación estadística", TAB_COMPARACION),
    ]
    return html.Div(
        [
            titular(
                "03 Los resultados · Modelos",
                "Tres clasificadores",
                "puestos a prueba",
                texto="Regresión logística, Random Forest y XGBoost, comparados con "
                      "validación cruzada anidada, cuatro técnicas de balanceo y cuatro "
                      "métodos de optimización, y evaluados en los pacientes que el EDA "
                      "dejó sin mirar.",
            ),
            _kpis(),
            dbc.Tabs(
                [dbc.Tab(label=etiqueta, tab_id=tab_id, label_class_name="sub-tab",
                         active_label_class_name="sub-tab-active") for etiqueta, tab_id in sub],
                id=ID_SUBTAB, active_tab=TAB_DISENO, className="sub-tabs mt-2",
            ),
            dcc.Loading(html.Div(_panel_diseno(), id=ID_CONTENIDO), type="dot",
                        target_components={ID_CONTENIDO: "children"}, delay_show=200),
            html.Div(
                callout(
                    [
                        "El desarrollo completo, con los espacios de búsqueda, la complejidad "
                        "computacional, LIME, la sensibilidad a la semilla y todas las pruebas, "
                        "está en el ",
                        html.A("capítulo 2 del Jupyter Book", href=URL_CAPITULO, target="_blank",
                               rel="noopener noreferrer", className="link-externo"),
                        ".",
                    ],
                    titulo="Dónde seguir",
                ),
                className="mt-4",
            ),
        ],
        className="vista-pestana",
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
# prevent_initial_call en todos: el layout ya trae el contenido y las figuras
# iniciales. Sin él, al abrir la pestaña cada callback volvía a calcular lo
# mismo, dcc.Loading ocultaba el panel mientras tanto y la pestaña parpadeaba.
@callback(Output(ID_CONTENIDO, "children"), Input(ID_SUBTAB, "active_tab"),
          prevent_initial_call=True)
def cambiar_subpestana(subtab: str):
    """Monta la subpestaña elegida."""
    return {
        TAB_PRUEBA: _panel_prueba,
        TAB_INTERPRETACION: _panel_interpretacion,
        TAB_COMPARACION: _panel_comparacion,
    }.get(subtab, _panel_diseno)()


@callback(Output(ID_MAPA, "figure"), Input(ID_METRICA, "value"), prevent_initial_call=True)
def cambiar_metrica(metrica: str):
    return fig_combinaciones(metrica or "auc")


@callback(
    Output(ID_MATRIZ, "figure"),
    Output(ID_MATRIZ_STATS, "children"),
    Output(ID_CALIBRACION, "figure"),
    Input(ID_MODELO, "value"),
    Input(ID_UMBRAL, "value"),
    prevent_initial_call=True,
)
def cambiar_umbral(modelo: str, umbral: float):
    modelo = modelo or "logistica"
    umbral = umbral if umbral is not None else 0.5
    return (fig_matriz_confusion(modelo, umbral), _stats_matriz(modelo, umbral),
            fig_calibracion(modelo))


@callback(Output(ID_ANYTIME, "figure"), Input(ID_ANYTIME_MODELO, "value"),
          prevent_initial_call=True)
def cambiar_anytime(modelo: str):
    return fig_anytime(modelo or "xgboost")
