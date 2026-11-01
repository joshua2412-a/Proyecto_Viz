"""
Pestaña 12 · Conclusiones
Qué quedó establecido con el análisis exploratorio y con los modelos, qué
significa para el objetivo de reducir el panel de secuenciación y qué falta.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.formato import num, p_tex, pct, tnum
from utils.config import nombre_corto
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
    vif_predictoras,
    prevalencia_genes,
    tamanos_particion,
)
from utils.modelo import ORDEN_MODELOS, cargar_resumen, panel_minimo
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

SIGUIENTES_PASOS = [
    (
        "Validar con una cohorte externa",
        "Todo lo observado proviene de TCGA. Evaluar los tres modelos, sin "
        "reentrenarlos, sobre una cohorte independiente diría si su rendimiento "
        "se sostiene fuera de esta muestra o es una particularidad suya.",
    ),
    (
        "Llevar el panel reducido a un estudio prospectivo",
        "El panel reducido se midió sobre pacientes a los que se secuenció todo. "
        "El paso que lo convierte en un ahorro real es un estudio en el que solo "
        "se secuencien sus genes (IDH1, TP53 y PTEN, o los siete del panel de "
        "ocho variables: IDH1, TP53, PTEN, NF1, CIC, ATRX e IDH2) y se compare "
        "la decisión con la histología.",
    ),
    (
        "Elegir el umbral según el coste clínico",
        "Los modelos clasifican con el umbral de 0,5. Dejar pasar un glioblastoma "
        "no cuesta lo mismo que sobretratar un glioma de bajo grado: fijar el "
        "umbral con ese coste, y recalibrar las probabilidades antes de usarlas, "
        "es una decisión que corresponde al equipo clínico.",
    ),
]


def _kpis() -> list[dict]:
    """Las cuatro cifras que resumen lo que encontró el análisis."""
    asociacion = asociacion_con_grado("train")
    edad = estadisticas_edad("train").set_index("grade_label")
    tamanos = tamanos_particion()

    resumen = cargar_resumen()
    meta = resumen["meta"]
    aucs = [resumen["modelos"][m]["auc_test"] for m in ORDEN_MODELOS]

    idh1 = float(asociacion.set_index("variable").loc["IDH1", "rho"])
    significativas = int(asociacion["significativa"].sum())

    return [
        {
            "valor": f"{num(idh1, 2, signo=True)}",
            "etiqueta": "Spearman de IDH1",
            "detalle": "El marcador más discriminante, asociado al bajo grado",
            "color": COLOR_LGG,  # signo negativo: empuja hacia LGG
        },
        {
            "valor": f"{num(edad.loc['GBM', 'media'] - edad.loc['LGG', 'media'], 1)} años",
            "etiqueta": "Brecha de edad",
            "detalle": "Entre el diagnóstico de GBM y el de LGG",
            "color": f"linear-gradient(180deg, {COLOR_LGG} 0 50%, {COLOR_GBM} 50% 100%)",
        },
        {
            "valor": f"{significativas} de {len(asociacion)}",
            "etiqueta": "Variables con señal",
            "detalle": f"Sobre {tamanos['train']} pacientes de entrenamiento",
        },
        {
            "valor": f"{num(min(aucs), 2)}–{num(max(aucs), 2)}",
            "etiqueta": "AUC de los modelos",
            "detalle": f"Los tres clasificadores, en {meta['n_test']} pacientes de prueba",
        },
    ]


def _tabla_hallazgos() -> dbc.Table:
    """Hallazgos principales con su evidencia cuantitativa, calculada en vivo."""
    asociacion = asociacion_con_grado("train").set_index("variable")
    prevalencia = prevalencia_genes("train").set_index("gen")
    edad = estadisticas_edad("train").set_index("grade_label")

    idh1 = prevalencia.loc["IDH1"]
    no_significativas = asociacion[~asociacion["significativa"]]
    vif = vif_predictoras("train")
    resumen = cargar_resumen()
    aucs = [resumen["modelos"][m]["auc_test"] for m in ORDEN_MODELOS]
    panel = resumen["panel"]
    auc_4, auc_23 = panel["auc"][panel["k"].index(4)], panel["auc"][-1]
    pm = panel_minimo(resumen)

    return data_table(
        ["Hallazgo", "Evidencia", "Consecuencia"],
        [
            [
                "IDH1 es el marcador dominante",
                f"$r = {tnum(asociacion.loc['IDH1', 'rho'], 2, signo=True)}$ · mutado en "
                f"{pct(idh1['LGG'], 1)} de LGG frente a {pct(idh1['GBM'], 1)} de GBM",
                "Su presencia es casi por sí sola una firma de glioma de bajo grado.",
            ],
            [
                "La edad aporta señal clínica independiente",
                f"$r = {tnum(asociacion.loc['Age_at_diagnosis', 'rho'], 2, signo=True)}$ · "
                f"{num(edad.loc['GBM', 'media'], 1)} años de media en GBM frente a "
                f"{num(edad.loc['LGG', 'media'], 1)} en LGG",
                "Una variable que ya está en la historia clínica y no cuesta nada "
                "medir discrimina casi tanto como una mutación.",
            ],
            [
                "La mayoría del panel no discrimina el grado",
                f"{len(no_significativas)} de {len(asociacion)} variables sin "
                r"asociación apreciable ($p \geq 0{,}05$ o $|r| < 0{,}10$)",
                "Sostiene la hipótesis central: un panel reducido podría bastar para "
                "decidir el grado.",
            ],
            [
                "No hay multicolinealidad estructural",
                f"Ningún par con V de Cramér > 0,70, y el factor de inflación de la "
                f"varianza más alto es {num(vif.iloc[0]['vif'], 2)} "
                f"({nombre_corto(vif.iloc[0]['variable'])}), lejos del umbral de 5",
                "El VIF es lo que lo sostiene: mira cada predictora contra todas "
                "las demás a la vez, no solo por parejas. Los marcadores pueden "
                "entrar juntos en un modelo sin inestabilizar sus coeficientes.",
            ],
            [
                "El grupo racial no es interpretable aquí",
                "Más del 90 % de la cohorte pertenece a una sola categoría",
                "La asociación que aparece es un artefacto de composición de la "
                "muestra, no un hallazgo biológico.",
            ],
            [
                "Los tres modelos rinden igual",
                f"AUC de prueba entre {num(min(aucs), 3)} y {num(max(aucs), 3)}; "
                f"la prueba de DeLong no separa ningún par "
                f"({p_tex(min(d['p'] for d in resumen['delong']))} en el más "
                "cercano, antes de corregir)",
                "La elección entre regresión logística, Random Forest y XGBoost "
                "puede hacerse por interpretabilidad: la logística no pierde nada.",
            ],
            [
                "Un panel reducido basta",
                f"IDH1, edad, TP53 y PTEN: AUC {num(auc_4, 3)} en validación "
                f"cruzada, frente a {num(auc_23, 3)} con las 23. La regla de un error "
                f"estándar pide {pm['k']} variables: IDH1, edad, TP53, PTEN, NF1, "
                f"CIC, ATRX e IDH2 (AUC {num(pm['auc_k'], 3)})",
                "Confirma la hipótesis del análisis exploratorio: la mayor parte "
                "del panel no añade capacidad para decidir el grado.",
            ],
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de conclusiones."""
    asociacion = asociacion_con_grado("train")
    significativas = int(asociacion["significativa"].sum())
    resumen = cargar_resumen()
    aucs = [resumen["modelos"][m]["auc_test"] for m in ORDEN_MODELOS]
    panel = resumen["panel"]
    auc_4, auc_23 = panel["auc"][panel["k"].index(4)], panel["auc"][-1]
    pm = panel_minimo(resumen)

    return html.Div(
        [
            page_header(
                "Conclusiones",
                "Un puñado de marcadores concentra casi toda la información sobre el "
                "grado del tumor: el análisis dice cuáles son y los modelos "
                "confirman que con ellos basta.",
                "check_circle",
            ),
            kpi_row(_kpis()),
            card(
                [
                    paragraph(
                        f"De las {len(asociacion)} variables disponibles, solo "
                        f"{significativas} muestran una asociación con el grado que sea "
                        "a la vez estadísticamente significativa y de magnitud "
                        "apreciable. El resto, por sí solo, aporta una varianza "
                        "explicativa insignificante: no distingue a un glioma de bajo "
                        "grado de un glioblastoma."
                    ),
                    paragraph(
                        "La fase de modelado convierte esa intuición en una cifra. Los "
                        "tres clasificadores separan los dos grados con un AUC de "
                        f"{num(min(aucs), 2)}–{num(max(aucs), 2)} en pacientes que no "
                        "vieron al entrenar, sin diferencias significativas entre "
                        "ellos. Un panel de solo cuatro variables (IDH1, edad, TP53 "
                        f"y PTEN) alcanza un AUC de {num(auc_4, 3)} frente a "
                        f"{num(auc_23, 3)} con las 23, y con {pm['k']} la diferencia ya "
                        "no se distingue de la variación entre pliegues. Para decidir "
                        "el grado no hace falta, en estos datos, secuenciar el panel "
                        "completo."
                    ),
                    callout(
                        "La señal que encuentran los contrastes coincide con la "
                        "literatura clínica: IDH1, IDH2, ATRX, CIC y TP53 hacia LGG, y "
                        "la edad, PTEN y EGFR hacia GBM. Que un análisis hecho a ciegas "
                        "sobre los datos reproduzca el conocimiento médico establecido "
                        "es una señal de validez, aunque no sustituye a una cohorte "
                        "externa.",
                        titulo="Coherencia con el conocimiento clínico",
                    ),
                ],
                titulo="Resultado principal",
            ),
            section_title("Hallazgos y su evidencia"),
            card(
                _tabla_hallazgos(),
                titulo="Siete hallazgos con respaldo cuantitativo",
                subtitulo="Contrastes sobre el conjunto de entrenamiento; AUC de "
                          "prueba sobre los pacientes reservados",
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
                                    "el análisis exploratorio y el modelado completos, "
                                    "con su desarrollo estadístico. ",
                                    enlace_externo("Abrirlo", URL_LIBRO),
                                ]
                            ),
                            html.Span(
                                [
                                    html.B("Dashboard: "),
                                    "esta capa interactiva, con el EDA navegable, la "
                                    "comparación de los tres modelos y un simulador de "
                                    "predicción.",
                                ]
                            ),
                            html.Span(
                                [
                                    html.B("Código: "),
                                    "pipeline reproducible con una semilla fija, del CSV "
                                    "a los modelos. ",
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
