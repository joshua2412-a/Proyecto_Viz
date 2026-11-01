"""
Pestaña 6 · Metodología
Cómo se pasó del CSV a los modelos: partición, control de calidad, las
pruebas estadísticas aplicadas a cada tipo de variable y el diseño del
modelado. Todo reproducible con una semilla.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from utils.formato import pct
from utils.components import (
    badge_numero,
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
from utils.config import (
    CATEGORICAL_FEATURES,
    GENE_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    URL_LIBRO_EDA,
)
from utils.data_loader import (
    asociacion_con_grado,
    proporcion_grado,
    tamanos_particion,
)
from utils.theme import COLOR_GBM, COLOR_LGG, SERIES

ETAPAS = [
    (
        "1",
        "Verificación del dataset",
        "839 registros y 24 variables, sin valores nulos ni faltantes. El único "
        "registro duplicado corresponde a dos pacientes distintos que comparten "
        "fenotipo y perfil genético, así que se conserva. Las variables binarias se "
        "convierten a tipo categórico para el análisis exploratorio.",
    ),
    (
        "2",
        "Partición estratificada 80/20",
        "Se reserva el 20 % de los pacientes antes de mirar cualquier estadístico, "
        "de modo que el EDA no filtre información del conjunto de prueba. La "
        "estratificación mantiene la proporción LGG/GBM en ambos lados. Ninguna "
        "prueba ni ninguna comparación contra el grado se calcula sobre esa "
        "reserva; en Exploración se puede comparar la distribución de cada "
        "variable entre particiones, que es descripción y no compromete nada.",
    ),
    (
        "3",
        "Análisis univariado",
        "Distribución de cada variable por separado: medidas de tendencia central y "
        "dispersión para la edad, frecuencias para las categóricas y las 20 "
        "mutaciones. Se evalúan normalidad (Shapiro-Wilk) y valores atípicos "
        "(regla del rango intercuartílico).",
    ),
    (
        "4",
        "Análisis bivariado contra el grado",
        "Cada predictora se cruza con la variable objetivo usando la prueba que "
        "corresponde a su naturaleza, y se cuantifica la fuerza de la asociación "
        "además de su significancia: un p-valor pequeño con una muestra grande no "
        "implica un efecto relevante.",
    ),
    (
        "5",
        "Diagnóstico de multicolinealidad",
        "Se mide la co-ocurrencia entre mutaciones con la V de Cramér para detectar "
        "pares redundantes, y el factor de inflación de la varianza (VIF) para "
        "detectar predictoras que se puedan reconstruir a partir de varias otras a "
        "la vez. Cualquiera de las dos inestabilizaría la estimación de parámetros "
        "en el modelo.",
    ),
    (
        "6",
        "Modelado y evaluación",
        "Regresión logística, Random Forest y XGBoost con validación cruzada "
        "anidada (10 pliegues externos y 3 internos), 4 técnicas de balanceo × 4 "
        "optimizadores (48 combinaciones), evaluación final en el conjunto de "
        "prueba y comparación con Friedman y DeLong.",
    ),
]

# En el orden en que se aplican: Shapiro-Wilk decide que la edad va con
# Mann-Whitney, y Fisher sustituye al chi-cuadrado cuando hace falta.
PRUEBAS = [
    (
        "Shapiro-Wilk",
        "Normalidad de la edad",
        "Determina qué familia de pruebas es válida después. Con n = 671 es muy "
        "sensible, así que su resultado se lee junto a la asimetría y la forma de la "
        "distribución, no de forma aislada.",
    ),
    (
        "Mann-Whitney (U)",
        "Edad al diagnóstico frente al grado",
        "No paramétrica: se elige porque Shapiro-Wilk rechaza la normalidad de la "
        "edad, que presenta un patrón bimodal. Compara distribuciones completas, no "
        "solo medias.",
    ),
    (
        "Chi-cuadrado de independencia",
        "Género, grupo racial y cada mutación frente al grado",
        "Contrasta si la distribución de una categórica cambia entre LGG y GBM. Las "
        "categorías con frecuencias muy bajas se agrupan antes para que las "
        "frecuencias esperadas no invaliden la prueba.",
    ),
    (
        "Prueba exacta de Fisher",
        "Tablas 2 × 2 con frecuencias esperadas bajas",
        # Comprobado sobre entrenamiento: la menor frecuencia esperada es 7,6
        # (FAT4 y PDGFRA), así que la sustitución nunca llega a dispararse.
        "Sustituye al chi-cuadrado cuando alguna frecuencia esperada es menor que 5. "
        "En entrenamiento no llega a hacer falta (la menor es 7,6, en FAT4 y "
        "PDGFRA), pero el libro la usa para confirmar el resultado del grupo racial "
        "(p = 0,0038), porque no depende de la aproximación asintótica.",
    ),
    (
        "V de Cramér",
        "Fuerza de las asociaciones categóricas",
        "Complementa al chi-cuadrado: este dice si hay asociación, la V dice cuánta. "
        "Se usa tanto contra el grado como entre pares de genes.",
    ),
    (
        "Correlación de Spearman",
        "Dirección y magnitud frente al grado",
        "Da el signo, que es lo que permite decir si una variable empuja hacia LGG o "
        "hacia GBM. Al ser de rangos, no exige linealidad ni normalidad.",
    ),
]


def _kpis() -> list[dict]:
    """Cifras de la partición y del alcance del análisis."""
    tamanos = tamanos_particion()
    proporciones = proporcion_grado("train").set_index("grado")
    asociacion = asociacion_con_grado("train")

    return [
        {
            "valor": f"{tamanos['train']}",
            "etiqueta": "Pacientes en entrenamiento",
            "detalle": f"LGG {pct(proporciones.loc['LGG', 'porcentaje'], 1)} · "
                       f"GBM {pct(proporciones.loc['GBM', 'porcentaje'], 1)}",
        },
        {
            "valor": f"{tamanos['test']}",
            "etiqueta": "Pacientes en prueba",
            "detalle": f"Reservados antes del EDA ({pct(TEST_SIZE * 100, 0)} del total)",
        },
        {
            "valor": f"{len(asociacion)}",
            "etiqueta": "Variables contrastadas",
            "detalle": f"{len(NUMERIC_FEATURES)} numérica · "
                       f"{len(CATEGORICAL_FEATURES)} categóricas · "
                       f"{len(GENE_FEATURES)} mutaciones",
        },
        {
            "valor": f"Semilla {RANDOM_STATE}",
            "etiqueta": "Reproducibilidad",
            "detalle": "Misma semilla en libro y dashboard",
        },
    ]


def _tabla_variables() -> dbc.Table:
    """Qué prueba recibe cada bloque de variables y por qué."""
    return data_table(
        ["Bloque", "Variables", "Prueba contra el grado", "Motivo"],
        [
            [
                "Clínica numérica",
                "Edad al diagnóstico",
                "Mann-Whitney (U)",
                "La variable es continua y no normal; se comparan las distribuciones "
                "de los dos grupos.",
            ],
            [
                "Clínicas categóricas",
                "Género y grupo racial",
                "Chi-cuadrado (o Fisher) + V de Cramér + Spearman",
                "El género es binario. El grupo racial tiene cuatro niveles sin orden "
                "natural, así que para contrastarlo se reagrupa en dos (White frente "
                "al resto) y así admite también dirección.",
            ],
            [
                "Mutacionales",
                f"{len(GENE_FEATURES)} genes",
                "Chi-cuadrado (o Fisher) + V de Cramér + Spearman",
                "Indicadores 0/1: se contrasta la independencia y se añade Spearman "
                "para conocer la dirección del efecto.",
            ],
        ],
    )


def layout() -> html.Div:
    """Layout de la pestaña de metodología."""
    return html.Div(
        [
            page_header(
                "Metodología",
                "Del CSV a los modelos, en seis etapas reproducibles con la misma "
                "semilla que usa el Jupyter Book.",
                "assignment",
            ),
            kpi_row(_kpis()),
            section_title("Las seis etapas"),
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
                    for i, (numero, titulo, descripcion) in enumerate(ETAPAS)
                ],
                className="g-3",
            ),
            section_title("Qué prueba se aplica a cada variable"),
            card(
                [
                    paragraph(
                        "La prueba no se elige por costumbre sino por la naturaleza del "
                        "dato: una variable continua y bimodal, una categórica de "
                        "cuatro niveles y un indicador binario no admiten el mismo "
                        "contraste."
                    ),
                    _tabla_variables(),
                    callout(
                        "Toda la descripción se hace sobre el conjunto de entrenamiento. "
                        "El de prueba solo se usa al final, para evaluar los tres "
                        "modelos (AUC de 0,923 a 0,930).",
                        titulo="Separación estricta",
                    ),
                ],
                titulo="Correspondencia entre tipo de dato y contraste",
            ),
            section_title("Las pruebas, una a una"),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            [
                                html.Div(aplicacion, className="card-subtitle-custom"),
                                paragraph(motivo),
                            ],
                            titulo=nombre,
                        ),
                        lg=4,
                        md=6,
                        xs=12,
                        className="mb-3",
                    )
                    for i, (nombre, aplicacion, motivo) in enumerate(PRUEBAS)
                ],
                className="g-3",
            ),
            card(
                [
                    bullet_list(
                        [
                            r"Nivel de significancia $\alpha = 0{,}05$ en todos los contrastes.",
                            "Una variable se considera con señal cuando es significativa "
                            r"y además alcanza $|r| \geq 0{,}10$: la significancia sola no basta "
                            "con muestras de este tamaño.",
                            "Con 23 contrastes se controla la tasa de falsos descubrimientos "
                            "con Benjamini-Hochberg (q-valor): las variables con señal la "
                            "conservan después de corregir.",
                            "El umbral de redundancia entre pares de variables se fija en "
                            "V de Cramér > 0,70, y el de multicolinealidad conjunta en un "
                            "VIF de 5.",
                        ],
                    ),
                    html.Div(
                        enlace_externo(
                            "Ver el desarrollo completo del EDA en el libro", URL_LIBRO_EDA
                        ),
                        className="mt-3",
                    ),
                ],
                titulo="Criterios de decisión",
                subtitulo="Los umbrales se fijaron antes de mirar los resultados",
            ),
        ],
        className="vista-pestana",
    )
