"""
Asistente virtual del tablero · lógica (sin Dash).

Responde en dos modos y elige solo:

1. **Respuestas guiadas.** Preguntas frecuentes, un glosario y la lectura de
   la predicción, escritas aquí y con las cifras calculadas en vivo desde los
   mismos datos que pintan las pestañas (`asociacion_con_grado` y
   `modelos/resumen_modelado.json`). Funcionan siempre, sin red y sin coste.
2. **Un modelo de IA**, si el servidor tiene su clave (en Cloud Run, como
   secreto). El texto libre lo responde ese modelo con todo lo anterior como
   contexto. Si la llamada falla (sin red, sin cuota), se cae a la respuesta
   guiada en vez de dejar al usuario sin nada. Proveedores, por orden:
   - **Gemini** con `GEMINI_API_KEY`, la clave gratuita de Google AI Studio.
     Es el que usa el proyecto: el plan gratuito basta para un tablero de
     curso. Ojo: en ese plan Google puede usar lo que se le envía para
     mejorar sus productos (aquí son cifras públicas y las preguntas).
   - **Claude** con `ANTHROPIC_API_KEY` (de pago por uso).

Hay tres cosas que nunca pasan por la IA, aunque haya clave: las preguntas
sugeridas (son fijas a propósito: instantáneas y siempre iguales), las órdenes
de navegación («llévame a Modelos») y la lectura de la predicción, cuyas
cifras se calculan con los propios modelos para que no haya ni una inventada.

Este módulo no importa Dash: la interfaz está en utils/asistente_panel.py.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from functools import lru_cache

from utils.config import AUTORES, GENE_DESCRIPCION, URL_DATASET, URL_LIBRO, URL_REPO_LIBRO
from utils.data_loader import asociacion_con_grado, nombre_variable
from utils.formato import lista_y, miles, num, pct, tnum
from utils.modelo import NOMBRE_MODELO, ORDEN_MODELOS, UMBRAL, cargar_resumen, explicar, panel_minimo, predecir
from utils.navegacion import DESCRIPCIONES, ETIQUETAS, ORDEN

log = logging.getLogger(__name__)

# Modelos. Se pueden cambiar sin tocar código con estas variables de entorno.
# Gemini: Flash-Lite está en el plan gratuito de AI Studio con un cupo diario
# mucho más holgado que Flash (que en octubre de 2026 se quedaba en unas 20
# peticiones al día), y basta para responder sobre un contexto cerrado.
# El cupo gratuito es por modelo y por proyecto, y no todos los modelos lo
# tienen en todas las cuentas: si uno responde «sin cupo» (429) o «no
# existe» (404), se prueba el siguiente de la lista.
MODELOS_GEMINI = ([os.environ["ASISTENTE_MODELO_GEMINI"]] if os.environ.get("ASISTENTE_MODELO_GEMINI")
                  else ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"])
# (Los 2.5 ya no se ofrecen a cuentas nuevas: responden 404 desde oct-2026.)
# Claude: Haiku es el más barato y basta para lo mismo.
MODELO_CLAUDE = os.environ.get("ASISTENTE_MODELO", "claude-haiku-5-5")
NOMBRE_PROVEEDOR = {"gemini": "Gemini", "claude": "Claude"}
MAX_PREGUNTA = 500        # caracteres: una pregunta, no un documento
MAX_TURNOS_IA = 8         # mensajes previos que se le pasan a la IA
MAX_TOKENS_IA = 700
# Gemini 3 piensa antes de responder y ese pensamiento cuenta dentro del
# máximo de salida: con 700 a veces se quedaba sin sitio para la respuesta.
MAX_TOKENS_GEMINI = 2048


# --------------------------------------------------------------------------- #
# Utilidades de texto
# --------------------------------------------------------------------------- #
def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni signos: «¿Qué es IDH1?» -> «que es idh1»."""
    sin_tildes = unicodedata.normalize("NFKD", texto or "")
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9ñ ]+", " ", sin_tildes.lower()).strip()


def _contiene(texto_norm: str, clave: str) -> bool:
    """La clave aparece al principio de una palabra («limitac» casa con
    «limitaciones», pero «pten» no casa dentro de otra palabra)."""
    return re.search(r"(?<![a-z0-9])" + re.escape(clave), texto_norm) is not None


def _respuesta(texto: str, ir: str | None = None, fuente: str = "guiada",
               navegar: bool = False, tono: str | None = None, fuerza: float = 0.0) -> dict:
    """Una respuesta del asistente.

    `tono` («lgg», «gbm» o None) y `fuerza` (0–1) tiñen a Glía: su núcleo toma
    el color del grado del que habla la respuesta (assets/glia.js).
    """
    return {"texto": texto.strip(), "ir": ir, "fuente": fuente, "navegar": navegar,
            "tono": tono, "fuerza": round(float(fuerza), 2)}


def _tono_gen(variable: str) -> tuple[str | None, float]:
    """Hacia qué grado empuja una variable en la muestra, y con qué fuerza."""
    h = hechos()
    rho, p = h["rho"].get(variable), h["p"].get(variable)
    if rho is None or p is None or p >= 0.05 or abs(rho) < 0.10:
        return None, 0.0
    return ("lgg" if rho < 0 else "gbm"), min(1.0, abs(rho) / 0.7)


# --------------------------------------------------------------------------- #
# Cifras en vivo (las mismas que muestran las pestañas)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def hechos() -> dict:
    """Las cifras que citan las respuestas, calculadas una vez por proceso."""
    resumen = cargar_resumen()
    tabla = asociacion_con_grado("train")
    modelos = resumen["modelos"]
    panel = dict(zip(resumen["panel"]["k"], resumen["panel"]["auc"]))
    con_senal = tabla[tabla["significativa"]]
    sin_senal = tabla[~tabla["significativa"]]
    return {
        "meta": resumen["meta"],
        "auc": {m: modelos[m]["auc_test"] for m in ORDEN_MODELOS},
        "ic": {m: modelos[m]["auc_test_ic"] for m in ORDEN_MODELOS},
        "config": {m: (modelos[m]["balanceo"], modelos[m]["metodo"]) for m in ORDEN_MODELOS},
        "delong_p_min": min(d["p"] for d in resumen["delong"]),
        "panel4": panel.get(4),
        "panel_min": panel_minimo(resumen),
        "panel23": panel.get(23),
        "importancia": resumen["importancia"],
        "con_senal": [(f["variable"], f["rho"]) for _, f in con_senal.iterrows()],
        "sin_senal": [f["variable"] for _, f in sin_senal.iterrows()],
        "rho": dict(zip(tabla["variable"], tabla["rho"])),
        "p": dict(zip(tabla["variable"], tabla["p_valor"])),
    }


def _nombre(variable: str) -> str:
    """Nombre legible de una predictora («Age_at_diagnosis» -> «Edad»...)."""
    if variable == "Race_binary":
        return "grupo racial"
    try:
        return nombre_variable(variable)
    except Exception:  # noqa: BLE001 - un nombre raro no debe tumbar la respuesta
        return variable


def _rho(v: float) -> str:
    # En LaTeX, como el resto del tablero: los globos del asistente se
    # componen con MathJax (dcc.Markdown con mathjax=True).
    return f"$\\rho = {tnum(v, 2, signo=True)}$"


UMBRAL_RHO = r"$|\rho| \geq 0{,}10$"


def _auc(v: float) -> str:
    return num(v, 3)


# --------------------------------------------------------------------------- #
# Preguntas frecuentes
#   id: lo usan las sugerencias de cada pestaña.
#   claves: comienzos de palabra (ya normalizados) que delatan la pregunta en
#           texto libre. Cuantas más coinciden, mejor la candidata.
#   texto: función que recibe `hechos()` y escribe la respuesta en Markdown.
# --------------------------------------------------------------------------- #
def _r_proyecto(h):
    return (
        "Este tablero estudia si un **subconjunto pequeño de variables** basta para "
        "distinguir un glioma de bajo grado (**LGG**) de un glioblastoma (**GBM**). "
        f"Usa {h['meta']['n_train'] + h['meta']['n_test']} pacientes de TCGA con su edad, "
        "género, grupo racial y 20 mutaciones.\n\n"
        "Se recorre en cuatro etapas: **el contexto** (por qué importa), **el método** "
        "(variables y pruebas), **los resultados** (análisis, modelos y un simulador de "
        "predicción) y **el cierre** (limitaciones y conclusiones)."
    )


def _r_lgg_gbm(h):
    return (
        "Los dos son gliomas, tumores que nacen de las células gliales del sistema "
        "nervioso central, pero se comportan muy distinto:\n\n"
        "- **LGG (glioma de bajo grado):** crecimiento lento y supervivencia larga. "
        "Aparece más en adultos jóvenes y suele portar la mutación de **IDH1**.\n"
        "- **GBM (glioblastoma):** el glioma más agresivo, de crecimiento rápido y "
        "pronóstico corto. Se concentra en edades avanzadas y presenta con más frecuencia "
        "alteraciones en **PTEN** y **EGFR**.\n\n"
        "La clasificación actual de la OMS combina la histología con el perfil "
        "molecular, que es la premisa de este proyecto."
    )


def _r_datos(h):
    m = h["meta"]
    total = m["n_train"] + m["n_test"]
    return (
        f"El dataset es *Glioma Grading Clinical and Mutation Features* (UCI), con "
        f"**{total} pacientes** de TCGA-LGG y TCGA-GBM: 487 LGG y 352 GBM.\n\n"
        "Tiene **23 predictoras**: edad al diagnóstico, género, grupo racial y el estado "
        "(mutado o no) de 20 genes. No hay valores faltantes.\n\n"
        f"Se partió 80/20 de forma estratificada: **{m['n_train']} de entrenamiento** "
        f"y **{m['n_test']} de prueba** ({m['gbm_test']} GBM en prueba).\n\n"
        f"Fuente: [UCI Machine Learning Repository]({URL_DATASET})."
    )


def _r_particion(h):
    m = h["meta"]
    return (
        f"Se reservó el 20 % de los pacientes ({m['n_test']}) **antes de calcular el primer "
        "estadístico**. Así ninguna decisión del análisis exploratorio ni del modelado se "
        "contamina con los datos con los que después se evalúan los modelos.\n\n"
        "La partición es estratificada (mantiene la proporción LGG/GBM) y usa la misma "
        "semilla que el Jupyter Book, por eso las cifras coinciden en los dos sitios."
    )


def _r_asociadas(h):
    top = h["con_senal"][:6]
    lineas = "\n".join(
        f"- **{_nombre(v)}** ({_rho(r)}, hacia {'GBM' if r > 0 else 'LGG'})" for v, r in top
    )
    return (
        "Ordenadas por la fuerza de su asociación con el grado (Spearman con el grado, "
        "sobre entrenamiento):\n\n"
        f"{lineas}\n\n"
        "**IDH1** es el caso extremo: su mutación es casi una firma de LGG. La **edad** es el "
        "principal correlato clínico. En total, "
        f"{len(h['con_senal'])} de 23 variables son significativas *y* tienen {UMBRAL_RHO}."
    )


def _r_no_aportan(h):
    return (
        f"Estas {len(h['sin_senal'])} variables no muestran una asociación apreciable con el "
        r"grado ($p \geq 0{,}05$ o $|\rho| < 0{,}10$):" "\n\n"
        f"{lista_y(_nombre(v) for v in h['sin_senal'])}.\n\n"
        "Que no discriminen el grado **no** las hace irrelevantes en oncología: pueden "
        "importar para el pronóstico o el tratamiento, que este análisis no mide."
    )


def _r_pruebas(h):
    return (
        "Cada variable se contrasta con la prueba que corresponde a su naturaleza "
        r"($\alpha = 0{,}05$):" "\n\n"
        "- **Edad:** U de Mann-Whitney, porque Shapiro-Wilk rechaza la normalidad (es bimodal).\n"
        "- **Género, grupo racial y mutaciones:** chi-cuadrado (o Fisher exacto si hay "
        "frecuencias esperadas < 5), con la **V de Cramér** como tamaño del efecto.\n"
        "- **Spearman** aporta el signo: si la variable empuja hacia LGG o hacia GBM.\n"
        "- **Benjamini-Hochberg** corrige por contrastes múltiples (q-valor).\n"
        "- **V de Cramér entre genes y VIF** descartan la multicolinealidad.\n\n"
        f"Una variable tiene señal si es significativa **y** {UMBRAL_RHO}: con muestras de este "
        "tamaño la significancia sola no basta."
    )


def _r_modelos(h):
    a, ic = h["auc"], h["ic"]
    filas = "\n".join(
        f"- **{NOMBRE_MODELO[m]}:** AUC {_auc(a[m])} (IC 95 %: {_auc(ic[m][0])}–{_auc(ic[m][1])})"
        for m in ORDEN_MODELOS
    )
    return (
        "Se entrenaron tres clasificadores y se evaluaron en los "
        f"{h['meta']['n_test']} pacientes de prueba:\n\n{filas}\n\n"
        "La prueba de DeLong **no separa** ningún par (p ≥ "
        f"{num(h['delong_p_min'], 2)}): con estos datos los tres son equivalentes. Por eso la "
        "elección puede hacerse por interpretabilidad, y la regresión logística no pierde nada."
    )


def _r_entrenamiento(h):
    m = h["meta"]
    return (
        f"No se entrenó un modelo, sino **{m['combinaciones']} combinaciones**: 3 modelos "
        "(logística, Random Forest y XGBoost) × 4 balanceos (ninguno, class_weight, SMOTE-NC "
        "y ADASYN) × 4 optimizadores (Grid, Random, Bayesiano y Genético).\n\n"
        f"Cada una se evaluó con **validación cruzada anidada** ({m['k_externo']} pliegues "
        f"externos × {m['k_interno']} internos): {miles(m['configuraciones'])} "
        f"configuraciones en unos {num(m['minutos'], 0)} minutos. El preprocesamiento y el "
        "balanceo van siempre dentro del pipeline, así que ningún paciente de validación "
        "influye en el entrenamiento.\n\n"
        "La mejor de cada familia se reajustó con los 671 de entrenamiento y se aplicó **una "
        "sola vez** al conjunto de prueba."
    )


def _r_panel(h):
    return (
        "Sí, en estos datos. Un panel de **cuatro variables (IDH1, edad, TP53 y PTEN)** "
        f"alcanza un AUC de **{_auc(h['panel4'])}** en validación cruzada, frente a "
        f"**{_auc(h['panel23'])}** con las 23. Con la regla de un error estándar, el panel "
        f"más pequeño que no se distingue del completo tiene **{h['panel_min']['k']} variables** "
        f"(AUC {_auc(h['panel_min']['auc_k'])}).\n\n"
        "Es la hipótesis central del proyecto: la mayor parte del panel no añade capacidad "
        "para decidir el grado. Ojo: se midió en pacientes a los que se secuenció todo; "
        "convertirlo en un ahorro real exige un estudio prospectivo."
    )


def _r_importancia(h):
    top = h["importancia"][:5]
    lineas = "\n".join(
        f"- **{_nombre(f['variable'])}:** {pct(f['xgboost'], 1)} en XGBoost · "
        f"{pct(f['logistica'], 1)} en la logística" for f in top
    )
    return (
        "Importancia medida con valores **SHAP** (porcentaje del total de |SHAP| medio):\n\n"
        f"{lineas}\n\n"
        "Un modelo lineal y uno de árboles, con supuestos opuestos, coinciden en el orden: "
        "el hallazgo no es un artefacto de uno de ellos."
    )


def _r_usar_prediccion(h):
    return (
        "En **Predicción** describes un paciente y ves qué le predicen los tres modelos:\n\n"
        "1. Elige un perfil de ejemplo o un paciente de prueba (o arma uno tú).\n"
        "2. Ajusta la edad, el género, el grupo racial y marca los genes mutados.\n"
        "3. Pulsa **Predecir**: verás la probabilidad de GBM de cada modelo, el "
        "veredicto y qué variables empujan hacia cada grado.\n"
        "4. Cambia de modelo arriba para ver *cómo decide* cada uno.\n\n"
        "Después pregúntame **«Interpreta mi predicción»** y te la explico."
    )


def _r_limitaciones(h):
    return (
        "Las principales:\n\n"
        "- **Una sola fuente:** todos los pacientes son de TCGA (centros sobre todo de EE. UU.) "
        "y no hay validación externa.\n"
        "- **Sesgo de composición:** más de nueve de cada diez pacientes son del grupo White; "
        "la asociación con el grupo racial no debe leerse como efecto biológico.\n"
        "- **Asociación no es causalidad.**\n"
        "- **Solo el grado histológico:** nada sobre supervivencia ni respuesta al tratamiento.\n"
        "- **Mutaciones como 0/1:** no se distingue el tipo de variante.\n"
        "- **El panel reducido** se midió con datos donde se secuenció todo.\n\n"
        "La pestaña Predicción es para entender los modelos, **no** un apoyo al diagnóstico."
    )


def _r_conclusiones(h):
    a = h["auc"]
    return (
        "Un puñado de marcadores concentra casi toda la información sobre el grado:\n\n"
        f"- Solo {len(h['con_senal'])} de 23 variables muestran una asociación significativa "
        "y apreciable; **IDH1** y la **edad** encabezan la lista.\n"
        f"- Los tres modelos separan los grados con AUC de {_auc(min(a.values()))} a "
        f"{_auc(max(a.values()))} en pacientes nuevos, sin diferencias significativas.\n"
        f"- Con **IDH1, edad, TP53 y PTEN** el AUC es {_auc(h['panel4'])} frente a "
        f"{_auc(h['panel23'])} con las 23; con {h['panel_min']['k']} variables la diferencia "
        "ya no se distingue de la variación entre pliegues.\n"
        "- La señal coincide con la literatura clínica: IDH1, IDH2, ATRX, CIC y TP53 hacia "
        "LGG; edad, PTEN y EGFR hacia GBM."
    )


def _r_documentacion(h):
    return (
        f"- **Jupyter Book** con el análisis y el modelado completos: [abrir el libro]({URL_LIBRO})\n"
        f"- **Repositorio** con el código del tablero y del libro: [GitHub]({URL_REPO_LIBRO})\n\n"
        "La pestaña **Documentación** explica cómo compilarlo y reproducir las cifras."
    )


def _r_autores(h):
    return ("El proyecto es de **" + lista_y(a["nombre"] for a in AUTORES) + "**, para el "
            "curso de Visualización de datos.")


def _r_diagnostico(h):
    return (
        "**No.** Es una herramienta docente: muestra cómo razonan unos modelos entrenados con "
        "671 pacientes de una sola cohorte (TCGA), sin validación externa. El diagnóstico de un "
        "glioma lo establece un equipo médico con histología, imagen y pruebas moleculares."
    )


def _r_navegar(h):
    return (
        "- El **índice** (arriba a la izquierda) muestra las 13 pestañas agrupadas por etapa y "
        "dónde estás.\n"
        "- **Anterior / siguiente** avanzan en orden; los **peldaños** de la doble hélice de la "
        "cabecera también son clicables.\n"
        "- El botón de la **luna / sol** cambia entre modo claro y oscuro.\n"
        "- Y puedes pedirme: *«llévame a Modelos»*."
    )


FAQ = [
    {"id": "proyecto", "pregunta": "¿De qué trata este dashboard?",
     "claves": ["trata", "proyecto", "dashboard", "tablero", "objetivo general", "para que sirve", "resumen"],
     "texto": _r_proyecto, "ir": "tab-introduccion"},
    {"id": "lgg_gbm", "pregunta": "¿Qué diferencia hay entre LGG y GBM?",
     "claves": ["diferencia", "lgg", "gbm", "glioblastoma", "bajo grado", "glioma", "grado"],
     "texto": _r_lgg_gbm, "ir": "tab-contexto"},
    {"id": "datos", "pregunta": "¿Qué datos se usaron?",
     "claves": ["datos", "dataset", "pacientes", "muestra", "tcga", "cohorte", "cuantos"],
     "texto": _r_datos, "ir": "tab-metodologia"},
    {"id": "particion", "pregunta": "¿Por qué se separaron entrenamiento y prueba?",
     "claves": ["particion", "entrenamiento", "prueba", "test", "separ", "reserv", "80"],
     "texto": _r_particion, "ir": "tab-metodologia"},
    {"id": "asociadas", "pregunta": "¿Qué variables se asocian más con el grado?",
     "claves": ["asoci", "importan", "relacion", "influy", "variable", "marcador", "señal", "senal", "correl", "spearman"],
     "texto": _r_asociadas, "ir": "tab-resultados"},
    {"id": "no_aportan", "pregunta": "¿Qué variables no aportan?",
     "claves": ["no aport", "no sirv", "irrelev", "sin asoci", "sin señal", "descart", "no influy"],
     "texto": _r_no_aportan, "ir": "tab-problema"},
    {"id": "pruebas", "pregunta": "¿Qué pruebas estadísticas se usaron?",
     "claves": ["prueba", "estadistic", "contraste", "test", "chi", "mann", "metodo", "hipotesis"],
     "texto": _r_pruebas, "ir": "tab-metodologia"},
    {"id": "modelos", "pregunta": "¿Qué modelo funciona mejor?",
     "claves": ["modelo", "mejor", "auc", "rinde", "rendimiento", "precision", "logistic", "random", "xgboost", "compar"],
     "texto": _r_modelos, "ir": "tab-modelos"},
    {"id": "entrenamiento", "pregunta": "¿Cómo se entrenaron los modelos?",
     "claves": ["entren", "validacion", "cruzada", "anidada", "hiperpar", "optimiz", "balance", "smote", "combinacion"],
     "texto": _r_entrenamiento, "ir": "tab-modelos"},
    {"id": "panel", "pregunta": "¿Bastan pocas variables para decidir el grado?",
     "claves": ["panel", "pocas", "bastan", "cuatro", "4 variables", "reduc", "ahorr", "coste", "costo", "minim"],
     "texto": _r_panel, "ir": "tab-conclusiones"},
    {"id": "importancia", "pregunta": "¿Qué variables pesan más en los modelos?",
     "claves": ["pesan", "peso", "shap", "importancia", "contribu"],
     "texto": _r_importancia, "ir": "tab-modelos"},
    {"id": "usar_prediccion", "pregunta": "¿Cómo uso la pestaña Predicción?",
     "claves": ["como uso", "usar", "simulador", "predecir", "formulario", "probar"],
     "texto": _r_usar_prediccion, "ir": "tab-prediccion"},
    {"id": "limitaciones", "pregunta": "¿Cuáles son las limitaciones?",
     "claves": ["limitac", "sesgo", "debilidad", "confiable", "fiable", "generaliz", "externa"],
     "texto": _r_limitaciones, "ir": "tab-limitaciones"},
    {"id": "conclusiones", "pregunta": "¿Cuáles son las conclusiones?",
     "claves": ["conclus", "hallazgo", "result", "aprend", "concluy"],
     "texto": _r_conclusiones, "ir": "tab-conclusiones"},
    {"id": "documentacion", "pregunta": "¿Dónde está el código y el Jupyter Book?",
     "claves": ["codigo", "jupyter", "libro", "github", "repositorio", "notebook", "reproduc", "documentac"],
     "texto": _r_documentacion, "ir": "tab-documentacion"},
    {"id": "autores", "pregunta": "¿Quién hizo el proyecto?",
     "claves": ["autor", "quien", "hizo", "creador", "equipo"],
     "texto": _r_autores, "ir": None},
    {"id": "diagnostico", "pregunta": "¿Puedo usarlo para diagnosticar?",
     "claves": ["diagnost", "medico", "clinic", "usar en", "paciente real", "tratamiento"],
     "texto": _r_diagnostico, "ir": "tab-limitaciones"},
    {"id": "navegar", "pregunta": "¿Cómo me muevo por el tablero?",
     "claves": ["naveg", "mover", "muevo", "indice", "pestañas", "pestanas", "modo oscuro", "ayuda"],
     "texto": _r_navegar, "ir": None},
]
_FAQ_POR_ID = {f["id"]: f for f in FAQ}


# --------------------------------------------------------------------------- #
# Glosario
#   alias: formas normalizadas con las que se puede preguntar por el término.
# --------------------------------------------------------------------------- #
GLOSARIO = {
    "AUC": (["auc", "area bajo la curva", "roc"],
            "Área bajo la curva ROC. Mide si el modelo **ordena bien** a los pacientes por riesgo "
            "de GBM: 0,5 es azar y 1 es perfecto. No depende del umbral ni se distorsiona con el "
            "desbalance, por eso es el criterio principal del proyecto.", "tab-modelos"),
    "Spearman": (["spearman", "rho", "correlacion de spearman"],
                 "Correlación de rangos entre −1 y 1. Aquí aporta el **signo**: negativo, la "
                 "variable empuja hacia LGG; positivo, hacia GBM. No exige linealidad ni "
                 "normalidad. El umbral de relevancia del proyecto es " + UMBRAL_RHO + ".", "tab-marco"),
    "V de Cramér": (["cramer", "v de cramer"],
                    "Tamaño del efecto del chi-cuadrado, entre 0 y 1: el chi-cuadrado dice **si** "
                    "hay asociación y la V dice **cuánta**. Se usa contra el grado y entre pares de "
                    "genes (redundancia si supera 0,70).", "tab-marco"),
    "U de Mann-Whitney": (["mann", "whitney", "u de mann"],
                          "Prueba no paramétrica que compara las distribuciones completas de dos "
                          "grupos. Se usa para la edad porque no sigue una distribución normal "
                          "(es bimodal).", "tab-marco"),
    "Chi-cuadrado": (["chi cuadrado", "chi2", "chi"],
                     "Contrasta si la distribución de una variable categórica cambia entre LGG y "
                     "GBM. Si alguna frecuencia esperada de una tabla 2×2 baja de 5, se usa la "
                     "prueba exacta de Fisher.", "tab-marco"),
    "p-valor": (["p valor", "pvalor", "p value", "significancia", "significativ"],
                "Probabilidad de ver una asociación al menos tan fuerte si en realidad no la "
                "hubiera. Por debajo de $\\alpha = 0{,}05$ se considera significativa, pero con muestras "
                "grandes conviene mirar también el tamaño del efecto.", "tab-marco"),
    "q-valor (Benjamini-Hochberg)": (["q valor", "benjamini", "hochberg", "falsos descubrimientos", "fdr"],
                                     "p-valor corregido por hacer muchos contrastes a la vez: "
                                     "controla la proporción de falsos descubrimientos. En este "
                                     "proyecto ninguna asociación del panel se cae al corregir.",
                                     "tab-resultados"),
    "VIF": (["vif", "inflacion de la varianza", "multicolinealidad", "colinealidad"],
            "Factor de inflación de la varianza: mide cuánto se explica una predictora con todas "
            "las demás. Por encima de 5 inestabilizaría un modelo; aquí el más alto es el de IDH1, "
            "cerca de 3,1.", "tab-resultados"),
    "SHAP": (["shap", "valores shap", "contribucion"],
             "Reparte la predicción de un modelo entre las variables: cuánto empuja cada una hacia "
             "GBM o hacia LGG respecto a la predicción media. Sumadas al valor base dan exactamente "
             "la predicción.", "tab-prediccion"),
    "Validación cruzada anidada": (["validacion cruzada", "anidada", "nested", "pliegue"],
                                   "Dos bucles: el interno elige los hiperparámetros y el externo "
                                   "mide el rendimiento con datos que esa elección no vio. Evita "
                                   "sobreestimar el desempeño.", "tab-modelos"),
    "DeLong": (["delong"],
               "Prueba que compara las AUC de dos modelos evaluados sobre los mismos pacientes. Aquí "
               "no encuentra diferencias entre los tres modelos.", "tab-modelos"),
    "SMOTE / ADASYN": (["smote", "adasyn", "balanceo", "desbalance", "class weight"],
                       "Técnicas de balanceo: crean pacientes sintéticos de la clase minoritaria "
                       "(SMOTE-NC, ADASYN) o pesan más sus errores (class_weight). En este proyecto "
                       "casi no mueven la AUC; cambian el recall.", "tab-modelos"),
    "Recall": (["recall", "sensibilidad", "exhaustividad"],
               "De los GBM reales, qué proporción detecta el modelo. Es la métrica clínica clave: "
               "un glioblastoma clasificado como LGG retrasa un tratamiento urgente.", "tab-modelos"),
    "Brier y calibración": (["brier", "calibra", "ece"],
                            "La calibración mide si una probabilidad del 70 % acierta de verdad un 70 % "
                            "de las veces. El Brier y el ECE la resumen: más bajo es mejor.",
                            "tab-modelos"),
    "Umbral": (["umbral", "punto de corte", "0 5"],
               "Probabilidad a partir de la cual el modelo dice GBM. Se usa 0,5; fijarlo según el "
               "coste de cada error es una decisión clínica.", "tab-modelos"),
    "Random Forest": (["random forest", "bosque"],
                      "Cientos de árboles de decisión entrenados con muestras distintas que votan: "
                      "la predicción es la media de sus votos.", "tab-prediccion"),
    "XGBoost": (["xgboost", "boosting", "gradient"],
                "Árboles que se construyen uno tras otro, cada uno corrigiendo los errores de los "
                "anteriores; sus aportes se suman en log-odds.", "tab-prediccion"),
    "Regresión logística": (["logistica", "regresion logistica"],
                            "Modelo lineal: suma las variables con un peso cada una y pasa el "
                            "resultado por una curva en S para obtener una probabilidad. Es el más "
                            "interpretable de los tres.", "tab-prediccion"),
    "TCGA": (["tcga", "cancer genome atlas"],
             "The Cancer Genome Atlas: proyecto público de EE. UU. que caracterizó molecularmente "
             "miles de tumores. De ahí vienen los 839 pacientes (TCGA-LGG y TCGA-GBM).",
             "tab-introduccion"),
}

# Los 20 genes también son términos del glosario: función biológica (del marco
# teórico) y hacia qué grado empujan en esta muestra (calculado en vivo).
_GENES = sorted(GENE_DESCRIPCION, key=len, reverse=True)


def _definicion_gen(gen: str) -> str:
    h = hechos()
    rho, p = h["rho"].get(gen), h["p"].get(gen)
    texto = f"**{gen}.** {GENE_DESCRIPCION[gen]}"
    if rho is not None:
        if p is not None and p < 0.05 and abs(rho) >= 0.10:
            hacia = "LGG" if rho < 0 else "GBM"
            texto += (f"\n\nEn esta muestra, su mutación se asocia con **{hacia}** "
                      f"({_rho(rho)}, sobre entrenamiento).")
        else:
            texto += ("\n\nEn esta muestra **no** muestra una asociación apreciable con el grado "
                      f"({_rho(rho)}).")
    return texto


def _buscar_gen(texto_norm: str) -> str | None:
    for gen in _GENES:
        if re.search(r"(?<![a-z0-9])" + gen.lower() + r"(?![a-z0-9])", texto_norm):
            return gen
    return None


def _buscar_termino(texto_norm: str) -> str | None:
    mejor, largo = None, 0
    for termino, (alias, _, _) in GLOSARIO.items():
        for a in alias:
            if _contiene(texto_norm, a) and len(a) > largo:
                mejor, largo = termino, len(a)
    return mejor


# --------------------------------------------------------------------------- #
# Navegación: «llévame a Modelos»
# --------------------------------------------------------------------------- #
_ALIAS_PESTANA = {
    "tab-introduccion": ["introduccion", "inicio", "portada"],
    "tab-contexto": ["contexto", "clinico"],
    "tab-problema": ["problema"],
    "tab-objetivos": ["objetivo"],
    "tab-marco": ["marco", "teori"],
    "tab-metodologia": ["metodologia", "metodo"],
    "tab-exploracion": ["exploracion", "explorar"],
    "tab-resultados": ["resultado"],
    "tab-modelos": ["modelos"],
    "tab-prediccion": ["prediccion", "predecir", "simulador"],
    "tab-limitaciones": ["limitacion"],
    "tab-conclusiones": ["conclusion"],
    "tab-documentacion": ["documentacion", "libro", "jupyter"],
}
_VERBOS_IR = r"(?<![a-z])(llevame|lleva me|llevar|ir a|vamos a|ve a|abre|abrir|abreme|muestrame|ensename|quiero ver|navega)"


def _destino(texto_norm: str) -> str | None:
    """Pestaña nombrada en una orden de navegación, o None si no es una orden."""
    if not re.search(_VERBOS_IR, texto_norm):
        return None
    for tab_id, alias in _ALIAS_PESTANA.items():
        if any(_contiene(texto_norm, a) for a in alias):
            return tab_id
    return None


# --------------------------------------------------------------------------- #
# Lectura de la predicción
# --------------------------------------------------------------------------- #
_INTENCION_PREDICCION = re.compile(
    r"(interpret|explica\w* (mi|la|esta|este|el)|mi (prediccion|perfil|paciente|resultado)"
    r"|este (perfil|paciente|resultado)|esta prediccion|por que (el|los|la) model"
    r"|por que (dice|sale|predice|da)|que significa (mi|este|esta|el resultado))"
)


def _gen_o_variable(v: str, paciente: dict) -> str:
    if v == "Age_at_diagnosis":
        return f"la edad ({num(paciente['Age_at_diagnosis'], 0)} años)"
    if v == "Gender":
        return "el género"
    if v == "Race_binary":
        return "el grupo racial"
    return f"{v} {'mutado' if paciente.get(v) else 'sin mutar'}"


def interpretar_prediccion(estado: dict | None) -> dict:
    """Explica en palabras la predicción que está en la pestaña Predicción.

    Las cifras salen de los mismos modelos y la misma explicación SHAP que la
    pestaña, así que el asistente y el tablero dicen exactamente lo mismo.
    """
    if not estado or not estado.get("paciente"):
        return _respuesta(
            "Todavía no has armado ningún perfil. Abre **Predicción**, elige un paciente o "
            "marca sus mutaciones, pulsa **Predecir** y vuelve a preguntarme.",
            ir="tab-prediccion",
        )
    paciente = estado["paciente"]
    modelo = estado.get("modelo") if estado.get("modelo") in ORDEN_MODELOS else "logistica"
    probas = predecir(paciente)
    exp = explicar(paciente, modelo)

    mutados = [g for g in GENE_DESCRIPCION if paciente.get(g) == 1]
    perfil = (f"Paciente de **{num(paciente['Age_at_diagnosis'], 0)} años** con "
              + (f"**{lista_y(mutados)}** mutado{'s' if len(mutados) > 1 else ''}."
                 if mutados else "**ninguna mutación** del panel."))

    gbm = [m for m in ORDEN_MODELOS if probas[m] >= UMBRAL]
    if len(gbm) == 3:
        veredicto = "Los tres modelos apuntan a **glioblastoma (GBM)**."
    elif not gbm:
        veredicto = "Los tres modelos apuntan a **glioma de bajo grado (LGG)**."
    else:
        veredicto = (f"Los modelos **no coinciden**: {lista_y(NOMBRE_MODELO[m] for m in gbm)} "
                     f"{'dice' if len(gbm) == 1 else 'dicen'} GBM.")
    cifras = " · ".join(f"{NOMBRE_MODELO[m]} {pct(probas[m] * 100, 0)}" for m in ORDEN_MODELOS)

    minimo, maximo = min(probas.values()), max(probas.values())
    if maximo < 0.35 or minimo > 0.65:
        firmeza = "Es una predicción **firme**: las tres probabilidades están lejos del umbral de 50 %."
    elif minimo >= 0.35 and maximo <= 0.65:
        firmeza = ("Está en **zona de incertidumbre**: las tres rondan el 50 %. Con estas variables "
                   "no se puede clasificar con confianza.")
    else:
        firmeza = "Alguna probabilidad está cerca del umbral, así que conviene mirar qué la empuja."

    c, escala = exp["contribuciones"], exp["escala"]
    minimo_c = 0.05 if escala == "log-odds" else 0.01
    hacia_gbm = sorted([v for v in c if c[v] >= minimo_c], key=lambda v: -c[v])[:3]
    hacia_lgg = sorted([v for v in c if c[v] <= -minimo_c], key=lambda v: c[v])[:3]
    empujes = []
    if hacia_gbm:
        empujes.append("- Hacia **GBM**: " + lista_y(_gen_o_variable(v, paciente) for v in hacia_gbm) + ".")
    if hacia_lgg:
        empujes.append("- Hacia **LGG**: " + lista_y(_gen_o_variable(v, paciente) for v in hacia_lgg) + ".")

    real = ""
    cargado = estado.get("cargado") or {}
    if cargado.get("grado") is not None and cargado.get("paciente") == paciente:
        grado = "GBM" if cargado["grado"] == 1 else "LGG"
        aciertos = [m for m in ORDEN_MODELOS if int(probas[m] >= UMBRAL) == cargado["grado"]]
        real = (f"\n\nEs el paciente {cargado.get('indice')} de prueba y su diagnóstico real en TCGA "
                f"es **{grado}**: " + ("aciertan los tres modelos." if len(aciertos) == 3 else
                                       "no acierta ningún modelo." if not aciertos else
                                       f"acierta{'n' if len(aciertos) > 1 else ''} "
                                       f"{lista_y(NOMBRE_MODELO[m] for m in aciertos)}."))

    texto = (
        f"{perfil}\n\n{veredicto} Probabilidad de GBM: {cifras}.\n\n{firmeza}\n\n"
        f"Según **{NOMBRE_MODELO[modelo]}** (el modelo elegido en la pestaña), lo que más pesa:\n\n"
        + ("\n".join(empujes) if empujes else "- Ninguna variable empuja de forma apreciable.")
        + real
        + "\n\n*Es una lectura de los modelos, no un diagnóstico.*"
    )
    p = probas[modelo]
    return _respuesta(texto, ir="tab-prediccion", fuente="prediccion",
                      tono="gbm" if p >= UMBRAL else "lgg", fuerza=abs(p - 0.5) * 2)


# --------------------------------------------------------------------------- #
# Sugerencias por pestaña (ids de FAQ, «g:<término>» del glosario o «prediccion»)
# --------------------------------------------------------------------------- #
SUGERENCIAS = {
    "tab-introduccion": ["proyecto", "navegar", "datos"],
    "tab-contexto": ["lgg_gbm", "g:IDH1", "datos"],
    "tab-problema": ["no_aportan", "panel", "asociadas"],
    "tab-objetivos": ["panel", "particion", "proyecto"],
    "tab-marco": ["pruebas", "g:Spearman", "g:V de Cramér"],
    "tab-metodologia": ["particion", "pruebas", "g:q-valor (Benjamini-Hochberg)"],
    "tab-exploracion": ["asociadas", "g:p-valor", "g:U de Mann-Whitney"],
    "tab-resultados": ["asociadas", "no_aportan", "g:VIF"],
    "tab-modelos": ["modelos", "entrenamiento", "g:AUC"],
    "tab-prediccion": ["prediccion", "usar_prediccion", "g:SHAP"],
    "tab-limitaciones": ["limitaciones", "diagnostico", "datos"],
    "tab-conclusiones": ["conclusiones", "panel", "importancia"],
    "tab-documentacion": ["documentacion", "autores", "entrenamiento"],
}


def texto_sugerencia(clave: str) -> str:
    if clave == "prediccion":
        return "Interpreta mi predicción"
    if clave.startswith("g:"):
        return f"¿Qué es {clave[2:]}?"
    return _FAQ_POR_ID[clave]["pregunta"]


def sugerencias(pestana: str | None) -> list[tuple[str, str]]:
    claves = SUGERENCIAS.get(pestana or "", SUGERENCIAS["tab-introduccion"])
    return [(c, texto_sugerencia(c)) for c in claves]


def responder_sugerencia(clave: str, estado_pred: dict | None) -> dict:
    """Respuesta fija de una sugerencia: nunca pasa por Claude."""
    if clave == "prediccion":
        return interpretar_prediccion(estado_pred)
    if clave.startswith("g:"):
        termino = clave[2:]
        if termino in GLOSARIO:
            _, definicion, ir = GLOSARIO[termino]
            return _respuesta(f"**{termino}.** {definicion}", ir=ir)
        if termino in GENE_DESCRIPCION:
            return _respuesta(_definicion_gen(termino), "tab-marco", tono=_tono_gen(termino)[0],
                              fuerza=_tono_gen(termino)[1])
    faq = _FAQ_POR_ID.get(clave)
    if faq:
        return _respuesta(faq["texto"](hechos()), ir=faq["ir"])
    return _respuesta(_SIN_RESPUESTA)


# --------------------------------------------------------------------------- #
# Respuesta guiada a texto libre
# --------------------------------------------------------------------------- #
_SIN_RESPUESTA = (
    "No tengo una respuesta preparada para eso. Puedo explicarte **los datos**, **las pruebas**, "
    "**los modelos**, **las conclusiones**, definir un término (*«¿qué es el AUC?»*), "
    "interpretar **tu predicción** o llevarte a una pestaña (*«llévame a Modelos»*)."
)
_PIDE_DEFINICION = re.compile(r"(que es|que son|que significa|significado|define|definicion|para que sirve|explica)")


def respuesta_guiada(pregunta: str, estado_pred: dict | None = None, minimo: int = 1) -> dict:
    t = normalizar(pregunta)
    if not t:
        return _respuesta(_SIN_RESPUESTA)

    # Un término por el que se pregunta directamente («¿qué es la V de
    # Cramér?», «IDH1») gana a las preguntas frecuentes, que son más amplias.
    corta = len(t.split()) <= 3
    gen = _buscar_gen(t)
    termino = _buscar_termino(t)
    if (gen or termino) and (corta or _PIDE_DEFINICION.search(t)):
        if gen:
            return _respuesta(_definicion_gen(gen), "tab-marco", tono=_tono_gen(gen)[0],
                              fuerza=_tono_gen(gen)[1])
        _, definicion, ir = GLOSARIO[termino]
        return _respuesta(f"**{termino}.** {definicion}", ir=ir)

    # Una clave de varias palabras («no sirv») pesa más que una suelta
    # («variables»): es mucho más específica de su pregunta.
    puntos = [(sum((1 + c.count(" ")) * _contiene(t, c) for c in f["claves"]), -i, f)
              for i, f in enumerate(FAQ)]
    mejor = max(puntos, key=lambda x: (x[0], x[1]))
    if mejor[0] >= minimo:
        faq = mejor[2]
        return _respuesta(faq["texto"](hechos()), ir=faq["ir"])
    if gen:
        return _respuesta(_definicion_gen(gen), "tab-marco", tono=_tono_gen(gen)[0],
                          fuerza=_tono_gen(gen)[1])
    if termino:
        _, definicion, ir = GLOSARIO[termino]
        return _respuesta(f"**{termino}.** {definicion}", ir=ir)
    return _respuesta(_SIN_RESPUESTA)


# --------------------------------------------------------------------------- #
# IA (Gemini o Claude)
# --------------------------------------------------------------------------- #
def _clave_gemini() -> str | None:
    # GOOGLE_API_KEY es el otro nombre que acepta la librería de Google
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def proveedor_ia() -> str | None:
    """«gemini», «claude» o None: el primero con clave y librería instalada.

    Se mira en cada pregunta para que añadir el secreto en Cloud Run no exija
    tocar el código.
    """
    if _clave_gemini():
        try:
            from google import genai  # noqa: F401
            return "gemini"
        except ImportError:
            log.warning("Asistente: hay GEMINI_API_KEY pero falta la librería google-genai.")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return "claude"
        except ImportError:
            log.warning("Asistente: hay ANTHROPIC_API_KEY pero falta la librería anthropic.")
    return None


def ia_disponible() -> bool:
    return proveedor_ia() is not None


@lru_cache(maxsize=1)
def _contexto_ia() -> str:
    """Todo lo que la IA puede citar. Es fijo por proceso, así que se reutiliza
    entre preguntas: Claude lo guarda en caché de forma explícita y Gemini de
    forma implícita, y las preguntas siguientes no lo pagan entero."""
    h = hechos()
    pestanas = "\n".join(f"- {tab_id}: {ETIQUETAS[tab_id]} — {DESCRIPCIONES[tab_id]}" for tab_id in ORDEN)
    asociacion = "\n".join(
        f"- {_nombre(v)} ({v}): ρ = {num(h['rho'][v], 3)}, p = {h['p'][v]:.2e}, "
        f"{'con señal' if v in dict(h['con_senal']) else 'sin señal apreciable'}"
        for v in h["rho"]
    )
    faq = "\n\n".join(f"### {f['pregunta']}\n{f['texto'](h)}" for f in FAQ)
    glosario = "\n".join(f"- **{t}**: {d}" for t, (_, d, _) in GLOSARIO.items())
    genes = "\n".join(f"- {g}: {d}" for g, d in GENE_DESCRIPCION.items())
    return f"""Eres Glía, la asistente virtual de un dashboard académico en Dash sobre gliomas: «Caracterización Clínico-Molecular del Grado Tumoral en Gliomas» (LGG frente a GBM, datos TCGA), de {lista_y(a['nombre'] for a in AUTORES)}. En el tablero te dibujan como una célula glial (un astrocito) hecha de partículas, porque los gliomas nacen de células como tú.

Cómo responder:
- Hablas en primera persona como Glía, con cercanía, pero sin exagerar el personaje: lo importante es la explicación, no la célula.
- No saludes ni te presentes en cada respuesta: entra directo a la explicación.
- Cuando expliques un gráfico, no describas detalles visuales (colores, rayas, formas, posición) que no aparezcan en su nota: no ves el gráfico, solo su título, su nota y las cifras de este contexto.
- Siempre en español, con coma decimal (0,923) y en un tono claro y cercano.
- Breve: normalmente 2 a 6 frases o una lista corta. Markdown sencillo (negritas y listas); nada de tablas ni encabezados.
- La notación matemática va en LaTeX entre $...$, como en el resto del tablero (por ejemplo $\\rho = -0{{,}}70$ o $|\\rho| \\geq 0{{,}}10$).
- Usa SOLO las cifras y hechos de este contexto. Si algo no está aquí, dilo con franqueza y sugiere la pestaña o el Jupyter Book donde podría estar. No inventes cifras.
- Solo hablas de este tablero y de lo que hace falta para entenderlo (gliomas, estadística, modelos). Si la pregunta es ajena (cultura general, tareas, programación de otra cosa…), no la respondas: di en una frase que solo puedes ayudar con el tablero y sugiere algo que sí puedes explicar.
- No des consejo médico ni diagnósticos: el tablero es una herramienta docente. Si alguien describe un caso real, recomiéndale hablar con su equipo médico.
- Si la respuesta está desarrollada en una pestaña concreta, termina con una única etiqueta [[ir:ID]] usando uno de estos ID. No escribas la etiqueta en ningún otro caso.

Pestañas (ID: nombre — contenido):
{pestanas}

Asociación de cada predictora con el grado (Spearman con el grado sobre los 671 de entrenamiento; ρ negativo empuja hacia LGG, positivo hacia GBM; «con señal» = p < 0,05 y |ρ| ≥ 0,10):
{asociacion}

Función de los genes del panel:
{genes}

Glosario:
{glosario}

Preguntas frecuentes con su respuesta de referencia:
{faq}

Enlaces: Jupyter Book {URL_LIBRO} · Repositorio {URL_REPO_LIBRO} · Dataset {URL_DATASET}
"""


def _situacion(pestana: str | None, estado_pred: dict | None, extra: str = "") -> str:
    """Lo que cambia en cada pregunta: dónde está el usuario y su predicción."""
    partes = [f"El usuario está en la pestaña: {ETIQUETAS.get(pestana or '', 'Introducción')}."]
    if extra:
        partes.append(extra)
    if estado_pred and estado_pred.get("paciente"):
        partes.append("Lectura de la predicción que tiene ahora en la pestaña Predicción "
                      "(cifras exactas de los modelos):\n" + interpretar_prediccion(estado_pred)["texto"])
    return "\n\n".join(partes)


_ETIQUETA_IR = re.compile(r"\[\[\s*ir\s*:\s*([a-z\-]+)\s*\]\]", re.IGNORECASE)


def _turnos(pregunta: str, historial: list[dict]) -> list[tuple[str, str]]:
    """La conversación como [(«usuario»|«asistente», texto)], lista para la IA.

    Las dos APIs exigen alternar turnos y empezar por el usuario: dos
    mensajes seguidos del mismo rol se unen y se descarta lo que hubiera antes
    de la primera pregunta (el saludo inicial no es un turno real).
    """
    turnos: list[list[str]] = []
    for m in historial[-MAX_TURNOS_IA:]:
        if m.get("inicial") or not m.get("texto"):
            continue
        rol = "usuario" if m.get("rol") == "usuario" else "asistente"
        if turnos and turnos[-1][0] == rol:
            turnos[-1][1] += "\n\n" + m["texto"]
        else:
            turnos.append([rol, m["texto"]])
    while turnos and turnos[0][0] != "usuario":
        turnos.pop(0)
    if turnos and turnos[-1][0] == "usuario":
        turnos[-1][1] += "\n\n" + pregunta
    else:
        turnos.append(["usuario", pregunta])
    return [(rol, texto) for rol, texto in turnos]


def _llamar_claude(turnos, situacion: str) -> str:
    import anthropic

    cliente = anthropic.Anthropic(timeout=25.0, max_retries=1)
    respuesta = cliente.messages.create(
        model=MODELO_CLAUDE,
        max_tokens=MAX_TOKENS_IA,
        system=[
            {"type": "text", "text": _contexto_ia(), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": situacion},
        ],
        messages=[{"role": "user" if rol == "usuario" else "assistant", "content": texto}
                  for rol, texto in turnos],
    )
    return "".join(b.text for b in respuesta.content if getattr(b, "type", "") == "text")


def _llamar_gemini(turnos, situacion: str) -> str:
    from google import genai
    from google.genai import types

    cliente = genai.Client(
        api_key=_clave_gemini(),
        # 25 s como con Claude: más vale caer a la respuesta guiada que dejar
        # al usuario mirando los puntos de «escribiendo».
        http_options=types.HttpOptions(timeout=25_000),
    )
    contenido = [types.Content(role="user" if rol == "usuario" else "model",
                               parts=[types.Part(text=texto)])
                 for rol, texto in turnos]
    config = types.GenerateContentConfig(
        # El contexto fijo va primero y la situación al final: Gemini
        # reutiliza en caché el principio común de las peticiones.
        system_instruction=_contexto_ia() + "\n\n" + situacion,
        max_output_tokens=MAX_TOKENS_GEMINI,
        temperature=0.3,
    )
    ultimo_error = None
    for modelo in MODELOS_GEMINI:
        if _GEMINI_EN_PAUSA.get(modelo, 0) > time.time():
            continue
        try:
            respuesta = cliente.models.generate_content(model=modelo, contents=contenido, config=config)
            return respuesta.text or ""
        except Exception as error:  # noqa: BLE001 - se decide abajo si se prueba otro
            texto = str(error)
            # 402: la clave es de un proyecto con facturación de prepago y sin
            # saldo. Afecta a todos los modelos, así que no se prueba otro.
            if "402" in texto or "prepayment" in texto.lower():
                raise
            sin_cupo = "429" in texto or "RESOURCE_EXHAUSTED" in texto
            no_existe = "404" in texto or "NOT_FOUND" in texto
            if not (sin_cupo or no_existe):
                raise
            # Un modelo sin cupo se deja de probar un rato (el cupo por minuto
            # se recupera solo); uno que no existe, hasta reiniciar el servidor.
            _GEMINI_EN_PAUSA[modelo] = time.time() + (600 if sin_cupo else 10**9)
            log.warning("Asistente: %s no disponible (%s); pruebo el siguiente modelo.",
                        modelo, texto[:300])
            ultimo_error = error
    raise ultimo_error or RuntimeError("429 RESOURCE_EXHAUSTED: ningún modelo de Gemini con cupo")


# Modelo de Gemini -> hasta cuándo no se prueba (segundos de time.time()).
_GEMINI_EN_PAUSA: dict[str, float] = {}


def responder_ia(pregunta: str, historial: list[dict], pestana: str | None,
                 estado_pred: dict | None, proveedor: str = "gemini", extra: str = "") -> dict:
    turnos = _turnos(pregunta, historial)
    situacion = _situacion(pestana, estado_pred, extra)
    llamar = _llamar_gemini if proveedor == "gemini" else _llamar_claude
    texto = llamar(turnos, situacion)
    ir = None
    encontrada = _ETIQUETA_IR.search(texto)
    if encontrada and encontrada.group(1) in ETIQUETAS:
        ir = encontrada.group(1)
    texto = _ETIQUETA_IR.sub("", texto).strip()
    if not texto:
        raise RuntimeError("la IA devolvió una respuesta vacía")
    respuesta = _respuesta(texto, ir=ir, fuente="ia")
    respuesta["proveedor"] = NOMBRE_PROVEEDOR[proveedor]
    return respuesta


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #
def responder(pregunta: str, historial: list[dict] | None = None, pestana: str | None = None,
              estado_pred: dict | None = None) -> dict:
    """Responde a una pregunta escrita por el usuario.

    Orden: navegación y lectura de la predicción (deterministas), la IA si
    hay clave y, si no la hay o falla, la respuesta guiada.
    """
    pregunta = (pregunta or "").strip()[:MAX_PREGUNTA]
    t = normalizar(pregunta)

    destino = _destino(t)
    if destino:
        return _respuesta(f"Te llevo a **{ETIQUETAS[destino]}**: {DESCRIPCIONES[destino]}",
                          ir=destino, navegar=True)

    if _INTENCION_PREDICCION.search(t):
        return interpretar_prediccion(estado_pred)

    # Saludos y agradecimientos: no merecen una llamada a la IA.
    if re.fullmatch(r"(hola|buenas|buenos dias|buenas tardes|buenas noches|hey|gracias|muchas gracias|ok|vale|listo)( .{0,20})?", t):
        if t.startswith(("gracias", "muchas", "ok", "vale", "listo")):
            return _respuesta("¡Con gusto! Si quieres seguir, elige una de las preguntas sugeridas o escríbeme otra.")
        return _respuesta("¡Hola! Soy el asistente del tablero. " + _SIN_RESPUESTA.split(". ", 1)[1])

    proveedor = proveedor_ia()
    if proveedor:
        try:
            return responder_ia(pregunta, historial or [], pestana, estado_pred, proveedor)
        except Exception as error:  # noqa: BLE001 - red, cuota agotada, clave mala...
            nombre = NOMBRE_PROVEEDOR[proveedor]
            log.warning("Asistente: %s no respondió (%s); uso la respuesta guiada.", nombre, error)
            guiada = respuesta_guiada(pregunta, estado_pred)
            detalle = str(error)
            sin_saldo = "402" in detalle or "prepayment" in detalle.lower()
            cuota = "429" in detalle or "RESOURCE_EXHAUSTED" in detalle
            guiada["aviso"] = (
                f"La clave de {nombre} es de un proyecto con facturación sin saldo; "
                "te dejo la respuesta guiada." if sin_saldo else
                f"{nombre} alcanzó su límite de uso por ahora; te dejo la respuesta guiada." if cuota else
                f"No pude contactar con {nombre}; te dejo la respuesta guiada.")
            return guiada

    return respuesta_guiada(pregunta, estado_pred)


# --------------------------------------------------------------------------- #
# «Pregúntale a Glía» junto a un gráfico
# --------------------------------------------------------------------------- #
def _sin_latex(texto: str) -> str:
    return re.sub(r"\$([^$]*)\$", r"\1", texto or "")


def explicar_grafico(titulo: str, nota: str, historial: list[dict] | None = None,
                     pestana: str | None = None, estado_pred: dict | None = None) -> dict:
    """Responde a «Explícame este gráfico» con el título y la nota de su tarjeta.

    Con IA, el título y la nota van como contexto. Sin IA, la respuesta es la
    propia nota de la tarjeta más la pregunta frecuente que mejor encaja con
    el título, para que el chip nunca se quede sin respuesta.
    """
    pregunta = f"Explícame el gráfico «{titulo}»"
    proveedor = proveedor_ia()
    if proveedor:
        extra = (f"El usuario pulsó «Pregúntale a Glía» junto al gráfico «{titulo}»."
                 + (f" La nota al pie de ese gráfico dice: {_sin_latex(nota)}" if nota else "")
                 + " Explica qué muestra y cómo leerlo en 3 a 5 frases, con las cifras del contexto.")
        try:
            return responder_ia(pregunta, historial or [], pestana, estado_pred, proveedor, extra)
        except Exception as error:  # noqa: BLE001 - se cae a la respuesta guiada
            log.warning("Asistente: no pude explicar el gráfico con IA (%s).", error)
    partes = [f"**{titulo}.**" + (f" {nota}" if nota else "")]
    # Solo se añade una pregunta frecuente si el título encaja con claridad
    # (dos claves o más): con una sola, «Distribución de la variable
    # objetivo» acababa respondiendo con el ranking de asociaciones.
    relacionada = respuesta_guiada(titulo, estado_pred, minimo=2)
    if relacionada["texto"] != _SIN_RESPUESTA:
        partes.append(relacionada["texto"])
    else:
        partes.append("Si quieres más detalle, pregúntame por una variable o una prueba "
                      "concreta del gráfico y te la explico.")
    return _respuesta("\n\n".join(partes), ir=None)
