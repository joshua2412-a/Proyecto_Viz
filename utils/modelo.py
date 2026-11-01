"""
Los tres clasificadores finales del Jupyter Book, para predecir y explicar.

El dashboard no entrena nada ni carga pickles. Lee modelos/modelos_finales.json,
que escribe `python -m modelado.exportar` (en jbook/) a partir de los pipelines
finales del capítulo 2: los coeficientes de la regresión logística y los
árboles de Random Forest y XGBoost como listas de nodos. Con eso basta numpy
para reproducir sus probabilidades (el exportador comprueba que coinciden con
los modelos originales hasta 1e-5), así que el contenedor no necesita
scikit-learn 1.9, imbalanced-learn ni XGBoost, y un cambio de versión de esas
librerías no rompe el tablero.

Las explicaciones son valores de Shapley exactos:
  * regresión logística: forma cerrada, beta_j * (x_j - media_j), en log-odds;
  * árboles: TreeSHAP dependiente de la trayectoria (Lundberg et al., 2018,
    algoritmo 2), el mismo que usan `shap.TreeExplainer` y el `pred_contribs`
    de XGBoost. En XGBoost sale en log-odds; en Random Forest, que promedia
    probabilidades, sale en probabilidad.
En los tres casos las contribuciones suman exactamente la predicción menos el
valor base.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from utils.config import BASE_DIR

MODELOS_PATH = BASE_DIR / "modelos" / "modelos_finales.json"
RESUMEN_PATH = BASE_DIR / "modelos" / "resumen_modelado.json"

#: Orden fijo de los modelos en todo el tablero.
ORDEN_MODELOS = ["logistica", "random_forest", "xgboost"]
NOMBRE_MODELO = {
    "logistica": "Regresión logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
NOMBRE_BALANCEO = {
    "ninguno": "sin balanceo",
    "smote": "SMOTE-NC",
    "adasyn": "ADASYN",
    "class_weight": "class_weight",
}
NOMBRE_OPTIMIZADOR = {
    "grid": "Grid Search",
    "random": "Random Search",
    "bayesiano": "Bayesiano (TPE)",
    "genetico": "Genético (DEAP)",
}
UMBRAL = 0.5


class ModelosNoDisponibles(FileNotFoundError):
    """Falta modelos/*.json: el capítulo de modelado no se ha exportado."""


@lru_cache(maxsize=1)
def cargar_modelos() -> dict:
    """Lee los modelos portables una sola vez por proceso."""
    if not MODELOS_PATH.exists():
        raise ModelosNoDisponibles(
            f"No se encontró {MODELOS_PATH}. Ejecuta `python -m modelado.exportar` "
            "dentro de jbook/ después de correr 02_Modelado.ipynb."
        )
    datos = json.loads(MODELOS_PATH.read_text(encoding="utf-8"))
    for modelo in datos["modelos"].values():
        if modelo["tipo"] == "bosque":
            modelo["arboles"] = [_arbol_numpy(t) for t in modelo["arboles"]]
    return datos


@lru_cache(maxsize=1)
def cargar_resumen() -> dict:
    """Tablas y curvas del capítulo 2 que muestra la pestaña Modelos."""
    if not RESUMEN_PATH.exists():
        raise ModelosNoDisponibles(
            f"No se encontró {RESUMEN_PATH}. Ejecuta `python -m modelado.exportar` "
            "dentro de jbook/."
        )
    return json.loads(RESUMEN_PATH.read_text(encoding="utf-8"))


def _arbol_numpy(t: dict) -> dict:
    """Convierte las listas del JSON en arrays (y precalcula lo que no cambia)."""
    arbol = {
        "izq": np.asarray(t["izq"], dtype=np.int64),
        "der": np.asarray(t["der"], dtype=np.int64),
        "var": np.asarray(t["var"], dtype=np.int64),
        "umbral": np.asarray(t["umbral"], dtype=float),
        "valor": np.asarray(t["valor"], dtype=float),
        "cobertura": np.asarray(t["cobertura"], dtype=float),
    }
    hojas = arbol["izq"] == -1
    arbol["esperado"] = float(
        (arbol["valor"][hojas] * arbol["cobertura"][hojas]).sum()
        / arbol["cobertura"][0]
    )
    return arbol


# --------------------------------------------------------------------------- #
# Del formulario al vector del modelo
# --------------------------------------------------------------------------- #
def vector_paciente(paciente: dict) -> np.ndarray:
    """Construye el vector de 23 predictoras en el orden del entrenamiento.

    `paciente` usa las variables del dataset: Age_at_diagnosis, Gender, Race
    (0-3) y los 20 genes (0/1). Race se agrupa en Race_binary igual que en el
    libro (0 = White, 1 = cualquier otro grupo) y la edad se estandariza con la
    media y la desviación del entrenamiento.
    """
    datos = cargar_modelos()
    valores = []
    for variable in datos["predictoras"]:
        if variable == "Age_at_diagnosis":
            valores.append(
                (float(paciente[variable]) - datos["media_edad"]) / datos["escala_edad"]
            )
        elif variable == "Race_binary":
            valores.append(0.0 if int(paciente.get("Race", 0)) == 0 else 1.0)
        else:
            valores.append(float(paciente.get(variable, 0)))
    return np.asarray(valores)


def _hoja(arbol: dict, x: np.ndarray, estricto: bool) -> int:
    n = 0
    izq, der, var, umbral = arbol["izq"], arbol["der"], arbol["var"], arbol["umbral"]
    while izq[n] != -1:
        v = x[var[n]]
        n = izq[n] if (v < umbral[n] if estricto else v <= umbral[n]) else der[n]
    return n


def _preparar(modelo: dict, x: np.ndarray) -> np.ndarray:
    # XGBoost compara en float32: sin esto, un paciente justo en un umbral
    # podría ir a la rama contraria que en el modelo original.
    return x.astype(np.float32).astype(float) if modelo.get("float32") else x


def predecir(paciente: dict) -> dict[str, float]:
    """Probabilidad de GBM según cada uno de los tres modelos."""
    datos = cargar_modelos()
    x = vector_paciente(paciente)
    salida = {}
    for clave in ORDEN_MODELOS:
        modelo = datos["modelos"][clave]
        if modelo["tipo"] == "lineal":
            z = float(x @ np.asarray(modelo["coeficientes"]) + modelo["intercepto"])
            salida[clave] = 1 / (1 + np.exp(-z))
            continue
        xp = _preparar(modelo, x)
        estricto = modelo["comparacion"] == "<"
        total = modelo["base"] + sum(
            arbol["valor"][_hoja(arbol, xp, estricto)] for arbol in modelo["arboles"]
        )
        salida[clave] = (
            total / len(modelo["arboles"])
            if modelo["agregacion"] == "media"
            else 1 / (1 + np.exp(-total))
        )
    return {k: float(v) for k, v in salida.items()}


# --------------------------------------------------------------------------- #
# TreeSHAP (Lundberg, Erion y Lee, 2018; algoritmo 2)
# --------------------------------------------------------------------------- #
def _extender(camino, pz, po, pi):
    l = len(camino)
    camino = [list(e) for e in camino] + [[pi, pz, po, 1.0 if l == 0 else 0.0]]
    for i in range(l - 1, -1, -1):
        camino[i + 1][3] += po * camino[i][3] * (i + 1) / (l + 1)
        camino[i][3] = pz * camino[i][3] * (l - i) / (l + 1)
    return camino


def _deshacer(camino, i):
    l = len(camino) - 1
    camino = [list(e) for e in camino]
    n = camino[l][3]
    _, pz, po, _ = camino[i]
    for j in range(l - 1, -1, -1):
        if po != 0:
            t = camino[j][3]
            camino[j][3] = n * (l + 1) / ((j + 1) * po)
            n = t - camino[j][3] * pz * (l - j) / (l + 1)
        else:
            camino[j][3] = camino[j][3] * (l + 1) / (pz * (l - j))
    for j in range(i, l):
        camino[j][0:3] = camino[j + 1][0:3]
    return camino[:l]


def _peso_deshecho(camino, i) -> float:
    """Suma de pesos del camino sin el elemento i (sin construirlo)."""
    l = len(camino) - 1
    n = camino[l][3]
    _, pz, po, _ = camino[i]
    total = 0.0
    for j in range(l - 1, -1, -1):
        if po != 0:
            w = n * (l + 1) / ((j + 1) * po)
            total += w
            n = camino[j][3] - w * pz * (l - j) / (l + 1)
        else:
            total += camino[j][3] * (l + 1) / (pz * (l - j))
    return total


def _treeshap(arbol: dict, x: np.ndarray, estricto: bool, phi: np.ndarray) -> None:
    izq, der, var = arbol["izq"], arbol["der"], arbol["var"]
    umbral, valor, cob = arbol["umbral"], arbol["valor"], arbol["cobertura"]

    def recorrer(j, camino, pz, po, pi):
        camino = _extender(camino, pz, po, pi)
        if izq[j] == -1:
            for i in range(1, len(camino)):
                w = _peso_deshecho(camino, i)
                phi[camino[i][0]] += w * (camino[i][2] - camino[i][1]) * valor[j]
            return
        d = var[j]
        ir_izq = x[d] < umbral[j] if estricto else x[d] <= umbral[j]
        caliente, frio = (izq[j], der[j]) if ir_izq else (der[j], izq[j])
        iz = io = 1.0
        k = next((i for i in range(1, len(camino)) if camino[i][0] == d), None)
        if k is not None:
            iz, io = camino[k][1], camino[k][2]
            camino = _deshacer(camino, k)
        recorrer(caliente, camino, iz * cob[caliente] / cob[j], io, d)
        recorrer(frio, camino, iz * cob[frio] / cob[j], 0.0, d)

    recorrer(0, [], 1.0, 1.0, -1)


@lru_cache(maxsize=256)
def _explicar_cache(clave: str, x_tupla: tuple) -> tuple[float, tuple]:
    modelo = cargar_modelos()["modelos"][clave]
    x = np.asarray(x_tupla)
    if modelo["tipo"] == "lineal":
        beta = np.asarray(modelo["coeficientes"])
        media = np.asarray(modelo["media_fondo"])
        base = float(modelo["intercepto"] + beta @ media)
        return base, tuple(beta * (x - media))
    xp = _preparar(modelo, x)
    estricto = modelo["comparacion"] == "<"
    phi = np.zeros(len(x))
    esperado = 0.0
    for arbol in modelo["arboles"]:
        _treeshap(arbol, xp, estricto, phi)
        esperado += arbol["esperado"]
    if modelo["agregacion"] == "media":
        phi /= len(modelo["arboles"])
        esperado /= len(modelo["arboles"])
    return float(modelo["base"] + esperado), tuple(phi)


def explicar(paciente: dict, clave: str) -> dict:
    """Contribución de cada variable a la predicción de un modelo.

    Devuelve el valor base (la predicción media), la contribución de cada una
    de las 23 predictoras y la escala en que están ("log-odds" o
    "probabilidad"). Base + suma de contribuciones = predicción del modelo en
    esa escala.
    """
    datos = cargar_modelos()
    x = vector_paciente(paciente)
    base, phi = _explicar_cache(clave, tuple(float(v) for v in x))
    return {
        "base": base,
        "contribuciones": dict(zip(datos["predictoras"], phi)),
        "escala": datos["modelos"][clave]["salida"],
    }


# --------------------------------------------------------------------------- #
# Dentro del bosque: lo que dice cada árbol y el camino de un paciente
# --------------------------------------------------------------------------- #
def probas_arboles(paciente: dict, clave: str = "random_forest") -> np.ndarray:
    """Probabilidad de GBM que da cada árbol del bosque a este paciente.

    Random Forest no cuenta votos: su predicción es la media de estas
    probabilidades (la de la hoja a la que llega el paciente en cada árbol).
    """
    modelo = cargar_modelos()["modelos"][clave]
    xp = _preparar(modelo, vector_paciente(paciente))
    estricto = modelo["comparacion"] == "<"
    return np.asarray([a["valor"][_hoja(a, xp, estricto)] for a in modelo["arboles"]])


def arbol_representativo(paciente: dict, clave: str = "random_forest") -> int:
    """Índice (desde 0) del árbol cuya respuesta más se parece a la del bosque."""
    p = probas_arboles(paciente, clave)
    return int(np.argmin(np.abs(p - p.mean())))


def obtener_arbol(indice: int, clave: str = "random_forest") -> dict:
    """Árbol `indice` del bosque, con los arrays de nodos."""
    return cargar_modelos()["modelos"][clave]["arboles"][indice]


def umbral_en_unidades(variable: str, umbral: float) -> float:
    """Deshace la estandarización de la edad para mostrar el corte en años."""
    if variable != "Age_at_diagnosis":
        return umbral
    datos = cargar_modelos()
    return umbral * datos["escala_edad"] + datos["media_edad"]


def camino_paciente(paciente: dict, indice: int, clave: str = "random_forest") -> dict:
    """Nodos por los que pasa el paciente en un árbol, de la raíz a la hoja.

    Cada paso guarda la variable que se pregunta, el corte (en años si es la
    edad) y si el paciente sigue por la rama izquierda (valor <= corte).
    """
    datos = cargar_modelos()
    modelo = datos["modelos"][clave]
    arbol = obtener_arbol(indice, clave)
    xp = _preparar(modelo, vector_paciente(paciente))
    estricto = modelo["comparacion"] == "<"
    izq, der, var, umbral = arbol["izq"], arbol["der"], arbol["var"], arbol["umbral"]
    nodos, pasos, n = [0], [], 0
    while izq[n] != -1:
        variable = datos["predictoras"][var[n]]
        v = xp[var[n]]
        a_izq = bool(v < umbral[n] if estricto else v <= umbral[n])
        pasos.append({
            "nodo": int(n),
            "variable": variable,
            "umbral": float(umbral_en_unidades(variable, float(umbral[n]))),
            "izquierda": a_izq,
            "proba_nodo": float(arbol["valor"][n]),
        })
        n = int(izq[n] if a_izq else der[n])
        nodos.append(n)
    return {
        "nodos": nodos,
        "pasos": pasos,
        "hoja": n,
        "proba": float(arbol["valor"][n]),   # en XGBoost: aporte en log-odds
        "cobertura": float(arbol["cobertura"][n]),
        "menor": estricto,                    # True: el corte es "<", no "<="
        "escala": modelo["salida"],
    }


# --------------------------------------------------------------------------- #
# Dentro de XGBoost y de la regresión logística
# --------------------------------------------------------------------------- #
def aportes_arboles(paciente: dict, clave: str = "xgboost") -> tuple[float, np.ndarray]:
    """Valor base y lo que suma cada árbol de XGBoost, en log-odds.

    XGBoost no promedia: arranca en el valor base y cada árbol añade (o resta)
    el valor de la hoja a la que llega el paciente. La suma final, pasada por la
    sigmoide, es la probabilidad.
    """
    modelo = cargar_modelos()["modelos"][clave]
    xp = _preparar(modelo, vector_paciente(paciente))
    estricto = modelo["comparacion"] == "<"
    aportes = np.asarray([a["valor"][_hoja(a, xp, estricto)] for a in modelo["arboles"]])
    return float(modelo["base"]), aportes


def arbol_mas_influyente(paciente: dict, clave: str = "xgboost") -> int:
    """Índice (desde 0) del árbol que más mueve la predicción de este paciente."""
    _, aportes = aportes_arboles(paciente, clave)
    return int(np.argmax(np.abs(aportes)))


def logodds_logistica(paciente: dict) -> float:
    """Suma de efectos de la regresión logística: intercepto + coeficientes · x."""
    modelo = cargar_modelos()["modelos"]["logistica"]
    x = vector_paciente(paciente)
    return float(x @ np.asarray(modelo["coeficientes"]) + modelo["intercepto"])


def panel_minimo(resumen: dict | None = None) -> dict:
    """Panel más pequeño que no se distingue del completo.

    La misma regla de un error estándar que aplica el notebook (sección
    2.12): el k más pequeño cuya AUC media llega a la del modelo con todas
    las variables menos un error estándar (DE entre pliegues / raíz de K).
    Se calcula aquí para que tablero y libro no puedan contradecirse.
    """
    resumen = resumen or cargar_resumen()
    p = resumen["panel"]
    auc, de = dict(zip(p["k"], p["auc"])), dict(zip(p["k"], p["de"]))
    k_todas = max(p["k"])
    umbral = auc[k_todas] - de[k_todas] / np.sqrt(resumen["meta"]["k_externo"])
    k_min = min(k for k in p["k"] if auc[k] >= umbral)
    return {"k": k_min, "auc_k": auc[k_min], "auc_4": auc.get(4),
            "auc_todas": auc[k_todas], "auc": auc, "umbral": float(umbral)}
