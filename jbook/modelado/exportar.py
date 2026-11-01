"""Exporta los modelos finales y los resultados del capítulo para el dashboard.

Uso (desde jbook/, después de ejecutar 02_Modelado.ipynb)::

    python -m modelado.exportar

Escribe dos archivos en ``modelos/`` (raíz del proyecto), que es lo único que
el dashboard necesita del modelado:

* ``modelos_finales.json``: los tres pipelines finales en un formato
  **portable**, sin pickles: los coeficientes de la regresión logística y los
  árboles de Random Forest y XGBoost como listas de nodos, junto con la media y
  la desviación del escalado de la edad. El dashboard predice y explica con
  numpy puro, así que no necesita scikit-learn 1.9, imbalanced-learn ni
  XGBoost, y un cambio de versión de esas librerías no lo rompe.
* ``resumen_modelado.json``: las tablas y curvas que muestra la pestaña de
  modelos (48 combinaciones, métricas de prueba, predicciones de prueba,
  calibración, SHAP, panel reducido, optimizadores y pruebas estadísticas).

El script verifica que el formato portable reproduce las probabilidades de
los modelos originales antes de escribir nada.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from . import config, datos, evaluacion as ev, experimentos as exp

DIR_SALIDA = config.RAIZ_PROYECTO / "modelos"


# ---------------------------------------------------------------------------
# Modelos portables
# ---------------------------------------------------------------------------
def _escalado(pipe) -> dict:
    esc = pipe.named_steps["escala"].named_transformers_["edad"]
    return {"media_edad": float(esc.mean_[0]), "escala_edad": float(esc.scale_[0])}


def _logistica(pipe, X_fondo) -> dict:
    m = pipe.named_steps["modelo"]
    fondo = pipe.named_steps["escala"].transform(X_fondo).mean(0)
    return {"tipo": "lineal", "coeficientes": m.coef_[0].tolist(),
            "intercepto": float(m.intercept_[0]), "media_fondo": fondo.tolist(),
            "salida": "log-odds"}


def _arbol_sklearn(t) -> dict:
    valor = t.value[:, 0, :]
    valor = valor[:, 1] / valor.sum(1)
    return {"izq": t.children_left.tolist(), "der": t.children_right.tolist(),
            "var": t.feature.tolist(), "umbral": t.threshold.tolist(),
            "valor": np.round(valor, 8).tolist(),
            "cobertura": t.weighted_n_node_samples.tolist()}


def _random_forest(pipe) -> dict:
    m = pipe.named_steps["modelo"]
    return {"tipo": "bosque", "comparacion": "<=", "agregacion": "media",
            "salida": "probabilidad", "base": 0.0,
            "arboles": [_arbol_sklearn(e.tree_) for e in m.estimators_]}


def _xgboost(pipe) -> dict:
    booster = pipe.named_steps["modelo"].get_booster()
    modelo = json.loads(booster.save_raw("json"))
    base = modelo["learner"]["learner_model_param"]["base_score"].strip("[]")
    base = float(base)
    arboles = []
    for t in modelo["learner"]["gradient_booster"]["model"]["trees"]:
        izq, der = t["left_children"], t["right_children"]
        arboles.append({"izq": izq, "der": der, "var": t["split_indices"],
                        "umbral": t["split_conditions"],
                        # en las hojas, split_conditions guarda el peso de la hoja
                        "valor": [t["split_conditions"][i] if izq[i] == -1 else 0.0
                                  for i in range(len(izq))],
                        "cobertura": t["sum_hessian"]})
    return {"tipo": "bosque", "comparacion": "<", "agregacion": "suma",
            "salida": "log-odds", "float32": True,
            "base": float(np.log(base / (1 - base))), "arboles": arboles}


def modelos_portables(finales: dict, X_train) -> dict:
    """Traduce los tres pipelines a un diccionario serializable en JSON."""
    salida = {"predictoras": config.PREDICTORAS, "semilla": config.SEED,
              **_escalado(finales["logistica"]["pipeline"]), "modelos": {}}
    for m, f in finales.items():
        pipe = f["pipeline"]
        portable = {"logistica": lambda: _logistica(pipe, X_train),
                    "random_forest": lambda: _random_forest(pipe),
                    "xgboost": lambda: _xgboost(pipe)}[m]()
        portable.update({"nombre": config.NOMBRES[m], "balanceo": f["balanceo"],
                         "optimizador": f["metodo"], "hiperparametros": f["params"]})
        salida["modelos"][m] = portable
    return salida


def _predecir_portable(portable: dict, X: pd.DataFrame) -> dict:
    """Predicción con el formato portable (la misma lógica que utils/modelo.py)."""
    A = X[config.PREDICTORAS].to_numpy(float).copy()
    A[:, 0] = (A[:, 0] - portable["media_edad"]) / portable["escala_edad"]
    salida = {}
    for m, mod in portable["modelos"].items():
        if mod["tipo"] == "lineal":
            z = A @ np.array(mod["coeficientes"]) + mod["intercepto"]
            salida[m] = 1 / (1 + np.exp(-z))
            continue
        B = A.astype(np.float32) if mod.get("float32") else A
        total = np.full(len(A), mod["base"])
        for t in mod["arboles"]:
            for k, x in enumerate(B):
                n = 0
                while t["izq"][n] != -1:
                    v, u = x[t["var"][n]], t["umbral"][n]
                    ir_izq = v < np.float32(u) if mod["comparacion"] == "<" else v <= u
                    n = t["izq"][n] if ir_izq else t["der"][n]
                total[k] += t["valor"][n]
        if mod["agregacion"] == "media":
            salida[m] = total / len(mod["arboles"])
        else:
            salida[m] = 1 / (1 + np.exp(-total))
    return salida


# ---------------------------------------------------------------------------
# Resumen para la pestaña de modelos
# ---------------------------------------------------------------------------
def _r(v, d=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), d)


def resumen(finales, X_train, y_train, X_test, y_test) -> dict:
    d = config.DIR_RESULTADOS
    r = pd.read_csv(d / "experimentos.csv")
    pl = pd.read_csv(d / "pliegues.csv")
    oof = pd.read_csv(d / "oof.csv")
    hist = json.loads((d / "historiales.json").read_text())
    cache = d / "cache"
    yt = y_test.to_numpy()

    combos = [{"modelo": f.modelo, "balanceo": f.balanceo, "metodo": f.metodo,
               **{f"{m}_{s}": _r(getattr(f, f"{m}_{s}")) for m in
                  ["auc", "recall", "precision", "f1", "brier", "accuracy"]
                  for s in ["media", "de"]},
               "t_total_s": _r(f.t_total_s, 1)} for f in r.itertuples()]

    proba = {m: f["pipeline"].predict_proba(X_test)[:, 1] for m, f in finales.items()}
    modelos = {}
    for m, f in finales.items():
        cid = exp.id_combo(m, f["balanceo"], f["metodo"])
        fila = r[r.combo == cid].iloc[0]
        lo, hi = ev.ic_bootstrap(roc_auc_score, yt, proba[m])
        fpr, tpr, umb = roc_curve(yt, proba[m])
        modelos[m] = {
            "balanceo": f["balanceo"], "metodo": f["metodo"],
            "hiperparametros": f["params"],
            "auc_cv_media": _r(fila.auc_media), "auc_cv_de": _r(fila.auc_de),
            "recall_cv_media": _r(fila.recall_media),
            "auc_pliegues": [_r(v) for v in pl[pl.combo == cid].sort_values("pliegue").auc],
            "auc_test": _r(roc_auc_score(yt, proba[m])), "auc_test_ic": [_r(lo), _r(hi)],
            "brier_test": _r(np.mean((proba[m] - yt) ** 2)),
            "ece_test": _r(ev.ece(yt, proba[m], 5)),
            "ece_oof": _r(ev.ece(y_train, oof[cid].to_numpy())),
            "roc": {"fpr": [_r(v) for v in fpr], "tpr": [_r(v) for v in tpr]},
            "calibracion_oof": ev.curva_confiabilidad(y_train, oof[cid].to_numpy(), 10)
                                 .round(4).to_dict("list"),
            "calibracion_test": ev.curva_confiabilidad(yt, proba[m], 5).round(4).to_dict("list"),
        }

    # Curvas anytime (brecha con el mejor del bloque) por modelo y optimizador
    P = config.PRESUPUESTO
    acum = lambda h: np.pad(np.maximum.accumulate(h), (0, max(0, P - len(h))), mode="edge")[:P]
    anytime = {}
    for m in config.MODELOS:
        curvas = {o: np.array([acum(h) for b in config.BALANCEOS
                               for h in hist[exp.id_combo(m, b, o)]])
                  for o in config.OPTIMIZADORES}
        ref = np.max([c[:, -1] for c in curvas.values()], axis=0)
        anytime[m] = {o: [_r(v, 5) for v in (ref[:, None] - c).mean(0)] for o, c in curvas.items()}
    efic = pd.read_csv(d / "eficiencia_optimizadores.csv")
    optimizadores = {o: {"auc_t6": _r(g.t6.mean()), "auc_t12": _r(g.t12.mean()),
                         "auc_t24": _r(g.t24.mean()), "area": _r(g.area.mean()),
                         "al_mejor": _r(g.llega.mean(), 3),
                         "evals_al_mejor": _r(g.evals.median(), 1),
                         "auc_externa": _r(g.auc.mean()),
                         "optimismo": _r((g.auc_interna - g.auc).mean()),
                         "s_por_eval": _r(g.t_por_eval.mean(), 3)}
                     for o, g in efic.groupby("metodo")}

    # Pruebas estadísticas
    matriz = pl.pivot(index="pliegue", columns="combo", values="auc")
    fr = ev.friedman(matriz)
    factores = {}
    for factor, niveles in [("modelo", config.MODELOS), ("balanceo", config.BALANCEOS),
                            ("metodo", config.OPTIMIZADORES)]:
        otros = [f for f in ("modelo", "balanceo", "metodo") if f != factor] + ["pliegue"]
        mat = pl.pivot_table(index=otros, columns=factor, values="auc")[niveles]
        rf_ = ev.friedman(mat)
        factores[factor] = {"chi2": _r(rf_["chi2"], 2), "gl": rf_["gl"], "p": _r(rf_["p"]),
                            "bloques": rf_["N"],
                            "cd": _r(ev.diferencia_critica(rf_["k"], rf_["N"]), 3),
                            "rangos": {k: _r(v, 3) for k, v in rf_["rangos"].items()}}
    pares = [("logistica", "random_forest"), ("logistica", "xgboost"),
             ("random_forest", "xgboost")]
    delong = []
    for a, b in pares:
        dl = ev.delong(yt, proba[a], proba[b])
        lo, hi = ev.ic_bootstrap(lambda y_, p, q: roc_auc_score(y_, p) - roc_auc_score(y_, q),
                                 yt, proba[a], proba[b])
        sa = pl[pl.combo == exp.id_combo(a, finales[a]["balanceo"], finales[a]["metodo"])].auc
        sb = pl[pl.combo == exp.id_combo(b, finales[b]["balanceo"], finales[b]["metodo"])].auc
        delong.append({"a": a, "b": b, "diferencia": _r(dl["diferencia"]), "z": _r(dl["z"], 3),
                       "p": _r(dl["p"]), "ic": [_r(lo), _r(hi)],
                       "cliff": _r(ev.delta_cliff(sa, sb), 3)})
    for fila, p_aj in zip(delong, ev.holm([x["p"] for x in delong])):
        fila["p_holm"] = _r(p_aj)

    imp = pd.read_csv(d / "importancia_shap.csv")
    panel = pd.read_csv(d / "panel_reducido.csv")
    recal = joblib.load(cache / "recalibracion.joblib")
    semillas = joblib.load(cache / "semillas.joblib")

    return {
        "meta": {"combinaciones": len(r), "configuraciones": int(r.n_eval_total.sum()),
                 "ajustes_internos": int(r.n_eval_total.sum()) * config.K_INTERNO,
                 "k_externo": config.K_EXTERNO, "k_interno": config.K_INTERNO,
                 "presupuesto": config.PRESUPUESTO, "semilla": config.SEED,
                 "minutos": _r(r.t_total_s.sum() / 60, 1),
                 "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                 "gbm_test": int(yt.sum())},
        "combinaciones": combos,
        "modelos": modelos,
        "test": {"y": yt.tolist(), "indice": [int(i) for i in X_test.index],
                 **{m: [_r(v) for v in p] for m, p in proba.items()}},
        "importancia": imp.rename(columns={"Regresión Logística": "logistica",
                                           "XGBoost": "xgboost"}).round(2).to_dict("records"),
        "panel": panel.round(4).to_dict("list"),
        "anytime": anytime,
        "optimizadores": optimizadores,
        "friedman": {"chi2": _r(fr["chi2"], 2), "gl": fr["gl"], "p": _r(fr["p"]),
                     "cd": _r(ev.diferencia_critica(fr["k"], fr["N"]), 2),
                     "variacion_dentro": _r(r.auc_de.mean()),
                     "variacion_entre": _r(r.auc_media.std(ddof=1)),
                     "rango_medias": _r(r.auc_media.max() - r.auc_media.min())},
        "factores": factores,
        "delong": delong,
        "recalibracion": recal.round(4).to_dict("records"),
        "semillas": semillas.groupby("modelo").auc_test.agg(["mean", "std", "min", "max"])
                            .round(4).to_dict("index"),
    }


def main():
    X_train, X_test, y_train, y_test = datos.particion()
    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    finales = joblib.load(config.DIR_RESULTADOS / "cache" / "modelos_finales.joblib")

    portable = modelos_portables(finales, X_train)
    comprobacion = _predecir_portable(portable, X_test)
    for m, f in finales.items():
        original = f["pipeline"].predict_proba(X_test)[:, 1]
        error = np.abs(original - comprobacion[m]).max()
        print(f"{m:<14} error máximo portable vs original: {error:.2e}")
        if error > 1e-5:
            raise RuntimeError(f"El modelo portable de {m} no reproduce el original")

    DIR_SALIDA.mkdir(exist_ok=True)
    (DIR_SALIDA / "modelos_finales.json").write_text(
        json.dumps(portable, separators=(",", ":")), encoding="utf-8")
    (DIR_SALIDA / "resumen_modelado.json").write_text(
        json.dumps(resumen(finales, X_train, y_train, X_test, y_test),
                   ensure_ascii=False, separators=(",", ":"), default=float),
        encoding="utf-8")
    for p in DIR_SALIDA.glob("*.json"):
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
