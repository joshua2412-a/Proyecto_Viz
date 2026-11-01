"""Exporta los resultados del capítulo 2 para el libro en R (bookdown).

El libro de R no reentrena nada: lee las tablas que deja este script, que se
calculan a partir de ``resultados/`` y ``resultados/cache/`` con las mismas
funciones del paquete (``modelado.evaluacion``) y las mismas semillas que usa
``02_Modelado.ipynb``. Así las cifras del libro de R son idénticas a las del
Jupyter Book, y R no necesita leer archivos ``.joblib``.

Uso (desde ``jbook/``, después de ejecutar ``02_Modelado.ipynb``)::

    python -m modelado.exportar_r --destino ../../Proyecto_Viz_R/rbook_gliomas/modelado
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from . import config, datos, evaluacion as ev

R = config.DIR_RESULTADOS
CACHE = R / "cache"
N = config.NOMBRES
P = config.PRESUPUESTO


def id_combo(modelo: str, balanceo: str, metodo: str) -> str:
    return f"{modelo}|{balanceo}|{metodo}"


def acumulada(h) -> np.ndarray:
    c = np.maximum.accumulate(np.asarray(h, dtype=float))
    return np.pad(c, (0, max(0, P - len(c))), mode="edge")[:P]


def metricas(y, proba, umbral: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= umbral).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "auc": roc_auc_score(y, proba), "brier": brier_score_loss(y, proba)}


def exportar(destino: Path) -> list[str]:
    destino.mkdir(parents=True, exist_ok=True)
    escalares: dict = {}
    archivos: list[str] = []

    def guardar(df: pd.DataFrame, nombre: str):
        df.to_csv(destino / nombre, index=False, encoding="utf-8")
        archivos.append(nombre)

    # -----------------------------------------------------------------------
    # 2.1 Entorno y partición
    # -----------------------------------------------------------------------
    entorno = json.loads((R / "entorno.json").read_text())
    guardar(pd.DataFrame(entorno.items(), columns=["componente", "valor"]), "entorno.csv")

    X_train, X_test, y_train, y_test = datos.particion()
    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    yt = y_test.to_numpy()
    guardar(pd.DataFrame({
        "conjunto": ["Entrenamiento", "Prueba"],
        "pacientes": [len(y_train), len(y_test)],
        "lgg": [int((y_train == 0).sum()), int((y_test == 0).sum())],
        "gbm": [int((y_train == 1).sum()), int((y_test == 1).sum())],
    }), "particion.csv")

    # -----------------------------------------------------------------------
    # 2.3 Validación anidada
    # -----------------------------------------------------------------------
    resumen = pd.read_csv(R / "experimentos.csv")
    pliegues = pd.read_csv(R / "pliegues.csv")
    historiales = json.loads((R / "historiales.json").read_text())
    geneticos = json.loads((R / "geneticos.json").read_text())
    oof = pd.read_csv(R / "oof.csv")
    guardar(resumen, "experimentos.csv")
    guardar(pliegues.drop(columns=["params"]), "pliegues.csv")
    escalares["n_eval_total"] = int(resumen["n_eval_total"].sum())
    escalares["minutos_total"] = float(resumen["t_total_s"].sum() / 60)

    # -----------------------------------------------------------------------
    # 2.4 Optimizadores: curvas anytime, eficiencia, historial y genético
    # -----------------------------------------------------------------------
    filas = []
    for m in config.MODELOS:
        curvas = {o: np.array([acumulada(h) for b in config.BALANCEOS
                               for h in historiales[id_combo(m, b, o)]])
                  for o in config.OPTIMIZADORES}
        referencia = np.max([c[:, -1] for c in curvas.values()], axis=0)
        for o in config.OPTIMIZADORES:
            brecha = (referencia[:, None] - curvas[o]).mean(0)
            filas += [{"modelo": m, "metodo": o, "t": t + 1, "brecha": float(v)}
                      for t, v in enumerate(brecha)]
    guardar(pd.DataFrame(filas), "anytime_brecha.csv")

    bloques = pd.read_csv(R / "eficiencia_optimizadores.csv")
    eficiencia = bloques.groupby("metodo").agg(
        t6=("t6", "mean"), t12=("t12", "mean"), t24=("t24", "mean"), area=("area", "mean"),
        llega=("llega", "mean"), evals=("evals", "median"), auc_externa=("auc", "mean"),
        auc_interna=("auc_interna", "mean"), t_eval=("t_por_eval", "mean"),
        t_total=("t_busqueda", "sum")).reindex(config.OPTIMIZADORES).reset_index()
    guardar(eficiencia, "eficiencia.csv")

    pruebas = []
    ref = bloques[bloques.metodo == "random"].set_index(["modelo", "balanceo", "pliegue"])
    for o in ["grid", "bayesiano", "genetico"]:
        otro = bloques[bloques.metodo == o].set_index(["modelo", "balanceo", "pliegue"])
        for col in ["t12", "t24", "area", "auc"]:
            d = otro[col] - ref.loc[otro.index, col]
            pruebas.append({"metodo": o, "medida": col, "dif_media": d.mean(),
                            "p": stats.wilcoxon(d, zero_method="zsplit").pvalue})
    pruebas = pd.DataFrame(pruebas)
    pruebas["p_holm"] = ev.holm(pruebas["p"])
    guardar(pruebas, "wilcoxon_optimizadores.csv")

    filas = []
    for o in ["random", "bayesiano"]:
        h = np.asarray(historiales[id_combo("xgboost", "ninguno", o)][0], dtype=float)
        filas += [{"metodo": o, "t": t + 1, "auc": v, "mejor": b}
                  for t, (v, b) in enumerate(zip(h, np.maximum.accumulate(h)))]
    guardar(pd.DataFrame(filas), "historial_xgb_pliegue1.csv")

    gen = pd.DataFrame([{"modelo": c.split("|")[0], "corrida": f"{c}|{k}", **g}
                        for c, por_pliegue in geneticos.items()
                        for k, registros in enumerate(por_pliegue) for g in registros])
    gen_m = (gen[gen.generacion <= 8]
             .groupby(["modelo", "generacion"])[["mejor", "media", "diversidad"]]
             .mean().reset_index())
    guardar(gen_m, "genetico_generaciones.csv")
    por_corrida = gen.groupby(["modelo", "corrida"]).agg(
        generaciones=("generacion", "max"), div_inicial=("diversidad", "first"),
        div_final=("diversidad", "last"))
    por_corrida["colapso"] = por_corrida.div_final < 0.05
    g = por_corrida.groupby("modelo")
    guardar(pd.DataFrame({
        "generaciones_media": g.generaciones.mean(), "generaciones_max": g.generaciones.max(),
        "div_inicial": g.div_inicial.mean(), "div_final": g.div_final.mean(),
        "pct_colapso": g.colapso.mean() * 100,
    }).reindex(config.MODELOS).rename_axis("modelo").reset_index(), "genetico_corridas.csv")

    halving = joblib.load(CACHE / "halving.joblib")
    h = halving.groupby("metodo").agg(
        candidatos=("candidatos", "mean"), arboles=("arboles", "mean"), t=("t", "mean"),
        auc_interna=("auc_interna", "mean"), auc_externa=("auc_externa", "mean"),
        auc_externa_de=("auc_externa", "std")).reset_index()
    guardar(h, "halving.csv")
    d = halving.pivot(index="pliegue", columns="metodo", values="auc_externa")
    escalares["halving_p"] = float(stats.wilcoxon(d["Successive Halving"] - d["Random Search"]).pvalue)

    # -----------------------------------------------------------------------
    # 2.5 Modelos finales
    # -----------------------------------------------------------------------
    finales = joblib.load(CACHE / "modelos_finales.joblib")
    seleccion = (resumen.loc[resumen.groupby("modelo")["auc_media"].idxmax()]
                 .set_index("modelo").loc[config.MODELOS])
    guardar(pd.DataFrame([{
        "modelo": m, "balanceo": f["balanceo"], "metodo": f["metodo"],
        "auc_media": seleccion.loc[m, "auc_media"], "auc_de": seleccion.loc[m, "auc_de"],
        "hiperparametros": json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in f["params"].items()}),
    } for m, f in finales.items()]), "finales.csv")
    mejor = seleccion["auc_media"].idxmax()
    escalares["mejor_familia"] = mejor

    # -----------------------------------------------------------------------
    # 2.6 Optimización computacional
    # -----------------------------------------------------------------------
    esc = joblib.load(CACHE / "escalamiento.joblib")
    guardar(esc, "escalamiento.csv")
    filas = []
    for m in config.MODELOS:
        for eje in ["n", "p"]:
            dd = esc[(esc.modelo == m) & (esc.eje == eje)]
            b = np.polyfit(np.log(dd.valor.iloc[-4:]), np.log(dd.t.iloc[-4:]), 1)[0]
            filas.append({"modelo": m, "eje": eje, "pendiente": float(b)})
    guardar(pd.DataFrame(filas), "escalamiento_pendientes.csv")
    guardar(joblib.load(CACHE / "xgb_metodos.joblib"), "xgb_metodos.csv")
    es = joblib.load(CACHE / "early_stopping.joblib")
    guardar(es.groupby("variante").agg(arboles=("arboles", "mean"), tiempo_s=("t", "mean"),
                                       auc_media=("auc", "mean"), auc_de=("auc", "std"))
            .reset_index(), "early_stopping.csv")
    guardar(joblib.load(CACHE / "paralelo.joblib"), "paralelo.csv")
    est = joblib.load(CACHE / "estandar_vs_optimizado.joblib")
    est = est.rename(columns={"Entrenamiento (ms)": "entrenamiento_ms",
                              "Inferencia 168 pac. (ms)": "inferencia_ms",
                              "Memoria pico (MB)": "memoria_mb",
                              "Tamaño modelo (KB)": "tamano_kb", "AUC CV": "auc_cv",
                              "_auc": "auc_media", "Modelo": "modelo", "Versión": "version"})
    guardar(est, "estandar_vs_optimizado.csv")

    # -----------------------------------------------------------------------
    # 2.7 Evaluación en prueba
    # -----------------------------------------------------------------------
    pred = pd.read_csv(R / "predicciones_test.csv")
    assert (pred["grade"].to_numpy() == yt).all(), "predicciones_test.csv no coincide con la partición"
    proba_test = {m: pred[f"proba_{m}"].to_numpy() for m in config.MODELOS}
    guardar(pred, "predicciones_test.csv")
    guardar(pd.read_csv(R / "metricas_test.csv"), "metricas_test.csv")
    filas = []
    for m in config.MODELOS:
        fpr, tpr, _ = roc_curve(yt, proba_test[m])
        filas += [{"modelo": m, "fpr": a, "tpr": b} for a, b in zip(fpr, tpr)]
    guardar(pd.DataFrame(filas), "roc_test.csv")

    # -----------------------------------------------------------------------
    # 2.8 Calibración
    # -----------------------------------------------------------------------
    oof_sel = {m: oof[id_combo(m, f["balanceo"], f["metodo"])].to_numpy()
               for m, f in finales.items()}
    filas, filas_m = [], []
    for fuente, y_, probas, bins in [("oof", y_train.to_numpy(), oof_sel, 10),
                                     ("test", yt, proba_test, 5)]:
        for m in config.MODELOS:
            c = ev.curva_confiabilidad(y_, probas[m], bins)
            c.insert(0, "modelo", m)
            c.insert(0, "fuente", fuente)
            filas.append(c)
    guardar(pd.concat(filas, ignore_index=True), "calibracion_curvas.csv")
    guardar(pd.DataFrame({
        "modelo": config.MODELOS,
        "brier_oof": [metricas(y_train, oof_sel[m])["brier"] for m in config.MODELOS],
        "ece_oof": [ev.ece(y_train, oof_sel[m]) for m in config.MODELOS],
        "brier_test": [metricas(yt, proba_test[m])["brier"] for m in config.MODELOS],
        "ece_test": [ev.ece(yt, proba_test[m], 5) for m in config.MODELOS],
    }), "calibracion_metricas.csv")
    guardar(pd.read_csv(R / "recalibracion.csv"), "recalibracion.csv")

    # -----------------------------------------------------------------------
    # 2.9 SHAP y LIME
    # -----------------------------------------------------------------------
    nombres = config.PREDICTORAS
    X_todos = pd.concat([X_train, X_test]).reset_index(drop=True)
    shap_x = pd.read_csv(R / "shap_xgboost.csv")
    base_xgb = float(shap_x["base"].iloc[0])
    sv_xgb = shap_x[nombres].to_numpy()

    pipe_rl = finales["logistica"]["pipeline"]
    escala = pipe_rl.named_steps["escala"]
    beta = pipe_rl.named_steps["modelo"].coef_[0]
    media = escala.transform(X_train).mean(0)
    sv_rl = (escala.transform(X_todos) - media) * beta
    base_rl = float(pipe_rl.named_steps["modelo"].intercept_[0] + beta @ media)

    guardar(X_todos.assign(conjunto=["train"] * len(X_train) + ["test"] * len(X_test)),
            "shap_valores_x.csv")
    guardar(pd.DataFrame(sv_xgb, columns=nombres), "shap_xgboost.csv")
    guardar(pd.DataFrame(sv_rl, columns=nombres), "shap_logistica.csv")
    guardar(pd.DataFrame({"variable": nombres, "coeficiente": beta}), "coeficientes_logistica.csv")
    escalares.update(base_xgb=base_xgb, base_rl=base_rl)
    guardar(pd.read_csv(R / "importancia_shap.csv"), "importancia_shap.csv")

    p_xgb = proba_test["xgboost"]
    error = np.abs(yt - p_xgb)
    i_bien = int(np.argmin(np.where(yt == 1, error, np.inf)))
    i_mal = int(np.argmax(error))
    sv_test = sv_xgb[len(X_train):]
    pesos_lime = joblib.load(CACHE / "lime.joblib")
    filas, comparacion = [], []
    for i, rotulo in [(i_bien, "bien clasificado"), (i_mal, "peor error")]:
        paciente = int(X_test.index[i])
        lime_i, lime_de = pesos_lime[i].mean(0), pesos_lime[i].std(0)
        shap_i = sv_test[i]
        for j, v in enumerate(nombres):
            filas.append({"paciente": paciente, "rotulo": rotulo, "real": int(yt[i]),
                          "proba": float(p_xgb[i]), "variable": v,
                          "valor": float(X_test.iloc[i][v]), "shap": float(shap_i[j]),
                          "lime_media": float(lime_i[j]), "lime_de": float(lime_de[j])})
        top = np.argsort(np.abs(shap_i))[::-1][:8]
        orden_lime = np.argsort(np.abs(lime_i))[::-1]
        comparacion.append({
            "paciente": paciente, "rotulo": rotulo,
            "spearman": float(stats.spearmanr(np.abs(shap_i), np.abs(lime_i)).statistic),
            "signos": float(np.mean(np.sign(shap_i[top]) == np.sign(lime_i[top]))),
            "top3_shap": ", ".join(nombres[j] for j in np.argsort(np.abs(shap_i))[::-1][:3]),
            "top3_lime": ", ".join(nombres[j] for j in orden_lime[:3]),
            "cv_lime_top5": float(np.mean(lime_de[orden_lime[:5]] / np.abs(lime_i[orden_lime[:5]]))),
        })
    guardar(pd.DataFrame(filas), "explicaciones_locales.csv")
    guardar(pd.DataFrame(comparacion), "lime_comparacion.csv")

    # -----------------------------------------------------------------------
    # 2.10 Robustez
    # -----------------------------------------------------------------------
    escalares.update(de_dentro=float(resumen["auc_de"].mean()),
                     de_entre=float(resumen["auc_media"].std(ddof=1)),
                     rango_medias=float(resumen["auc_media"].max() - resumen["auc_media"].min()))
    semillas = joblib.load(CACHE / "semillas.joblib")
    guardar(semillas.groupby("modelo").agg(
        auc_interna_media=("auc_interna", "mean"), auc_interna_de=("auc_interna", "std"),
        auc_test_media=("auc_test", "mean"), auc_test_de=("auc_test", "std"),
        auc_test_min=("auc_test", "min"), auc_test_max=("auc_test", "max"),
    ).reindex(config.MODELOS).reset_index(), "semillas.csv")

    # -----------------------------------------------------------------------
    # 2.11 Comparación estadística
    # -----------------------------------------------------------------------
    matriz = pliegues.pivot(index="pliegue", columns="combo", values="auc")
    fr = ev.friedman(matriz)
    escalares.update(friedman_chi2=fr["chi2"], friedman_gl=int(fr["gl"]), friedman_p=fr["p"],
                     iman_f=fr["F_ID"], iman_p=fr["p_ID"],
                     cd48=ev.diferencia_critica(fr["k"], fr["N"]))
    guardar(fr["rangos"].rename_axis("combo").reset_index(name="rango"), "rangos48.csv")

    factores = {"modelo": config.MODELOS, "balanceo": config.BALANCEOS,
                "metodo": config.OPTIMIZADORES}
    filas_f, filas_r, filas_n = [], [], []
    for factor, niveles in factores.items():
        otros = [f for f in factores if f != factor] + ["pliegue"]
        mat = pliegues.pivot_table(index=otros, columns=factor, values="auc")[niveles]
        r = ev.friedman(mat)
        cd = ev.diferencia_critica(r["k"], r["N"])
        filas_f.append({"factor": factor, "bloques": r["N"], "k": r["k"], "chi2": r["chi2"],
                        "gl": r["gl"], "p": r["p"], "cd": cd})
        filas_r += [{"factor": factor, "nivel": i, "rango": v} for i, v in r["rangos"].items()]
        nem = ev.nemenyi(r["rangos"], r["N"])
        filas_n += [{"factor": factor, "nivel_a": a, "nivel_b": b, "p": nem.loc[a, b]}
                    for a in nem.index for b in nem.columns]
    guardar(pd.DataFrame(filas_f), "friedman_factores.csv")
    guardar(pd.DataFrame(filas_r), "rangos_factores.csv")
    guardar(pd.DataFrame(filas_n), "nemenyi_factores.csv")

    def dif_auc(y_, a, b):
        return roc_auc_score(y_, a) - roc_auc_score(y_, b)

    pares = [("logistica", "random_forest"), ("logistica", "xgboost"),
             ("random_forest", "xgboost")]
    filas = []
    for a, b in pares:
        dl = ev.delong(yt, proba_test[a], proba_test[b])
        lo, hi = ev.ic_bootstrap(dif_auc, yt, proba_test[a], proba_test[b])
        filas.append({"modelo_a": a, "modelo_b": b, "diferencia": dl["diferencia"],
                      "ic_inf": lo, "ic_sup": hi, "z": dl["z"], "p": dl["p"]})
    delong_tab = pd.DataFrame(filas)
    delong_tab["p_holm"] = ev.holm(delong_tab["p"])
    guardar(delong_tab, "delong.csv")

    filas = []
    for m in config.MODELOS:
        for met, f_ in [("AUC", roc_auc_score),
                        ("Recall (GBM)", lambda y_, p: np.mean(p[y_ == 1] >= 0.5))]:
            lo, hi = ev.ic_bootstrap(f_, yt, proba_test[m])
            filas.append({"modelo": m, "metrica": met, "estimacion": f_(yt, proba_test[m]),
                          "ic_inf": lo, "ic_sup": hi})
    guardar(pd.DataFrame(filas), "ic_test.csv")

    sel_auc = {m: pliegues.loc[pliegues.combo == id_combo(m, f["balanceo"], f["metodo"]),
                               "auc"].to_numpy() for m, f in finales.items()}
    filas = []
    for a, b in pares:
        d1 = ev.delta_cliff(sel_auc[a], sel_auc[b])
        d2 = ev.delta_cliff(pliegues.loc[pliegues.modelo == a, "auc"],
                            pliegues.loc[pliegues.modelo == b, "auc"])
        filas.append({"comparacion": f"{N[a]} vs {N[b]}", "delta_sel": d1,
                      "magnitud_sel": ev.magnitud_cliff(d1), "delta_familia": d2,
                      "magnitud_familia": ev.magnitud_cliff(d2)})
    for o in ["grid", "bayesiano", "genetico"]:
        d2 = ev.delta_cliff(pliegues.loc[pliegues.metodo == o, "auc"],
                            pliegues.loc[pliegues.metodo == "random", "auc"])
        filas.append({"comparacion": f"{N[o]} vs Random Search", "delta_sel": np.nan,
                      "magnitud_sel": "", "delta_familia": d2,
                      "magnitud_familia": ev.magnitud_cliff(d2)})
    guardar(pd.DataFrame(filas), "cliff.csv")

    # -----------------------------------------------------------------------
    # 2.12 Panel reducido
    # -----------------------------------------------------------------------
    panel, rankings = joblib.load(CACHE / "panel_reducido.joblib")
    pc = panel.groupby("k").agg(auc=("auc", "mean"), de=("auc", "std"),
                                recall=("recall", "mean"))
    completo = pc.loc[23, "auc"]
    k_min = int(pc.index[(pc.auc >= completo - pc.loc[23, "de"]
                          / np.sqrt(config.K_EXTERNO))].min())
    escalares["k_min"] = k_min
    guardar(pc.reset_index(), "panel_reducido.csv")
    frecuencia = pd.Series([v for r in rankings for v in r[:k_min]]).value_counts()
    guardar(pd.DataFrame({
        "variable": frecuencia.index, "pliegues": frecuencia.to_numpy(),
        "posicion_media": [np.mean([r.index(v) + 1 for r in rankings]) for v in frecuencia.index],
    }), "panel_frecuencia.csv")

    (destino / "escalares.json").write_text(
        json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in escalares.items()},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    archivos.append("escalares.json")
    return archivos


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--destino", type=Path, default=R / "para_r",
                        help="Carpeta de salida (por defecto resultados/para_r).")
    args = parser.parse_args()
    archivos = exportar(args.destino)
    print(f"{len(archivos)} archivos escritos en {args.destino.resolve()}")


if __name__ == "__main__":
    main()
