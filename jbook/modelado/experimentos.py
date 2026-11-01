"""Ejecuta las 48 combinaciones modelo × balanceo × optimizador.

Uso (desde la raíz del proyecto o desde ``jbook/``)::

    python -m modelado.experimentos            # dentro de jbook/

Cada combinación se guarda en ``resultados/parciales/<id>.json`` en cuanto
termina, así que una ejecución interrumpida se reanuda donde quedó. Al final
se consolidan las tablas maestras:

* ``resultados/experimentos.csv``: una fila por combinación (media y
  desviación de cada métrica en el bucle externo, tiempos, semilla).
* ``resultados/pliegues.csv``: una fila por combinación y pliegue externo,
  con los hiperparámetros elegidos en ese pliegue.
* ``resultados/historiales.json``: la AUC interna de cada evaluación, para
  las curvas *anytime*.
* ``resultados/geneticos.json``: evolución por generación del algoritmo
  genético (mejor, media y diversidad).
* ``resultados/oof.csv``: probabilidades fuera de pliegue por combinación.
"""

from __future__ import annotations

import itertools
import json
import platform
import time

import numpy as np
import pandas as pd

from . import config, datos, optimizacion


def id_combo(modelo: str, balanceo: str, metodo: str) -> str:
    """Identificador estable de una combinación."""
    return f"{modelo}|{balanceo}|{metodo}"


def combinaciones():
    """Las 3 × 4 × 4 = 48 combinaciones en orden modelo → balanceo → optimizador."""
    return list(itertools.product(config.MODELOS, config.BALANCEOS, config.OPTIMIZADORES))


def correr(n_jobs: int = 2, solo=None, verbose: bool = True):
    """Corre (o reanuda) el experimento completo y consolida las tablas.

    Parameters
    ----------
    n_jobs : int
        Procesos para paralelizar los pliegues externos.
    solo : list of tuple, optional
        Subconjunto de combinaciones ``(modelo, balanceo, metodo)``.
    """
    X_train, _, y_train, _ = datos.particion()
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    dir_parc = config.DIR_RESULTADOS / "parciales"
    dir_parc.mkdir(parents=True, exist_ok=True)

    for modelo, balanceo, metodo in (solo or combinaciones()):
        cid = id_combo(modelo, balanceo, metodo)
        archivo = dir_parc / (cid.replace("|", "__") + ".json")
        if archivo.exists():
            continue
        t0 = time.perf_counter()
        filas, hist, extras, oof = optimizacion.validacion_anidada(
            metodo, modelo, balanceo, X_train, y_train, n_jobs=n_jobs)
        total = time.perf_counter() - t0
        archivo.write_text(json.dumps({
            "modelo": modelo, "balanceo": balanceo, "metodo": metodo,
            "semilla": config.SEED, "t_total": total, "filas": filas,
            "historiales": hist, "extras": extras, "oof": oof.tolist(),
        }))
        if verbose:
            auc = np.mean([f["auc"] for f in filas])
            print(f"{cid:<40} AUC externa {auc:.4f}   {total:6.1f} s", flush=True)
    return consolidar()


def consolidar():
    """Une los parciales en las tablas maestras y las devuelve."""
    dir_parc = config.DIR_RESULTADOS / "parciales"
    resumen, pliegues, historiales, geneticos, oof = [], [], {}, {}, {}
    for modelo, balanceo, metodo in combinaciones():
        cid = id_combo(modelo, balanceo, metodo)
        archivo = dir_parc / (cid.replace("|", "__") + ".json")
        if not archivo.exists():
            continue
        r = json.loads(archivo.read_text())
        df = pd.DataFrame(r["filas"])
        for f in r["filas"]:
            pliegues.append({"combo": cid, "modelo": modelo, "balanceo": balanceo,
                             "metodo": metodo, **f})
        fila = {"combo": cid, "modelo": modelo, "balanceo": balanceo, "metodo": metodo}
        for m in ["auc", "accuracy", "precision", "recall", "f1", "brier", "auc_interna"]:
            fila[f"{m}_media"] = df[m].mean()
            fila[f"{m}_de"] = df[m].std(ddof=1)
        fila.update({
            "n_eval_total": int(df["n_eval"].sum()),
            "t_total_s": r["t_total"],
            "t_busqueda_s": df["t_busqueda"].sum(),
            "t_por_eval_s": df["t_por_eval"].mean(),
            "t_prediccion_ms": df["t_prediccion"].mean() * 1000,
            "semilla": r["semilla"],
        })
        resumen.append(fila)
        historiales[cid] = r["historiales"]
        if metodo == "genetico":
            geneticos[cid] = [e.get("generaciones", []) for e in r["extras"]]
        oof[cid] = r["oof"]

    d = config.DIR_RESULTADOS
    resumen = pd.DataFrame(resumen)
    pliegues = pd.DataFrame(pliegues)
    resumen.to_csv(d / "experimentos.csv", index=False)
    pliegues.to_csv(d / "pliegues.csv", index=False)
    (d / "historiales.json").write_text(json.dumps(historiales))
    (d / "geneticos.json").write_text(json.dumps(geneticos))
    pd.DataFrame(oof).to_csv(d / "oof.csv", index=False)
    (d / "entorno.json").write_text(json.dumps(entorno(), indent=2))
    return resumen, pliegues


def entorno() -> dict:
    """Versiones de Python y de las librerías usadas (para reproducibilidad)."""
    import imblearn
    import optuna
    import scipy
    import sklearn
    import xgboost
    import deap
    return {"python": platform.python_version(), "plataforma": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__,
            "xgboost": xgboost.__version__, "optuna": optuna.__version__,
            "deap": deap.__version__, "semilla": config.SEED,
            "k_externo": config.K_EXTERNO, "k_interno": config.K_INTERNO,
            "presupuesto": config.PRESUPUESTO}


if __name__ == "__main__":
    correr()
