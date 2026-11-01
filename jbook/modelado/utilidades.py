"""Utilidades pequeñas compartidas por el notebook."""

from __future__ import annotations

import joblib

from . import config


def en_cache(nombre: str, funcion, recalcular: bool = False):
    """Devuelve el resultado guardado en ``resultados/cache/<nombre>.joblib``
    o lo calcula con ``funcion()`` y lo guarda.

    Sirve para que el libro se pueda reejecutar en minutos: los cálculos
    costosos (búsquedas, escalamiento, semillas) se hacen una sola vez y
    ``recalcular=True`` fuerza a repetirlos.

    Parameters
    ----------
    nombre : str
        Nombre del archivo, sin extensión.
    funcion : callable
        Función sin argumentos que produce el resultado.
    recalcular : bool
        Ignora el archivo guardado y vuelve a calcular.
    """
    ruta = config.DIR_RESULTADOS / "cache" / f"{nombre}.joblib"
    if ruta.exists() and not recalcular:
        return joblib.load(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    resultado = funcion()
    joblib.dump(resultado, ruta)
    return resultado
