"""
Formato numérico en español.

El tablero mezclaba dos convenciones: el texto escrito a mano decía "p < 0,05"
con coma, y cualquier cifra calculada salía como "42.0%" con punto, porque es
lo que hace Python por defecto. Quedaban las dos en el mismo párrafo.

Este módulo es el único sitio donde se decide cómo se escribe un número. La
convención es la española: coma decimal, punto para los miles y espacio antes
del signo de porcentaje.

Las figuras de Plotly no pasan por aquí: se arreglan de una vez con el
parámetro `separators` de la plantilla (ver utils/theme.py).
"""

from __future__ import annotations

import math


def _es_nulo(valor) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def num(valor, decimales: int = 1, signo: bool = False) -> str:
    """Un número con coma decimal. `signo=True` fuerza el + en los positivos."""
    if _es_nulo(valor):
        return "—"
    texto = f"{valor:+.{decimales}f}" if signo else f"{valor:.{decimales}f}"
    return texto.replace(".", ",")


def miles(valor) -> str:
    """Un entero con punto de millar: 21042 -> 21.042."""
    if _es_nulo(valor):
        return "—"
    return f"{int(round(valor)):,}".replace(",", ".")


def pct(valor, decimales: int = 1, signo: bool = False) -> str:
    """Un porcentaje: 42.04 -> "42,0 %". El espacio antes del signo es el correcto."""
    if _es_nulo(valor):
        return "—"
    return f"{num(valor, decimales, signo)} %"


def p_valor(valor, umbral: float = 0.0001) -> str:
    """Un p-valor como se escribe en todo el tablero."""
    if _es_nulo(valor):
        return "—"
    return "p < 0,0001" if valor < umbral else f"p = {num(valor, 4)}"
