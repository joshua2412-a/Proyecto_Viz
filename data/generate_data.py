"""
Generación del dataset sintético de rotación laboral (employee attrition).

Los datos se construyen con un proceso generador explícito: primero se simulan
las variables del empleado con distribuciones plausibles y dependencias entre
ellas (la antigüedad depende de la edad, el salario del departamento y la
experiencia...), y después se calcula la probabilidad de abandono con un modelo
logístico "verdadero". Así el dataset tiene una estructura realista y aprendible
en lugar de ruido puro.

Uso:
    python data/generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permite ejecutar el archivo directamente (python data/generate_data.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import (  # noqa: E402  (import tras ajustar sys.path)
    DATA_PATH,
    DEPARTAMENTOS,
    N_EMPLEADOS,
    RANDOM_STATE,
)

# --------------------------------------------------------------------------- #
# Parámetros del proceso generador
# --------------------------------------------------------------------------- #
# Peso de cada departamento en la plantilla
PESOS_DEPARTAMENTO = {
    "Ventas": 0.22,
    "Tecnología": 0.20,
    "Operaciones": 0.18,
    "Soporte": 0.14,
    "Marketing": 0.10,
    "Finanzas": 0.09,
    "Recursos Humanos": 0.07,
}

# Salario base mensual por departamento (COP)
SALARIO_BASE = {
    "Ventas": 3_800_000,
    "Tecnología": 6_200_000,
    "Operaciones": 3_400_000,
    "Soporte": 3_000_000,
    "Marketing": 4_300_000,
    "Finanzas": 5_200_000,
    "Recursos Humanos": 4_000_000,
}

# Carga de trabajo típica (horas semanales) por departamento
HORAS_BASE = {
    "Ventas": 48,
    "Tecnología": 46,
    "Operaciones": 47,
    "Soporte": 45,
    "Marketing": 44,
    "Finanzas": 43,
    "Recursos Humanos": 42,
}

# Efecto propio del departamento sobre el log-odds de abandono
EFECTO_DEPARTAMENTO = {
    "Ventas": 0.45,
    "Soporte": 0.35,
    "Operaciones": 0.10,
    "Tecnología": 0.00,
    "Marketing": -0.10,
    "Recursos Humanos": -0.25,
    "Finanzas": -0.35,
}

# Coeficientes del modelo logístico "verdadero" que decide el abandono
INTERCEPTO = -1.55
BETA = {
    "satisfaccion": -0.62,      # menos satisfacción -> más abandono
    "horas_trabajadas": 0.055,  # sobrecarga -> más abandono
    "anios_empresa": -0.075,    # más antigüedad -> más arraigo
    "promociones": -0.50,       # reconocimiento -> menos abandono
    "salario_z": -0.55,         # salario por debajo del mercado -> más abandono
    "edad": -0.02,              # plantilla más joven rota más
}


def _sigmoide(z: np.ndarray) -> np.ndarray:
    """Función logística estable para vectores de log-odds."""
    return 1.0 / (1.0 + np.exp(-z))


def generar_datos(n: int = N_EMPLEADOS, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Construye el DataFrame sintético de empleados.

    Parámetros
    ----------
    n : número de empleados a simular.
    random_state : semilla para que el dataset sea reproducible.
    """
    rng = np.random.default_rng(random_state)

    # --- Departamento --------------------------------------------------- #
    pesos = np.array([PESOS_DEPARTAMENTO[d] for d in DEPARTAMENTOS])
    pesos = pesos / pesos.sum()
    departamento = rng.choice(DEPARTAMENTOS, size=n, p=pesos)

    # --- Edad ----------------------------------------------------------- #
    edad = np.clip(rng.normal(36, 9, n), 21, 60).round().astype(int)

    # --- Antigüedad (acotada por la edad: nadie entra antes de los 20) --- #
    max_antiguedad = np.clip(edad - 20, 0, 35)
    anios_empresa = np.minimum(rng.exponential(5.5, n), max_antiguedad)
    anios_empresa = np.clip(anios_empresa, 0, 35).round().astype(int)

    # --- Salario: base del departamento + experiencia + ruido ----------- #
    base = np.array([SALARIO_BASE[d] for d in departamento])
    salario = (
        base
        + anios_empresa * 145_000
        + (edad - 36) * 45_000
        + rng.normal(0, 550_000, n)
    )
    salario = np.clip(salario, 1_500_000, 18_000_000).round(-4).astype(int)

    # --- Horas trabajadas por semana ------------------------------------ #
    horas_base = np.array([HORAS_BASE[d] for d in departamento])
    horas_trabajadas = np.clip(horas_base + rng.normal(0, 4.5, n), 35, 70)
    horas_trabajadas = horas_trabajadas.round().astype(int)

    # --- Promociones: crecen con la antigüedad (proceso de Poisson) ----- #
    promociones = rng.poisson(np.clip(anios_empresa / 4.0, 0.05, 4.0))
    promociones = np.clip(promociones, 0, 6).astype(int)

    # --- Satisfacción: penalizada por sobrecarga, elevada por promociones #
    satisfaccion = (
        3.4
        - 0.055 * (horas_trabajadas - 45)
        + 0.22 * promociones
        + rng.normal(0, 0.75, n)
    )
    satisfaccion = np.clip(satisfaccion, 1.0, 5.0).round(1)

    # --- Abandono: modelo logístico verdadero --------------------------- #
    salario_z = (salario - salario.mean()) / salario.std()
    efecto_depto = np.array([EFECTO_DEPARTAMENTO[d] for d in departamento])

    log_odds = (
        INTERCEPTO
        + BETA["satisfaccion"] * (satisfaccion - 3.0)
        + BETA["horas_trabajadas"] * (horas_trabajadas - 45)
        + BETA["anios_empresa"] * (anios_empresa - 6)
        + BETA["promociones"] * promociones
        + BETA["salario_z"] * salario_z
        + BETA["edad"] * (edad - 36)
        + efecto_depto
    )
    probabilidad = _sigmoide(log_odds)
    abandono = rng.binomial(1, probabilidad)

    df = pd.DataFrame(
        {
            "edad": edad,
            "salario": salario,
            "anios_empresa": anios_empresa,
            "departamento": departamento,
            "satisfaccion": satisfaccion,
            "horas_trabajadas": horas_trabajadas,
            "promociones": promociones,
            "abandono": abandono,
        }
    )
    return df


def guardar_datos(df: pd.DataFrame, ruta: Path = DATA_PATH) -> Path:
    """Escribe el dataset en CSV creando la carpeta si hace falta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    return ruta


def main() -> None:
    """Punto de entrada por línea de comandos."""
    df = generar_datos()
    ruta = guardar_datos(df)

    tasa = df["abandono"].mean()
    print("Dataset sintético de rotación laboral generado")
    print(f"  Archivo            : {ruta}")
    print(f"  Registros          : {len(df):,}")
    print(f"  Variables          : {df.shape[1]}")
    print(f"  Tasa de abandono   : {tasa:.1%}")
    print("\nAbandono por departamento:")
    resumen = (
        df.groupby("departamento")["abandono"].agg(["size", "mean"]).sort_values("mean", ascending=False)
    )
    for depto, fila in resumen.iterrows():
        print(f"  {depto:<18} {int(fila['size']):>5} empleados   {fila['mean']:.1%}")


if __name__ == "__main__":
    main()
