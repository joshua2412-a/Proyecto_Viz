"""
Configuración central del proyecto.

Aquí viven las rutas, los nombres de columnas y las etiquetas legibles.
Todos los módulos (datos, modelo y pestañas) importan desde aquí para que
no haya rutas ni strings "mágicos" repartidos por el código.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto (siempre absolutas, sin importar desde dónde se ejecute)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
ASSETS_DIR = BASE_DIR / "assets"

DATA_PATH = DATA_DIR / "employee_attrition.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Parámetros de generación de datos y entrenamiento
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_EMPLEADOS = 2000
TEST_SIZE = 0.2

DEPARTAMENTOS = [
    "Ventas",
    "Tecnología",
    "Operaciones",
    "Soporte",
    "Marketing",
    "Finanzas",
    "Recursos Humanos",
]

# --------------------------------------------------------------------------- #
# Esquema del dataset
# --------------------------------------------------------------------------- #
TARGET = "abandono"

NUMERIC_FEATURES = [
    "edad",
    "salario",
    "anios_empresa",
    "satisfaccion",
    "horas_trabajadas",
    "promociones",
]

CATEGORICAL_FEATURES = ["departamento"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Etiquetas legibles para títulos de ejes, tablas y formularios
LABELS = {
    "edad": "Edad (años)",
    "salario": "Salario mensual (COP)",
    "anios_empresa": "Antigüedad (años)",
    "departamento": "Departamento",
    "satisfaccion": "Satisfacción laboral (1-5)",
    "horas_trabajadas": "Horas trabajadas / semana",
    "promociones": "Promociones recibidas",
    "abandono": "Abandono",
}

# Valores por defecto del formulario de predicción (perfil "promedio")
DEFAULT_INPUT = {
    "edad": 35,
    "salario": 4_500_000,
    "anios_empresa": 5,
    "departamento": "Ventas",
    "satisfaccion": 3.0,
    "horas_trabajadas": 45,
    "promociones": 1,
}

# Rangos válidos del formulario (mínimo, máximo, paso)
INPUT_RANGES = {
    "edad": (21, 60, 1),
    "salario": (1_500_000, 15_000_000, 100_000),
    "anios_empresa": (0, 35, 1),
    "satisfaccion": (1.0, 5.0, 0.1),
    "horas_trabajadas": (35, 70, 1),
    "promociones": (0, 6, 1),
}

# Umbrales de riesgo usados en la pestaña de predicción
RIESGO_MEDIO = 0.35
RIESGO_ALTO = 0.60
