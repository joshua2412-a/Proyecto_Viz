"""
Carga del dataset y estadística descriptiva.

Este módulo es la ÚNICA puerta de entrada a los datos. El alcance del proyecto
es el análisis exploratorio: aquí no se entrena ni se carga ningún modelo.

La partición train/test se reconstruye aquí con los mismos parámetros del
notebook (80/20, estratificada, `random_state=42`), de modo que las cifras del
dashboard coinciden exactamente con las del Jupyter Book: el EDA se describe
sobre el conjunto de entrenamiento y el de prueba queda reservado.

Las funciones usan `lru_cache` para que el CSV se lea una sola vez por proceso,
aunque diez pestañas lo pidan.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from utils.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    GENDER_LABELS,
    GENE_FEATURES,
    GRADE_LABELS,
    NUMERIC_FEATURES,
    RACE_LABELS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

# Ámbitos de datos admitidos por las funciones descriptivas
AMBITOS = ("train", "test", "full")
AMBITO_NOMBRES = {
    "train": "conjunto de entrenamiento",
    "test": "conjunto de prueba",
    "full": "dataset completo",
}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Lee TCGA_InfoWithGrade.csv y añade las columnas de etiquetas legibles.

    Valida el esquema antes de devolver nada: si el CSV no trae exactamente las
    columnas esperadas, es mejor un error claro aquí que un gráfico vacío tres
    pestañas más adelante.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}.\n"
            "Coloca el archivo TCGA_InfoWithGrade.csv en la carpeta dataset/ "
            "(ver dataset/README.md)."
        )

    df = pd.read_csv(DATA_PATH)

    faltantes = [columna for columna in [TARGET, *FEATURES] if columna not in df.columns]
    if faltantes:
        raise ValueError(
            f"El dataset {DATA_PATH.name} no tiene las columnas esperadas. "
            f"Faltan: {', '.join(faltantes)}"
        )

    df["grade_label"] = df[TARGET].map(GRADE_LABELS)
    df["gender_label"] = df["Gender"].map(GENDER_LABELS)
    df["race_label"] = df["Race"].map(RACE_LABELS)
    return df


@lru_cache(maxsize=4)
def _indices_particion() -> dict[str, tuple[int, ...]]:
    """Índices de la partición 80/20 estratificada (idéntica a la del libro)."""
    df = load_data()
    entrenamiento, prueba = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    return {
        "train": tuple(entrenamiento),
        "test": tuple(prueba),
        "full": tuple(df.index),
    }


def get_dataframe(ambito: str = "train") -> pd.DataFrame:
    """Copia defensiva del dataset en el ámbito pedido.

    `ambito`: "train" (671 pacientes, el que describe el EDA del libro),
    "test" (168) o "full" (839).
    """
    if ambito not in AMBITOS:
        raise ValueError(f"Ámbito no válido: {ambito!r}. Usa uno de {AMBITOS}.")
    indices = list(_indices_particion()[ambito])
    return load_data().loc[indices].copy()


def tamanos_particion() -> dict[str, int]:
    """Número de pacientes en cada ámbito."""
    return {clave: len(valor) for clave, valor in _indices_particion().items()}


# --------------------------------------------------------------------------- #
# Descriptivos de la variable objetivo y de las clínicas
# --------------------------------------------------------------------------- #
def proporcion_grado(ambito: str = "train") -> pd.DataFrame:
    """Frecuencia y porcentaje de LGG y GBM."""
    df = get_dataframe(ambito)
    conteo = df["grade_label"].value_counts().reindex(["LGG", "GBM"]).fillna(0)
    return pd.DataFrame(
        {
            "grado": conteo.index,
            "pacientes": conteo.values.astype(int),
            "porcentaje": (conteo.values / conteo.values.sum() * 100),
        }
    )


def estadisticas_edad(ambito: str = "train") -> pd.DataFrame:
    """Descriptivos de la edad al diagnóstico por grado tumoral."""
    df = get_dataframe(ambito)
    resumen = (
        df.groupby("grade_label")["Age_at_diagnosis"]
        .agg(
            pacientes="size",
            media="mean",
            desviacion="std",
            mediana="median",
            q1=lambda serie: serie.quantile(0.25),
            q3=lambda serie: serie.quantile(0.75),
            minimo="min",
            maximo="max",
        )
        .reindex(["LGG", "GBM"])
        .reset_index()
    )
    resumen["iqr"] = resumen["q3"] - resumen["q1"]
    return resumen


def prueba_edad_por_grado(ambito: str = "train") -> dict:
    """Prueba U de Mann-Whitney para la edad entre LGG y GBM.

    Se usa una prueba no paramétrica porque Shapiro-Wilk rechaza la normalidad
    de `Age_at_diagnosis` (distribución bimodal), igual que en el notebook.
    """
    df = get_dataframe(ambito)
    lgg = df.loc[df[TARGET] == 0, "Age_at_diagnosis"]
    gbm = df.loc[df[TARGET] == 1, "Age_at_diagnosis"]
    estadistico, p_valor = stats.mannwhitneyu(lgg, gbm, alternative="two-sided")
    return {
        "estadistico": float(estadistico),
        "p_valor": float(p_valor),
        "n_lgg": int(len(lgg)),
        "n_gbm": int(len(gbm)),
    }


def distribucion_clinica(variable: str, ambito: str = "train") -> pd.DataFrame:
    """Tabla de contingencia (porcentaje dentro de cada grado) de Gender o Race."""
    if variable not in CATEGORICAL_FEATURES:
        raise ValueError(f"{variable!r} no es una variable clínica categórica.")

    columna_etiqueta = {"Gender": "gender_label", "Race": "race_label"}[variable]
    df = get_dataframe(ambito)

    tabla = (
        df.groupby(["grade_label", columna_etiqueta], observed=True)
        .size()
        .rename("pacientes")
        .reset_index()
    )
    totales = tabla.groupby("grade_label", observed=True)["pacientes"].transform("sum")
    tabla["porcentaje"] = tabla["pacientes"] / totales * 100
    tabla = tabla.rename(columns={columna_etiqueta: "categoria"})
    return tabla


# --------------------------------------------------------------------------- #
# Descriptivos de las mutaciones
# --------------------------------------------------------------------------- #
def prevalencia_genes(ambito: str = "train") -> pd.DataFrame:
    """Prevalencia de mutación de cada gen, global y por grado tumoral.

    La columna `diferencia` (prevalencia en GBM menos prevalencia en LGG) es la
    que ordena el gráfico: resume de un tirón hacia qué grado empuja cada gen.
    """
    df = get_dataframe(ambito)
    filas = []
    for gen in GENE_FEATURES:
        prevalencia_lgg = df.loc[df[TARGET] == 0, gen].mean() * 100
        prevalencia_gbm = df.loc[df[TARGET] == 1, gen].mean() * 100
        filas.append(
            {
                "gen": gen,
                "prevalencia": df[gen].mean() * 100,
                "mutados": int(df[gen].sum()),
                "LGG": prevalencia_lgg,
                "GBM": prevalencia_gbm,
                "diferencia": prevalencia_gbm - prevalencia_lgg,
            }
        )
    return pd.DataFrame(filas).sort_values("prevalencia", ascending=False)


def _cramer_v(tabla: np.ndarray) -> tuple[float, float]:
    """V de Cramér y p-valor de chi-cuadrado para una tabla de contingencia."""
    chi2, p_valor, _, _ = stats.chi2_contingency(tabla)
    n = tabla.sum()
    grados = min(tabla.shape[0] - 1, tabla.shape[1] - 1)
    if n == 0 or grados == 0:
        return 0.0, 1.0
    return float(np.sqrt(chi2 / (n * grados))), float(p_valor)


def asociacion_con_grado(ambito: str = "train") -> pd.DataFrame:
    """Fuerza y signo de la asociación de cada predictora con el grado.

    - `rho`: correlación de Spearman con `Grade` (da el signo: negativo empuja
      hacia LGG, positivo hacia GBM).
    - `v_cramer` y `p_valor`: prueba chi-cuadrado de independencia para las
      variables categóricas y binarias.
    """
    df = get_dataframe(ambito)
    filas = []

    for variable in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *GENE_FEATURES]:
        rho, p_spearman = stats.spearmanr(df[variable], df[TARGET])

        if variable in NUMERIC_FEATURES:
            v_cramer, p_valor = np.nan, float(p_spearman)
            tipo = "Clínica numérica"
        else:
            tabla = pd.crosstab(df[variable], df[TARGET]).values
            v_cramer, p_valor = _cramer_v(tabla)
            tipo = "Clínica categórica" if variable in CATEGORICAL_FEATURES else "Mutación"

        filas.append(
            {
                "variable": variable,
                "tipo": tipo,
                "rho": float(rho),
                "v_cramer": v_cramer,
                "p_valor": p_valor,
                "significativa": bool(p_valor < 0.05 and abs(rho) >= 0.10),
            }
        )

    return (
        pd.DataFrame(filas)
        .sort_values("rho", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def matriz_asociacion_genes(ambito: str = "train", top_n: int = 12) -> pd.DataFrame:
    """Matriz de V de Cramér entre los genes más prevalentes (multicolinealidad).

    Se limita a los `top_n` genes con más mutaciones: con los 20 la matriz se
    vuelve ilegible y las casillas de los genes raros no aportan información.
    """
    df = get_dataframe(ambito)
    genes = prevalencia_genes(ambito)["gen"].head(top_n).tolist()

    matriz = pd.DataFrame(np.eye(len(genes)), index=genes, columns=genes)
    for i, gen_a in enumerate(genes):
        for gen_b in genes[i + 1:]:
            tabla = pd.crosstab(df[gen_a], df[gen_b]).values
            valor = _cramer_v(tabla)[0] if tabla.shape == (2, 2) else 0.0
            matriz.loc[gen_a, gen_b] = valor
            matriz.loc[gen_b, gen_a] = valor
    return matriz


# --------------------------------------------------------------------------- #
# Cachés
# --------------------------------------------------------------------------- #
def clear_caches() -> None:
    """Invalida las cachés (útil tras reemplazar el dataset)."""
    load_data.cache_clear()
    _indices_particion.cache_clear()
