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

from utils.formato import miles, num, p_valor, pct
from utils.config import (
    CATEGORICAL_FEATURES,
    CLINICA_DESCRIPCION,
    DATA_PATH,
    FEATURES,
    GENDER_LABELS,
    GENE_DESCRIPCION,
    GENE_FEATURES,
    GRADE_LABELS,
    LABELS,
    MUTACION_LABELS,
    NUMERIC_FEATURES,
    RACE_LABELS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    VARIABLE_PAPEL,
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
# Exploración interactiva (pestaña Exploración)
# --------------------------------------------------------------------------- #
# Esta sección alimenta un módulo que deja al visitante elegir cualquiera de las
# 23 predictoras y verla sola (univariado) o contra el grado (bivariado). El
# resto del tablero cuenta un recorrido fijo; aquí la ruta la elige él.
#
# Toda la estadística vive aquí y no en la pestaña: las funciones devuelven
# diccionarios y DataFrames, nunca componentes de Dash.

TIPO_NUMERICA = "numerica"
TIPO_BINARIA = "binaria"
TIPO_CATEGORICA = "categorica"

TIPO_NOMBRES = {
    TIPO_NUMERICA: "Numérica continua",
    TIPO_BINARIA: "Binaria",
    TIPO_CATEGORICA: "Categórica nominal",
}

# Umbrales del proyecto, los mismos que usa asociacion_con_grado()
ALFA = 0.05
EFECTO_MINIMO = 0.10


def tipo_variable(variable: str) -> str:
    """Clasifica una predictora en numérica, binaria o categórica nominal."""
    if variable in NUMERIC_FEATURES:
        return TIPO_NUMERICA
    if variable in GENE_FEATURES or variable == "Gender":
        return TIPO_BINARIA
    if variable in CATEGORICAL_FEATURES:
        return TIPO_CATEGORICA
    raise ValueError(f"{variable!r} no es una predictora del dataset.")


def etiquetas_categorias(variable: str) -> dict:
    """Mapa código -> etiqueta legible para las variables no numéricas."""
    if variable == "Gender":
        return dict(GENDER_LABELS)
    if variable == "Race":
        return dict(RACE_LABELS)
    if variable in GENE_FEATURES:
        return dict(MUTACION_LABELS)
    return {}


def nombre_variable(variable: str) -> str:
    """Nombre legible: el de LABELS para las clínicas, el símbolo para los genes."""
    if variable in GENE_FEATURES:
        return variable
    return LABELS.get(variable, variable)


def opciones_variables() -> list[dict]:
    """Opciones del selector de la pestaña Exploración, en bloques.

    El orden no es alfabético a propósito: primero las clínicas y luego los
    genes por prevalencia, que es como se recorre el análisis en el libro.
    """
    opciones = [
        {"label": "Edad al diagnóstico", "value": "Age_at_diagnosis"},
        {"label": "Género", "value": "Gender"},
        {"label": "Grupo racial reportado", "value": "Race"},
    ]
    for gen in prevalencia_genes("train")["gen"]:
        opciones.append({"label": f"{gen} · mutación", "value": gen})
    return opciones


def ficha_variable(variable: str) -> dict:
    """Ficha de diccionario de datos de una sola variable.

    Es el mismo contenido que la tabla de operacionalización del marco teórico,
    pero recortado a la variable elegida para que quepa junto al gráfico. El
    marco teórico sigue mostrando la tabla completa: aquí no se sustituye nada.
    """
    tipo = tipo_variable(variable)
    etiquetas = etiquetas_categorias(variable)

    if variable in GENE_FEATURES:
        descripcion = GENE_DESCRIPCION.get(variable, "")
        papel = "Predictora molecular · chi-cuadrado, V de Cramér y Spearman"
    else:
        descripcion = CLINICA_DESCRIPCION.get(variable, "")
        papel = VARIABLE_PAPEL.get(variable, "")

    if tipo == TIPO_NUMERICA:
        codificacion = "Continua, en años (con decimales)"
    else:
        codificacion = " · ".join(f"{k} = {v}" for k, v in etiquetas.items())

    return {
        "variable": variable,
        "nombre": nombre_variable(variable),
        "tipo": tipo,
        "tipo_nombre": TIPO_NOMBRES[tipo],
        "codificacion": codificacion,
        "descripcion": descripcion,
        "papel": papel,
        "es_gen": variable in GENE_FEATURES,
    }


def _serie_legible(variable: str, ambito: str) -> pd.Series:
    """La columna con las categorías ya traducidas a texto."""
    df = get_dataframe(ambito)
    etiquetas = etiquetas_categorias(variable)
    return df[variable].map(etiquetas) if etiquetas else df[variable]


def resumen_univariado(variable: str, ambito: str = "train") -> dict:
    """Estadísticos descriptivos de una variable, sin mirar el grado.

    Devuelve una lista de pares (etiqueta, valor ya formateado) lista para
    pintar, porque el formato de cada estadístico depende de su tipo y no tiene
    sentido repetir esa lógica en la pestaña.
    """
    df = get_dataframe(ambito)
    tipo = tipo_variable(variable)
    serie = df[variable]
    n = int(serie.size)

    if tipo == TIPO_NUMERICA:
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        atipicos = int(((serie < bajo) | (serie > alto)).sum())
        filas = [
            ("Observaciones", f"{n}"),
            ("Media", num(serie.mean(), 2)),
            ("Mediana", num(serie.median(), 2)),
            ("Desviación estándar", num(serie.std(), 2)),
            ("Rango intercuartílico", f"{num(q1, 2)} a {num(q3, 2)}"),
            ("Mínimo y máximo", f"{num(serie.min(), 2)} a {num(serie.max(), 2)}"),
            ("Asimetría", num(serie.skew(), 2, signo=True)),
            ("Atípicos (criterio 1,5 · IQR)", f"{atipicos}"),
        ]
        return {"tipo": tipo, "n": n, "filas": filas, "atipicos": atipicos}

    legible = _serie_legible(variable, ambito)
    conteo = legible.value_counts()
    moda, frecuencia = conteo.index[0], int(conteo.iloc[0])
    filas = [
        ("Observaciones", f"{n}"),
        ("Categorías", f"{conteo.size}"),
        ("Más frecuente", f"{moda}"),
        ("Frecuencia", f"{frecuencia} ({pct(frecuencia / n * 100)})"),
        ("Menos frecuente", f"{conteo.index[-1]}"),
        ("Frecuencia", f"{int(conteo.iloc[-1])} ({pct(conteo.iloc[-1] / n * 100)})"),
    ]
    if tipo == TIPO_BINARIA and variable in GENE_FEATURES:
        filas.insert(1, ("Prevalencia de mutación", pct(serie.mean() * 100)))
    return {"tipo": tipo, "n": n, "filas": filas, "conteo": conteo}


def distribucion_univariada(variable: str, ambito: str = "train") -> pd.DataFrame:
    """Conteo y porcentaje por categoría, para la figura univariada."""
    legible = _serie_legible(variable, ambito)
    conteo = legible.value_counts()
    return pd.DataFrame(
        {
            "categoria": conteo.index.astype(str),
            "pacientes": conteo.values.astype(int),
            "porcentaje": conteo.values / conteo.values.sum() * 100,
        }
    )


def tabla_bivariada(variable: str, ambito: str = "train") -> pd.DataFrame:
    """Reparto de LGG y GBM DENTRO de cada categoría de la variable.

    El porcentaje se calcula dentro de la categoría y no sobre el total a
    propósito: la pregunta del análisis es qué proporción de los pacientes con
    una mutación acaba siendo GBM, y un conteo bruto la esconde detrás del
    tamaño de cada grupo.
    """
    df = get_dataframe(ambito).copy()
    df["categoria"] = _serie_legible(variable, ambito).astype(str)

    tabla = (
        df.groupby(["categoria", "grade_label"], observed=True)
        .size()
        .rename("pacientes")
        .reset_index()
    )
    totales = tabla.groupby("categoria", observed=True)["pacientes"].transform("sum")
    tabla["porcentaje"] = tabla["pacientes"] / totales * 100
    tabla["total_categoria"] = totales
    return tabla


def prueba_bivariada(variable: str, ambito: str = "train") -> dict:
    """La prueba que corresponde a la variable, con su tamaño del efecto.

    Qué prueba y por qué:
      - Numérica: U de Mann-Whitney. No paramétrica porque Shapiro-Wilk rechaza
        la normalidad de la edad (es bimodal), igual que en el notebook.
      - Binaria o categórica: chi-cuadrado de independencia, con la V de Cramér
        como magnitud, porque el chi-cuadrado solo dice si hay asociación.

    El signo de Spearman da la dirección, pero solo se devuelve cuando tiene
    sentido: en una variable nominal de cuatro niveles como Race, ordenar las
    categorías es arbitrario y el signo no significaría nada.
    """
    df = get_dataframe(ambito)
    tipo = tipo_variable(variable)
    rho, p_spearman = stats.spearmanr(df[variable], df[TARGET])
    rho = float(rho)

    if tipo == TIPO_NUMERICA:
        lgg = df.loc[df[TARGET] == 0, variable]
        gbm = df.loc[df[TARGET] == 1, variable]
        estadistico, p_valor = stats.mannwhitneyu(lgg, gbm, alternative="two-sided")
        p_normalidad = float(stats.shapiro(df[variable])[1]) if len(df) <= 5000 else float("nan")
        detalle = [
            ("Hipótesis nula", "Las dos distribuciones de la variable son iguales "
                               "en LGG y en GBM."),
            ("Estadístico U", miles(estadistico)),
            ("Tamaño del efecto", f"ρ de Spearman = {num(rho, 3, signo=True)}"),
            ("Medianas", f"LGG {num(lgg.median())} · GBM {num(gbm.median())}"),
            ("Normalidad (Shapiro-Wilk)",
             p_valor(p_normalidad)),
        ]
        return {
            "prueba": "U de Mann-Whitney",
            "motivo": "La variable es continua y no sigue una distribución normal, "
                      "así que se comparan rangos en vez de medias.",
            "p_valor": float(p_valor),
            "rho": rho,
            "v_cramer": float("nan"),
            "detalle": detalle,
            "direccion": "LGG" if rho < 0 else "GBM",
            "hay_direccion": True,
            "aviso": "",
        }

    tabla = pd.crosstab(df[variable], df[TARGET])
    chi2, p_valor, gl, esperadas = stats.chi2_contingency(tabla.values)
    v_cramer = _cramer_v(tabla.values)[0]
    celdas_escasas = int((esperadas < 5).sum())

    detalle = [
        ("Hipótesis nula", "La variable y el grado tumoral son independientes."),
        ("Estadístico χ²", f"{num(chi2, 2)} con {gl} grado{'s' if gl != 1 else ''} de libertad"),
        ("Tamaño del efecto", f"V de Cramér = {num(v_cramer, 3)}"),
    ]
    if tipo == TIPO_BINARIA:
        detalle.append(("Dirección", f"ρ de Spearman = {num(rho, 3, signo=True)}"))
    else:
        detalle.append(("Dirección", "no aplica: la variable es nominal y ordenar "
                                     "sus categorías sería arbitrario"))

    aviso = ""
    if celdas_escasas:
        aviso = (f"{celdas_escasas} de las {esperadas.size} casillas tienen una "
                 "frecuencia esperada menor que 5, así que la aproximación "
                 "chi-cuadrado pierde precisión y el p-valor hay que tomarlo con "
                 "cautela.")

    return {
        "prueba": "Chi-cuadrado de independencia",
        "motivo": "Las dos variables son categóricas, así que se compara la tabla "
                  "de contingencia observada con la que cabría esperar si no "
                  "hubiera relación.",
        "p_valor": float(p_valor),
        "rho": rho,
        "v_cramer": float(v_cramer),
        "detalle": detalle,
        "direccion": "LGG" if rho < 0 else "GBM",
        "hay_direccion": tipo == TIPO_BINARIA,
        "aviso": aviso,
    }


def lectura_univariada(variable: str, ambito: str = "train") -> list[str]:
    """Dos o tres frases que describen la variable por sí sola."""
    resumen = resumen_univariado(variable, ambito)
    ficha = ficha_variable(variable)
    df = get_dataframe(ambito)
    lineas = []

    if resumen["tipo"] == TIPO_NUMERICA:
        serie = df[variable]
        sesgo = serie.skew()
        forma = ("es prácticamente simétrica" if abs(sesgo) < 0.25
                 else "se alarga hacia la derecha" if sesgo > 0
                 else "se alarga hacia la izquierda")
        lineas.append(
            f"La distribución {forma} (asimetría {num(sesgo, 2, signo=True)}) y se concentra "
            f"entre los {serie.quantile(0.25):.0f} y los {serie.quantile(0.75):.0f} "
            f"años, con una mediana de {serie.median():.0f}."
        )
        if resumen["atipicos"]:
            lineas.append(
                f"Hay {resumen['atipicos']} observaciones atípicas según el criterio "
                "de 1,5 veces el rango intercuartílico; no se eliminan, porque son "
                "edades clínicamente posibles."
            )
        else:
            lineas.append(
                "No hay observaciones atípicas según el criterio de 1,5 veces el "
                "rango intercuartílico."
            )
        return lineas

    conteo = resumen["conteo"]
    total = resumen["n"]
    dominante = conteo.index[0]
    porcentaje = conteo.iloc[0] / total * 100

    if ficha["es_gen"]:
        prevalencia = df[variable].mean() * 100
        frecuencia = ("muy frecuente" if prevalencia >= 40
                      else "frecuente" if prevalencia >= 15
                      else "poco frecuente" if prevalencia >= 5
                      else "rara")
        lineas.append(
            f"La mutación en {variable} es {frecuencia} en esta cohorte: aparece en "
            f"{pct(prevalencia)} de los pacientes ({int(df[variable].sum())} de {total})."
        )
        if prevalencia < 5:
            lineas.append(
                "Con tan pocos casos mutados, cualquier asociación que se le "
                "encuentre descansa sobre un grupo muy pequeño."
            )
    else:
        lineas.append(
            f"La categoría más frecuente es «{dominante}», con {pct(porcentaje)} "
            f"de los {total} pacientes."
        )
        if porcentaje >= 80:
            lineas.append(
                "El reparto está muy desbalanceado: las categorías minoritarias "
                "tienen tan pocas observaciones que cualquier porcentaje calculado "
                "sobre ellas es inestable."
            )
        elif conteo.size == 2:
            lineas.append(
                f"La otra categoría, «{conteo.index[-1]}», reúne el "
                f"{pct(conteo.iloc[-1] / total * 100)} restante."
            )
    return lineas


def lectura_bivariada(variable: str, ambito: str = "train") -> list[str]:
    """Dos o tres frases sobre la relación de la variable con el grado."""
    prueba = prueba_bivariada(variable, ambito)
    tipo = tipo_variable(variable)
    significativa = prueba["p_valor"] < ALFA and abs(prueba["rho"]) >= EFECTO_MINIMO
    lineas = []

    if tipo == TIPO_NUMERICA:
        df = get_dataframe(ambito)
        lgg = df.loc[df[TARGET] == 0, variable].median()
        gbm = df.loc[df[TARGET] == 1, variable].median()
        lineas.append(
            f"La mediana es de {lgg:.0f} años en LGG y de {gbm:.0f} en GBM: una "
            f"diferencia de {abs(gbm - lgg):.0f} años, y la U de Mann-Whitney la "
            f"da por distinguible del azar ({p_valor(prueba['p_valor'])})."
        )
        lineas.append(
            f"La magnitud de la asociación es ρ = {num(prueba['rho'], 2, signo=True)}, el "
            "correlato clínico más fuerte del panel, aunque no separa los grupos "
            "por sí sola: las dos distribuciones se solapan ampliamente."
        )
        return lineas

    tabla = tabla_bivariada(variable, ambito)
    gbm = tabla[tabla["grade_label"] == "GBM"].sort_values("porcentaje", ascending=False)
    if not gbm.empty:
        alta, baja = gbm.iloc[0], gbm.iloc[-1]
        lineas.append(
            f"De los pacientes con «{alta['categoria']}», el {pct(alta['porcentaje'])} "
            f"son GBM; con «{baja['categoria']}» son el {pct(baja['porcentaje'])}. "
            f"La brecha es de {num(alta['porcentaje'] - baja['porcentaje'])} puntos."
        )

    if significativa:
        lineas.append(
            f"El chi-cuadrado encuentra asociación ({p_valor(prueba['p_valor'])}) y "
            f"la V de Cramér la sitúa en {num(prueba['v_cramer'], 2)}"
            + (f", empujando hacia {prueba['direccion']}." if prueba["hay_direccion"] else ".")
        )
    elif prueba["p_valor"] < ALFA:
        lineas.append(
            f"El chi-cuadrado sale significativo ({p_valor(prueba['p_valor'])}), pero "
            f"la magnitud es pequeña (ρ = {num(prueba['rho'], 2, signo=True)}, por debajo del umbral "
            "de 0,10 que usa el proyecto): con 671 pacientes, una diferencia mínima "
            "basta para alcanzar significancia."
        )
    else:
        lineas.append(
            f"El chi-cuadrado no encuentra asociación ({p_valor(prueba['p_valor'])}): "
            "esta variable no distingue los dos grados en esta muestra."
        )

    if prueba["aviso"]:
        lineas.append(prueba["aviso"])
    return lineas


# --------------------------------------------------------------------------- #
# Cachés
# --------------------------------------------------------------------------- #
def clear_caches() -> None:
    """Invalida las cachés (útil tras reemplazar el dataset)."""
    load_data.cache_clear()
    _indices_particion.cache_clear()
