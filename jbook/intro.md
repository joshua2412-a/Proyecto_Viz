<div align="center">

# Caracterización Clínico-Molecular del Grado Tumoral en Gliomas

**Autores**

Alejandro Cantillo Escorcia

Joshua Hincapie LLorente

<br>

---

</div>



Los gliomas representan el tumor cerebral primario más común en adultos y se clasifican principalmente en Gliomas de Bajo Grado (LGG) y Glioblastoma Multiforme (GBM). Si bien los criterios histológicos e imagenológicos tradicionales han sido la base del diagnóstico, la caracterización biomolecular y clínica se ha vuelto imprescindible para determinar el pronóstico y el tratamiento idóneo. Sin embargo, las pruebas de secuenciación genética completa suponen un coste económico elevado para los sistemas de salud y los pacientes.

Este proyecto aborda este desafío analítico y médico mediante la integración de Análisis Exploratorio de Datos (EDA), aprendizaje automático y visualización interactiva. Utilizando datos recopilados de los proyectos TCGA-LGG y TCGA-GBM de *The Cancer Genome Atlas*, el objetivo central es construir una solución analítica integral que identifique el subconjunto óptimo de factores clínicos y mutaciones genéticas para clasificar con precisión la severidad del glioma, reduciendo los costos asociados a pruebas moleculares innecesarias.



## Objetivo general

Construir una solución analítica integral que permita identificar el subconjunto óptimo de factores clínicos y mutaciones genéticas para clasificar con precisión la severidad del glioma (LGG frente a GBM), buscando reducir la necesidad de pruebas moleculares innecesarias.

El énfasis está en identificar las variables que aportan mayor información para la clasificación del grado tumoral, en lugar de incorporar la mayor cantidad posible de características. De esta manera, el análisis busca determinar cuáles de las variables disponibles presentan una asociación relevante con el grado del tumor y cuáles pueden ser descartadas sin perder capacidad discriminativa.

La fase de modelado traduce estos hallazgos en clasificadores predictivos: regresión logística, Random Forest y XGBoost, combinados con cuatro técnicas de balanceo y cuatro métodos de optimización de hiperparámetros, evaluados con validación cruzada anidada y comparados con pruebas estadísticas formales.

## Objetivos específicos

- Describir el perfil clínico y mutacional de los pacientes, considerando variables como edad, género, grupo racial y estado de mutación de los genes disponibles, además de evaluar la calidad de los datos mediante la identificación de valores faltantes, valores atípicos y características de sus distribuciones.

- Separar el 20 % de los pacientes mediante una partición estratificada antes de realizar el análisis exploratorio, de modo que las decisiones tomadas durante el proceso no estén influenciadas por los datos destinados a la evaluación final.

- Contrastar cada predictor con el grado tumoral mediante pruebas estadísticas adecuadas a su naturaleza. Para las variables numéricas se empleará la prueba de Mann-Whitney y, para las variables binarias y categóricas, la prueba de Chi-cuadrado y la V de Cramér, con el propósito de ordenar las variables según su capacidad discriminativa.

- Evaluar la coocurrencia entre las mutaciones genéticas para detectar posibles relaciones o redundancias entre marcadores y determinar si el conjunto de variables retenido puede incorporarse posteriormente a un modelo sin generar problemas de multicolinealidad.

- Identificar las variables que presentan una asociación significativa con el grado tumoral y distinguirlas de aquellas que aportan poca o ninguna evidencia de relación, con el objetivo de determinar un conjunto reducido de características relevantes para la clasificación.

- Entrenar regresión logística, Random Forest y XGBoost con validación cruzada anidada, comparar técnicas de balanceo y de optimización de hiperparámetros, evaluar discriminación, calibración e interpretabilidad, y determinar con pruebas estadísticas si las diferencias entre modelos son reales.

- Presentar el análisis exploratorio y el modelado mediante un Jupyter Book y un tablero interactivo que permitan consultar los hallazgos, comparar los modelos y obtener predicciones sin necesidad de acceder directamente al código.



**Ficha Técnica del Dataset**

* **Nombre:** Glioma Grading Clinical and Mutation Features
* **Fuente:** [Glioma Grading Clinical and Mutation Features](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)
* **Muestra:** 839 registros de pacientes.
* **Atributos (23):** 20 genes con alta frecuencia de mutación y 3 variables clínicas relevantes.
* **Variable Objetivo:** Clasificación binaria entre **LGG** (Lower-Grade Glioma) y **GBM** (Glioblastoma Multiforme).
* **Naturaleza de datos:** Tabular, multivariada (numérica y categórica).



**Dashboard interactivo**

El análisis de este libro tiene una contraparte interactiva: un tablero en Dash
que recorre el planteamiento del problema, el marco teórico, la metodología, los
resultados del EDA y los modelos con gráficos navegables, y permite describir un
paciente para ver qué predice cada modelo y por qué. Está desplegado en Google Cloud Run:
[panel de visualización de datos](https://gliomas-dashboard-984926604434.us-central1.run.app).

La fuente de este libro y el código del dashboard viven en el mismo proyecto:
los notebooks en `jbook/`, la aplicación en la raíz, y una sola copia del
dataset en `dataset/` para que ambos analicen exactamente los mismos datos.

```{tableofcontents}
```

