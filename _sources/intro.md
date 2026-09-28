<div align="center">

# Caracterización Clínico-Molecular del Grado Tumoral en Gliomas

**Optimización del Diagnóstico de Gliomas mediante Ciencia de Datos y Aprendizaje Automático**
<br>

**Autores**

Alejandro Cantillo Escorcia

Joshua Hincapie LLorente

<br>

---

</div>



Los gliomas representan el tumor cerebral primario más común en adultos y se clasifican principalmente en Gliomas de Bajo Grado (LGG) y Glioblastoma Multiforme (GBM). Si bien los criterios histológicos e imagenológicos tradicionales han sido la base del diagnóstico, la caracterización biomolecular y clínica se ha vuelto imprescindible para determinar el pronóstico y el tratamiento idóneo. Sin embargo, las pruebas de secuenciación genética completa suponen un coste económico elevado para los sistemas de salud y los pacientes.

Este proyecto aborda este desafío analítico y médico mediante la integración de Análisis Exploratorio de Datos (EDA), aprendizaje automático y visualización interactiva. Utilizando datos recopilados de los proyectos TCGA-LGG y TCGA-GBM de *The Cancer Genome Atlas*, el objetivo central es construir una solución analítica integral que identifique el subconjunto óptimo de factores clínicos y mutaciones genéticas para clasificar con precisión la severidad del glioma, reduciendo los costos asociados a pruebas moleculares innecesarias.



**Ficha Técnica del Dataset**

* **Nombre:** Glioma Grading Clinical and Mutation Features
* **Fuente:** [Glioma Grading Clinical and Mutation Features](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)
* **Muestra:** 839 registros de pacientes.
* **Atributos (23):** 20 genes con alta frecuencia de mutación y 3 variables clínicas relevantes.
* **Variable Objetivo:** Clasificación binaria entre **LGG** (Lower-Grade Glioma) y **GBM** (Glioblastoma Multiforme).
* **Naturaleza de datos:** Tabular, multivariada (numérica y categórica).



**Dashboard interactivo**

El análisis de este libro tiene una contraparte interactiva: un tablero en Dash
que replica el EDA con gráficos navegables e incorpora un simulador que estima
la probabilidad de GBM a partir de la edad, el género, el grupo racial y las 20
mutaciones del panel. Está desplegado en Google Cloud Run:
[panel de visualización y predicción](https://gliomas-dashboard-984926604434.us-central1.run.app).

La fuente de este libro y el código del dashboard viven en el mismo proyecto:
los notebooks en `jbook/`, la aplicación en la raíz, y una sola copia del
dataset en `dataset/` para que ambos analicen exactamente los mismos datos.

```{tableofcontents}
```
