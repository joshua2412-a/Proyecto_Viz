# Caracterización Clínico-Molecular del Grado Tumoral en Gliomas

Proyecto de visualización y analítica de datos sobre los proyectos **TCGA-LGG** y
**TCGA-GBM** de *The Cancer Genome Atlas*. Clasifica el grado de un glioma
—glioma de bajo grado (**LGG**) frente a glioblastoma multiforme (**GBM**)— a
partir de la edad al diagnóstico, el perfil demográfico y 20 mutaciones
genéticas, con el objetivo de identificar el subconjunto mínimo de marcadores
que hace falta secuenciar.

El proyecto se entrega en dos piezas que comparten una única copia de los datos:

| Pieza | Qué es | Dónde vive |
|---|---|---|
| **Dashboard** | Capa interactiva: EDA navegable y simulador de perfiles | raíz del repositorio · Dash + Plotly |
| **Jupyter Book** | Análisis completo con su desarrollo estadístico | `jbook/` · publicado en [GitHub Pages](https://joshua2412-a.github.io/Proyecto_Viz/) |

**Autores:** Alejandro Cantillo Escorcia · Joshua Hincapie LLorente

---

## 1. Estructura del proyecto

```
Proyecto_Viz/
├── app.py                  # Dash: estructura, barra superior y enrutado de pestañas
├── Dockerfile              # Imagen del contenedor (Cloud Run)
├── .dockerignore           # Qué NO entra en la imagen
├── app.yaml                # Despliegue alternativo en App Engine
├── requirements.txt        # Dependencias del dashboard
├── dataset/
│   ├── README.md           # Cómo obtener el CSV y esquema esperado
│   └── TCGA_InfoWithGrade.csv   # ← única copia de los datos (no versionada por defecto)
├── model/
│   ├── train_model.py      # Entrena y persiste (mismo pipeline que el notebook)
│   ├── model.pkl           # Pipeline entrenado (se genera)
│   └── metrics.json        # Métricas del conjunto de prueba (se genera)
├── utils/
│   ├── config.py           # Rutas, esquema TCGA, etiquetas, umbrales, enlaces
│   ├── theme.py            # Paleta validada y plantilla de Plotly
│   ├── data_loader.py      # Carga de datos y modelo + estadística descriptiva
│   ├── figures.py          # Constructores de todas las figuras
│   └── components.py       # Componentes de interfaz reutilizables
├── tabs/                   # Una pestaña por archivo, cada una con su layout()
│   ├── introduccion.py     ├── metodologia.py     ├── limitaciones.py
│   ├── contexto.py         ├── resultados.py      ├── conclusiones.py
│   ├── problema.py         ├── prediccion.py      └── documentacion.py
│   ├── objetivos.py        └── marco_teorico.py
├── assets/style.css        # Estilos (Dash los carga automáticamente)
├── jbook/                  # Fuente del Jupyter Book
│   ├── _config.yml         ├── intro.md           ├── 01_EDA.ipynb
│   ├── _toc.yml            ├── requirements.txt   └── 02_Baseline.ipynb
└── scripts/
    └── publicar_libro.py   # Compila el libro y actualiza gh-pages
```

### Principio de diseño

| Capa | Responsabilidad | Regla |
|---|---|---|
| `dataset/` | Guardar los datos | Única copia: la leen el dashboard y los notebooks |
| `model/` | **Entrenar** y persistir | Solo escribe artefactos |
| `utils/data_loader.py` | **Cargar** artefactos y calcular descriptivos | Nunca entrena |
| `utils/figures.py` | Construir figuras | Sin estado de interfaz |
| `tabs/*.py` | Contenido de una pestaña | No importa otras pestañas |
| `app.py` | Estructura y enrutado | No contiene contenido |
| `jbook/` | Documentación del análisis | Lee el mismo CSV que la app |

---

## 2. Requisitos

- Python 3.11 o 3.12
- El archivo `dataset/TCGA_InfoWithGrade.csv` (ver [`dataset/README.md`](dataset/README.md))

---

## 3. Entorno virtual e instalación

### Windows (PowerShell)

```powershell
# 1. Crear el entorno virtual
python -m venv venv

# 2. Activarlo
.\venv\Scripts\Activate.ps1
# Si PowerShell bloquea el script:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Anaconda PowerShell (el entorno del proyecto)

```powershell
# Python 3.12 para igualar el runtime de App Engine (python312 en app.yaml)
conda create -n gliomas python=3.12 -y
conda activate gliomas

# Las dependencias van por pip, no por conda: así los pines del requirements
# (dash<3, scikit-learn<1.8) se resuelven igual que en el servidor.
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El Jupyter Book va en un entorno aparte, porque `jupyter-book` arrastra Sphinx y
sus pines chocan con los de Dash:

```powershell
conda create -n gliomas-libro python=3.12 -y
conda activate gliomas-libro
pip install -r jbook\requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Ejecución

### Opción A — un solo comando

```bash
python app.py
```

Si `model/model.pkl` no existe, `app.py` entrena el modelo antes de arrancar.
El dashboard queda en <http://127.0.0.1:8080>, el mismo puerto que usa el
contenedor.

### Opción B — paso a paso

```bash
# 1. Entrenar el modelo -> model/model.pkl + model/metrics.json
python model/train_model.py

# 2. (opcional) repetir la búsqueda de hiperparámetros con GridSearchCV
python model/train_model.py --buscar

# 3. Levantar el dashboard
python app.py
```

> Si falta el dataset, la app arranca igual y cada pestaña con datos muestra un
> aviso explicando qué archivo falta y dónde colocarlo.

---

## 5. El dataset

**Glioma Grading Clinical and Mutation Features** · 839 pacientes · 24
variables · sin valores faltantes.
[Ficha en UCI](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)

| Variable | Tipo | Codificación |
|---|---|---|
| `Grade` | binaria | 0 = LGG · 1 = GBM — **variable objetivo** |
| `Age_at_diagnosis` | continua | Años, con decimales (recogen los días exactos) |
| `Gender` | binaria | 0 = masculino · 1 = femenino |
| `Race` | categórica | 0 = White · 1 = Black or African American · 2 = Asian · 3 = American Indian or Alaska Native |
| 20 genes | binarias | 0 = no mutado (wildtype) · 1 = mutado |

Genes del panel: `IDH1`, `TP53`, `ATRX`, `PTEN`, `EGFR`, `CIC`, `MUC16`,
`PIK3CA`, `NF1`, `PIK3R1`, `FUBP1`, `RB1`, `NOTCH1`, `BCOR`, `CSMD3`,
`SMARCA4`, `GRIN2A`, `IDH2`, `FAT4`, `PDGFRA`.

`utils/data_loader.py` valida el esquema al cargar: si falta una columna, el
error dice exactamente cuál.

---

## 6. El modelo

Pipeline idéntico al del notebook `jbook/02_Baseline.ipynb`:

| Bloque | Variables | Transformación |
|---|---|---|
| Clínica numérica | `Age_at_diagnosis` | `StandardScaler` |
| Clínicas categóricas | `Gender`, `Race` | `OneHotEncoder(drop="if_binary")` |
| Mutacionales | 20 genes | `passthrough` (ya son 0/1) |

- **Partición:** 80/20 estratificada, `random_state=42` → 671 entrenamiento / 168 prueba
- **Modelo:** `LogisticRegression(C=0.3162, penalty="l1", solver="liblinear", class_weight=None)`
- **Selección:** `GridSearchCV` sobre 52 combinaciones, 5-fold estratificado, métrica AUC-ROC

Desempeño reportado en el libro sobre el conjunto de prueba:

| Métrica | Valor |
|---|---|
| Accuracy | 0.87 |
| Precisión (GBM) | 0.79 |
| Recall (GBM) | 0.93 |
| F1-score (GBM) | 0.86 |
| AUC-ROC (validación cruzada) | 0.9165 |

Las cifras que muestra el dashboard se leen de `model/metrics.json`, así que se
actualizan solas al reentrenar.

---

## 7. Contenido de las pestañas

| Pestaña | Contenido |
|---|---|
| **Introducción** | El problema clínico, ficha técnica del dataset y guía de navegación |
| **Contexto clínico** | Qué distingue LGG de GBM, composición de la cohorte y edad al diagnóstico |
| **Problema** | Coste de la secuenciación completa y reparto de las 23 predictoras entre las que aportan señal y las que no |
| **Objetivos** | Objetivo general, seis específicos y criterios de cumplimiento contrastados con las métricas reales |
| **Marco teórico** | Operacionalización de variables, formulación de la regresión logística, odds ratios, métricas y panel de genes |
| **Metodología** | Seis etapas, preprocesamiento, hiperparámetros y coste computacional |
| **Resultados** | *EDA* con selector de conjunto (entrenamiento / prueba / completo): distribución del grado, edad, variables clínicas, prevalencia de mutaciones, asociación con el grado y matriz de multicolinealidad. *Modelo*: matriz de confusión, curva ROC y coeficientes |
| **Predicción** | Formulario (edad, género, grupo racial, 20 mutaciones), probabilidad estimada de GBM, desglose de contribuciones al log-odds y curva de sensibilidad a la edad |
| **Limitaciones** | Ocho fronteras del trabajo, con la tabla de representatividad de la cohorte |
| **Conclusiones** | Hallazgos con su evidencia, siguientes pasos y entregables |
| **Documentación** | Enlaces al Jupyter Book capítulo a capítulo y vista embebida |

---

## 8. El Jupyter Book

La fuente vive en `jbook/`; el sitio publicado es el resultado de compilarla.

```bash
pip install -r jbook/requirements.txt

# Compilar en local
jupyter-book build jbook/
# abre jbook/_build/html/index.html

# Compilar y publicar en la rama gh-pages de este mismo repositorio
python scripts/publicar_libro.py

# Igual, pero reejecutando los notebooks desde el CSV
python scripts/publicar_libro.py --forzar-ejecucion
```

Por defecto `_config.yml` usa `execute_notebooks: "off"`: publica las salidas ya
guardadas en los `.ipynb`, lo que hace el build rápido y sin dependencias
científicas. `--forzar-ejecucion` recalcula todo y necesita el dataset.

El notebook `01_EDA.ipynb` guarda `jbook/datos_modelado.pkl` con las
particiones, y `02_Baseline.ipynb` lo lee: al reejecutar, el orden importa.

---

## 9. Despliegue en Google Cloud (Docker + Artifact Registry + Cloud Run)

Es la ruta del Módulo 5 del curso: se empaqueta la app en un contenedor, la
imagen se guarda en Artifact Registry y se despliega como servicio en Cloud Run.
`app.py` expone `server = app.server`, que es el objeto WSGI que gunicorn
necesita, y el `Dockerfile` entrena el modelo dentro de la imagen.

### 9.1 Probar el contenedor en local

```powershell
docker build -t mi-dash .
docker run -p 8080:8080 mi-dash      # http://localhost:8080
```

### 9.2 Preparar el proyecto en Google Cloud

En la consola web ([console.cloud.google.com](https://console.cloud.google.com)):

1. Crear un proyecto nuevo y vincularlo a una cuenta de facturación.
2. Habilitar las APIs **Artifact Registry** y **Cloud Build**.
3. Crear un repositorio en Artifact Registry (formato Docker, región
   `us-central1`).

### 9.3 Subir la imagen

Opción A — como enseña el módulo, exportando la imagen y subiéndola a Cloud
Shell:

```powershell
docker save -o mi-dash.tar mi-dash      # el .tar pesa ~700 MB - 1 GB
```

Y en Cloud Shell, tras subir el archivo:

```bash
docker load < mi-dash.tar
docker images
docker tag mi-dash:latest us-central1-docker.pkg.dev/TU-PROYECTO/TU-REPO/mi-dash:1.0
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud services enable artifactregistry.googleapis.com
docker push us-central1-docker.pkg.dev/TU-PROYECTO/TU-REPO/mi-dash:1.0
```

Opción B — construyendo en la nube, sin subir el `.tar` (mucho más rápido, la
imagen nunca sale de Google Cloud):

```powershell
gcloud builds submit --tag us-central1-docker.pkg.dev/TU-PROYECTO/TU-REPO/mi-dash:1.0
```

### 9.4 Desplegar en Cloud Run

Desde la consola web (Cloud Run → Crear servicio → seleccionar la imagen del
repositorio) o por comando:

```powershell
gcloud run deploy gliomas-dashboard `
  --image us-central1-docker.pkg.dev/TU-PROYECTO/TU-REPO/mi-dash:1.0 `
  --region us-central1 `
  --allow-unauthenticated `
  --port 8080 `
  --memory 1Gi
```

`--memory 1Gi` importa: con pandas, scikit-learn y plotly cargados, los 512 MB
que Cloud Run asigna por defecto se quedan cortos y la instancia se reinicia.

### 9.5 Después del primer despliegue

Pon la URL definitiva en `URL_DASHBOARD` de `utils/config.py`, en
`extra_footer` de `jbook/_config.yml` y en `jbook/intro.md`, y vuelve a publicar
el libro con `python scripts/publicar_libro.py`.

> **Alternativa sin Docker:** `app.yaml` y `.gcloudignore` dejan el proyecto
> listo también para App Engine (`gcloud app deploy`). Es una segunda vía, no la
> del módulo; si no la vas a usar, puedes borrar esos dos archivos.

---

## 10. Reproducibilidad

- Semilla fija (`RANDOM_STATE = 42`) en `utils/config.py`, compartida por el
  dashboard y los notebooks: la partición 80/20 es exactamente la misma en las
  dos piezas.
- `scikit-learn` está fijado a `<1.8` a propósito: en 1.8 se deprecó el
  argumento `penalty` de `LogisticRegression`, que es el que usa el pipeline
  del notebook.
- El EDA del dashboard describe por defecto el conjunto de **entrenamiento**,
  igual que el libro, para no filtrar información del conjunto de prueba.
