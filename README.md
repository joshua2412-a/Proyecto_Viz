# Dashboard de Rotación Laboral (Employee Attrition)

Aplicación analítica en **Dash** con arquitectura modular: cada pestaña del
dashboard es un archivo `.py` independiente que expone una función `layout()`.
Incluye generación de datos sintéticos, un modelo de regresión logística
entrenado y persistido, análisis exploratorio, evaluación del modelo y un
simulador de predicción en tiempo real.

---

## 1. Estructura del proyecto

```
Proyecto_Viz/
├── app.py                      # Aplicación principal: instancia Dash y enruta pestañas
├── requirements.txt            # Dependencias
├── README.md
├── .gitignore
│
├── data/
│   ├── generate_data.py        # Genera el dataset sintético
│   └── employee_attrition.csv  # (generado) 2.000 empleados
│
├── model/
│   ├── train_model.py          # Entrena y persiste el modelo
│   ├── model.pkl               # (generado) pipeline: preprocesamiento + modelo
│   └── metrics.json            # (generado) métricas, ROC y coeficientes
│
├── tabs/                       # Una pestaña = un archivo con layout()
│   ├── introduccion.py         # Explicación del problema
│   ├── contexto.py             # Impacto empresarial y coste de la rotación
│   ├── problema.py             # Tasa de abandono por departamento
│   ├── objetivos.py            # Objetivos y justificación
│   ├── marco_teorico.py        # Teoría + tabla de operacionalización
│   ├── metodologia.py          # Datos y modelo
│   ├── resultados.py           # EDA + métricas, ROC y matriz de confusión
│   ├── prediccion.py           # Formulario interactivo con el modelo .pkl
│   ├── limitaciones.py         # Limitaciones del estudio
│   └── conclusiones.py         # Hallazgos y recomendaciones
│
├── utils/                      # Código compartido (evita duplicación)
│   ├── config.py               # Rutas, esquema de datos, etiquetas, umbrales
│   ├── theme.py                # Paleta pastel y plantilla de Plotly
│   ├── components.py           # Componentes de interfaz reutilizables
│   ├── figures.py              # Constructores de todas las figuras
│   └── data_loader.py          # Carga de dataset, modelo y métricas (con caché)
│
└── assets/
    └── style.css               # Estilos (Dash lo carga automáticamente)
```

### Principio de diseño

| Capa | Responsabilidad | Regla |
|---|---|---|
| `data/` | Crear el dataset | No sabe nada del dashboard |
| `model/` | **Entrenar** y persistir | Solo escribe artefactos |
| `utils/data_loader.py` | **Cargar** artefactos | Nunca entrena |
| `utils/figures.py` | Construir figuras | Sin estado de interfaz |
| `tabs/*.py` | Contenido de una pestaña | No importa otras pestañas |
| `app.py` | Estructura y enrutado | No contiene contenido |

Añadir una pestaña nueva son dos pasos: crear `tabs/mi_pestana.py` con una
función `layout()` y sumar una fila a la lista `PESTANAS` de `app.py`.

---

## 2. Requisitos

- **Python 3.10 u 3.11** (recomendados; 3.12 también funciona).
- Sin GPU ni servicios externos: todo corre en local.

---

## 3. Entorno virtual e instalación

### Windows (PowerShell)

```powershell
cd C:\Users\joshu\Documents\GitHub\Proyecto_Viz

# 1. Crear el entorno virtual
py -3.11 -m venv venv

# 2. Activarlo
.\venv\Scripts\Activate.ps1
# Si PowerShell bloquea el script:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (CMD)

```cmd
py -3.11 -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para salir del entorno: `deactivate`.

---

## 4. Ejecución

### Opción A — un solo comando (recomendada)

```bash
python app.py
```

`app.py` comprueba al arrancar si existen el dataset y el modelo; si faltan,
los genera y entrena automáticamente. Luego abre:

**http://127.0.0.1:8050**

### Opción B — paso a paso (útil para entender el flujo o reentrenar)

```bash
# 1. Generar el dataset sintético -> data/employee_attrition.csv
python data/generate_data.py

# 2. Entrenar el modelo -> model/model.pkl + model/metrics.json
python model/train_model.py

# 3. Levantar el dashboard
python app.py
```

Ambos scripts imprimen un resumen en consola: tasa de abandono por
departamento el primero, métricas y variables más influyentes el segundo.

### Cambiar el puerto

Edita la última línea de `app.py`:

```python
app.run(debug=True, host="127.0.0.1", port=8060)
```

### Despliegue en producción

`app.py` expone `server = app.server`, así que funciona con cualquier WSGI:

```bash
gunicorn app:server -b 0.0.0.0:8050        # Linux / macOS
waitress-serve --port=8050 app:server      # Windows
```

Recuerda poner `debug=False` antes de desplegar.

---

## 5. El dataset

2.000 registros generados con semilla fija (`RANDOM_STATE = 42`), por lo que
son reproducibles en cualquier máquina.

| Variable | Tipo | Rango | Descripción |
|---|---|---|---|
| `edad` | entero | 21 – 60 | Edad del empleado |
| `salario` | entero | 1.5M – 18M COP | Salario mensual bruto |
| `anios_empresa` | entero | 0 – 35 | Antigüedad en la organización |
| `departamento` | texto | 7 categorías | Área funcional |
| `satisfaccion` | decimal | 1.0 – 5.0 | Satisfacción laboral declarada |
| `horas_trabajadas` | entero | 35 – 70 | Horas por semana |
| `promociones` | entero | 0 – 6 | Ascensos obtenidos |
| `abandono` | binario | 0 / 1 | **Variable objetivo** |

El abandono no es aleatorio: se calcula con un modelo logístico generador
(satisfacción, sobrecarga, salario relativo, antigüedad, promociones y efecto
del departamento) y se extrae de una distribución de Bernoulli. La tasa
resultante ronda el **19%**.

---

## 6. El modelo

```python
Pipeline([
    ("preprocesamiento", ColumnTransformer([
        ("numericas",   StandardScaler(),              NUMERIC_FEATURES),
        ("categoricas", OneHotEncoder(drop="first"),   ["departamento"]),
    ])),
    ("clasificador", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
```

- Partición **80/20 estratificada** con semilla fija.
- `class_weight="balanced"` para que la clase minoritaria (abandono) no quede
  ignorada: prioriza **recall**, porque no detectar una salida cuesta más que
  una conversación de más.
- El preprocesamiento va **dentro** del pipeline: el `.pkl` acepta datos en
  crudo y el formulario de predicción no replica ninguna transformación.

Desempeño de referencia sobre el conjunto de prueba:

| Métrica | Valor aproximado |
|---|---|
| Accuracy | 0.73 |
| Precisión | 0.40 |
| Recall | 0.83 |
| F1-score | 0.54 |
| AUC-ROC | 0.84 |
| AUC validación cruzada (5-fold) | 0.79 ± 0.02 |

---

## 7. Contenido de las pestañas

| Pestaña | Contenido |
|---|---|
| **Introducción** | El problema, las variables observadas y guía de navegación |
| **Contexto** | Coste estimado de la rotación y seis dimensiones de impacto |
| **Problema** | Tasa por departamento, brecha entre áreas, preguntas de investigación |
| **Objetivos** | Objetivo general, seis específicos, justificación y alcance |
| **Marco teórico** | Conceptos, teorías, fórmula del modelo y **tabla de operacionalización** |
| **Metodología** | Flujo en seis etapas, supuestos y definición de métricas |
| **Resultados** | *EDA*: dona, barras por departamento, explorador de histogramas y cajas, matriz de correlación. *Métricas*: accuracy, precisión, recall, F1, curva ROC, matriz de confusión y coeficientes |
| **Predicción** | Formulario con sliders, carga de `model.pkl`, indicador de riesgo, comparación del perfil y acciones sugeridas |
| **Limitaciones** | Nueve limitaciones por categoría y líneas de mejora |
| **Conclusiones** | Seis hallazgos, recomendaciones por horizonte y cumplimiento de objetivos |

---

## 8. Diseño visual

- **Bootstrap** vía `dash-bootstrap-components` (tema FLATLY) más `assets/style.css`.
- Paleta pastel en superficies (fondos, tarjetas, cabeceras) y dos colores de
  serie con contraste suficiente para los datos: **azul = permanece**,
  **coral = abandona**, siempre en ese orden.
- La paleta de series fue validada para daltonismo (protanopia, deuteranopia y
  tritanopia) y contraste sobre fondo claro; además, la identidad nunca depende
  solo del color: hay leyenda y etiquetas directas en todos los gráficos
  comparativos.
- Colores de estado (verde / ámbar / rojo) reservados para los niveles de
  riesgo y siempre acompañados de icono y texto.

---

## 9. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `ModuleNotFoundError: No module named 'dash'` | El entorno virtual no está activado o faltan dependencias: `pip install -r requirements.txt` |
| `FileNotFoundError: ... employee_attrition.csv` | Ejecuta `python data/generate_data.py` |
| `FileNotFoundError: ... model.pkl` | Ejecuta `python model/train_model.py` |
| `Address already in use` | El puerto 8050 está ocupado: cambia `port=` en `app.py` |
| Warning al cargar el `.pkl` tras actualizar scikit-learn | Reentrena: `python model/train_model.py` |
| La página se ve sin estilos | Comprueba que `assets/style.css` existe y reinicia la app |

---

## 10. Aviso de uso

El modelo estima **asociación estadística, no causalidad**, sobre datos
sintéticos. Una probabilidad alta es un motivo para conversar con la persona,
nunca un criterio para tomar decisiones laborales sobre ella.
