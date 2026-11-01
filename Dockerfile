# Imagen del dashboard de gliomas
#
# Construir:   docker build -t mi-dash .
# Ejecutar:    docker run -p 8080:8080 mi-dash      -> http://localhost:8080
# Exportar:    docker save -o mi-dash.tar mi-dash
#
# `app:server` es el objeto Flask que expone app.py (server = app.server): es lo
# que gunicorn necesita, porque Dash por sí solo no es una aplicación WSGI.

# Python 3.12 para igualar el entorno conda local y evitar diferencias de
# versión en pandas, scipy y scikit-learn entre el contenedor y el desarrollo.
FROM python:3.12-slim

# Evita que Python escriba .pyc y fuerza logs sin búfer (se ven en Cloud Run)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar primero las dependencias: si el código cambia pero requirements.txt no,
# Docker reutiliza esta capa de la caché y el build es mucho más rápido
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto (.dockerignore excluye jbook/, venv, cachés...)
COPY . .

# Nota: el dashboard no entrena modelos. Las pestañas Modelos y Predicción leen
# modelos/*.json (exportados desde jbook/ con `python -m modelado.exportar`) y
# predicen con numpy, así que la imagen no necesita XGBoost ni imbalanced-learn.

# Puerto de la aplicación (Cloud Run envía tráfico al 8080 por defecto)
EXPOSE 8080

# Un solo worker con varios hilos: cada worker carga su propia copia de pandas,
# scipy y el dataset, así que dos workers duplicarían la memoria sin aportar
# nada (el trabajo del dashboard es de espera, no de CPU).
# Sin --preload: app.py arranca un hilo que construye todas las pestañas al
# iniciar (`_precalentar`). Con --preload ese hilo correría en el proceso
# maestro, el worker heredaría las cachés a medio llenar y, si el hilo tenía
# un bloqueo tomado al hacer fork, podría quedarse colgado.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "app:server"]
