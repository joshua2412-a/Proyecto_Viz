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

# Nota: el dashboard cubre el análisis exploratorio y no carga ni entrena ningún
# modelo, así que la imagen solo necesita el dataset y el código de las pestañas.

# Puerto de la aplicación (Cloud Run envía tráfico al 8080 por defecto)
EXPOSE 8080

# Un solo worker con varios hilos: cada worker carga su propia copia de pandas,
# scipy y el dataset, así que dos workers duplicarían la memoria sin aportar
# nada (el trabajo del dashboard es de espera, no de CPU).
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "--preload", "app:server"]
