# Imagen del dashboard de gliomas
#
# Construir:   docker build -t mi-dash .
# Ejecutar:    docker run -p 8080:8080 mi-dash      -> http://localhost:8080
# Exportar:    docker save -o mi-dash.tar mi-dash
#
# `app:server` es el objeto Flask que expone app.py (server = app.server): es lo
# que gunicorn necesita, porque Dash por sí solo no es una aplicación WSGI.

# Python 3.12 para igualar el entorno conda local: así el model.pkl que se
# genera dentro de la imagen y el que se genera en local son intercambiables.
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

# Entrenar el modelo dentro de la imagen, con el mismo Python y el mismo
# scikit-learn que la van a usar: evita cualquier problema de compatibilidad al
# deserializar un .pkl entrenado en otra versión. Es determinista (semilla 42),
# así que produce exactamente el mismo modelo que en local.
RUN python model/train_model.py

# Puerto de la aplicación (Cloud Run envía tráfico al 8080 por defecto)
EXPOSE 8080

# Un solo worker con varios hilos: cada worker carga pandas, scikit-learn y su
# propia copia del modelo, así que dos workers duplicarían la memoria.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "--preload", "app:server"]
