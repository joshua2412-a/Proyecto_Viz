"""
Mapa de navegación del tablero: el orden de las pestañas, cómo se llaman y
cómo se agrupan.

Es la única fuente de esta información. Antes vivía en dos sitios —la lista
PESTANAS de app.py y los BLOQUES de la guía de la portada— con las
descripciones escritas dos veces, palabra por palabra. Ahora la guía de la
portada y el índice desplegable leen de aquí, así que añadir una pestaña es
tocar un solo archivo.

Este módulo no importa nada del proyecto a propósito: app.py le añade los
módulos de cada pestaña y las vistas le añaden el formato. Así puede leerlo
cualquiera sin arrastrar dependencias.
"""

from __future__ import annotations

# (id, etiqueta, icono, descripción) en el orden en que se recorre el tablero.
# El icono es de Bootstrap Icons, que ya carga app.py.
PESTANAS = [
    ("tab-introduccion", "Introducción", "bi-house",
     "Portada, ficha del dataset y guía de lectura."),
    ("tab-contexto", "Contexto clínico", "bi-heart-pulse",
     "Qué son LGG y GBM y por qué importa distinguirlos."),
    ("tab-problema", "Problema", "bi-exclamation-circle",
     "El coste de la secuenciación completa como cuello de botella."),
    ("tab-objetivos", "Objetivos", "bi-bullseye",
     "Qué se propone resolver el proyecto y con qué criterio."),
    ("tab-marco", "Marco teórico", "bi-journal-text",
     "Las variables del panel y las pruebas estadísticas."),
    ("tab-metodologia", "Metodología", "bi-clipboard-data",
     "Partición, control de calidad y contrastes aplicados."),
    ("tab-exploracion", "Exploración", "bi-sliders",
     "Cada variable por separado, con su ficha y sus pruebas."),
    ("tab-resultados", "Resultados", "bi-bar-chart-line",
     "El análisis exploratorio, variable a variable."),
    ("tab-limitaciones", "Limitaciones", "bi-exclamation-triangle",
     "Qué NO se puede concluir con este trabajo."),
    ("tab-conclusiones", "Conclusiones", "bi-check2-circle",
     "Hallazgos del EDA y siguiente paso del proyecto."),
    ("tab-documentacion", "Documentación", "bi-book",
     "El Jupyter Book, el repositorio y cómo compilar."),
]

ORDEN = [tab_id for tab_id, _, _, _ in PESTANAS]
ETIQUETAS = {tab_id: etiqueta for tab_id, etiqueta, _, _ in PESTANAS}
ICONOS = {tab_id: icono for tab_id, _, icono, _ in PESTANAS}
DESCRIPCIONES = {tab_id: texto for tab_id, _, _, texto in PESTANAS}

# Las etapas del trabajo. `en_guia` marca las que salen en la guía de la
# portada: allí no se lista la introducción (ya estás en ella) ni la
# documentación (tiene su propia tira al final), pero el índice desplegable sí
# las necesita, porque es la única forma de llegar a ellas desde otra pestaña.
BLOQUES = [
    {
        "numero": "",
        "titulo": "Inicio",
        "resumen": "",
        "pestanas": ["tab-introduccion"],
        "en_guia": False,
    },
    {
        "numero": "01",
        "titulo": "El contexto",
        "resumen": "Qué está en juego al distinguir los dos grados y por qué "
                   "hacerlo bien sale caro hoy.",
        "pestanas": ["tab-contexto", "tab-problema", "tab-objetivos"],
        "en_guia": True,
    },
    {
        "numero": "02",
        "titulo": "El método",
        "resumen": "Las variables del panel, la partición de los datos y las "
                   "pruebas que sostienen cada cifra del tablero.",
        "pestanas": ["tab-marco", "tab-metodologia", "tab-exploracion"],
        "en_guia": True,
    },
    {
        "numero": "03",
        "titulo": "Los hallazgos",
        "resumen": "El análisis variable a variable, lo que se puede concluir "
                   "de él y, sobre todo, lo que no.",
        "pestanas": ["tab-resultados", "tab-limitaciones", "tab-conclusiones"],
        "en_guia": True,
    },
    {
        "numero": "",
        "titulo": "Anexo",
        "resumen": "",
        "pestanas": ["tab-documentacion"],
        "en_guia": False,
    },
]

# Red de seguridad: si alguien añade una pestaña y se olvida de meterla en un
# bloque, el índice la perdería en silencio. Mejor que falle al arrancar.
_en_bloques = [tab_id for bloque in BLOQUES for tab_id in bloque["pestanas"]]
if sorted(_en_bloques) != sorted(ORDEN):
    faltan = set(ORDEN) - set(_en_bloques)
    sobran = set(_en_bloques) - set(ORDEN)
    raise RuntimeError(
        "BLOQUES y PESTANAS no coinciden en utils/navegacion.py. "
        f"Sin bloque: {sorted(faltan)}. Desconocidas: {sorted(sobran)}."
    )
