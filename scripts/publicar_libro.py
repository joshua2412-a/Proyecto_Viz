"""
Compila el Jupyter Book de jbook/ y lo publica en la rama gh-pages de este
mismo repositorio.

El sitio publicado (https://joshua2412-a.github.io/Proyecto_Viz/) es el
resultado de `jupyter-book build`, no la fuente: por eso la rama gh-pages solo
contiene HTML, mientras la fuente (notebooks, _config.yml, _toc.yml) vive en
`main`, dentro de jbook/. Las dos ramas del mismo repositorio, cada una con su
contenido.

Este script no depende de ghp-import: clona la rama gh-pages en una carpeta
temporal, reemplaza su contenido por el build nuevo y hace push. Si la rama no
existe todavía, la crea.

Uso:
    python scripts/publicar_libro.py                 # compila y publica
    python scripts/publicar_libro.py --solo-build    # compila y no publica
    python scripts/publicar_libro.py --solo-publicar # publica el build existente
    python scripts/publicar_libro.py --forzar-ejecucion  # recalcula notebooks

Requisitos:
    pip install -r jbook/requirements.txt
    git con identidad configurada (user.name y user.email) y acceso de
    escritura al repositorio.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LIBRO_DIR = BASE_DIR / "jbook"
BUILD_DIR = LIBRO_DIR / "_build" / "html"

REPO_LIBRO = "https://github.com/joshua2412-a/Proyecto_Viz.git"
RAMA = "gh-pages"
URL_SITIO = "https://joshua2412-a.github.io/Proyecto_Viz/"


def ejecutar(comando: list[str], cwd: Path | None = None) -> None:
    """Lanza un comando y aborta el script si falla."""
    print(f"  $ {' '.join(comando)}")
    resultado = subprocess.run(comando, cwd=cwd)
    if resultado.returncode != 0:
        sys.exit(f"\n[error] Falló: {' '.join(comando)}")


def compilar(forzar_ejecucion: bool) -> None:
    """Compila el libro con jupyter-book."""
    if shutil.which("jupyter-book") is None:
        sys.exit(
            "[error] No se encontró jupyter-book.\n"
            "        Instálalo con: pip install -r jbook/requirements.txt"
        )

    print("\n[1/2] Compilando el libro...")
    comando = ["jupyter-book", "build", str(LIBRO_DIR)]
    if forzar_ejecucion:
        # Recalcula todas las salidas: necesita dataset/TCGA_InfoWithGrade.csv
        comando += ["--all"]
        ejecutar(["jupyter-book", "config", "sphinx", str(LIBRO_DIR)])
    ejecutar(comando)

    if not (BUILD_DIR / "index.html").exists():
        sys.exit(f"[error] El build no generó {BUILD_DIR / 'index.html'}")
    print(f"  Libro compilado en {BUILD_DIR}")


def rama_existe() -> bool:
    """¿Existe ya la rama de publicación en el remoto?"""
    resultado = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", REPO_LIBRO, RAMA],
        capture_output=True, text=True,
    )
    if resultado.returncode == 2:      # la rama no existe (pero el repo sí)
        return False
    if resultado.returncode != 0:      # no hay red, no hay permisos, no hay repo
        sys.exit(
            f"\n[error] No se pudo consultar {REPO_LIBRO}:\n"
            f"        {resultado.stderr.strip()}"
        )
    return True


def comprobar_identidad_git() -> None:
    """Avisa antes de empezar si git no sabe quién hace el commit."""
    faltantes = [
        clave for clave in ("user.name", "user.email")
        if not subprocess.run(
            ["git", "config", "--get", clave], capture_output=True, text=True
        ).stdout.strip()
    ]
    if faltantes:
        sys.exit(
            "[error] git no tiene configurada tu identidad "
            f"({', '.join(faltantes)}), así que el commit fallaría.\n"
            '        git config --global user.name "Tu Nombre"\n'
            '        git config --global user.email "tu@correo.com"'
        )


def publicar() -> None:
    """Sustituye el contenido de la rama de publicación por el build generado.

    Si la rama todavía no existe en el remoto (primera publicación), la crea
    como rama huérfana: sin historial compartido con `main`, que es justo lo que
    GitHub Pages espera de una rama de publicación.
    """
    comprobar_identidad_git()
    existe = rama_existe()
    print(f"\n[2/2] Publicando en {REPO_LIBRO} ({RAMA})...")

    with tempfile.TemporaryDirectory() as temporal:
        clon = Path(temporal) / RAMA

        if existe:
            ejecutar(
                ["git", "clone", "--depth", "1", "--branch", RAMA,
                 "--single-branch", REPO_LIBRO, str(clon)]
            )
            # Borra todo el contenido anterior menos el historial de git
            for elemento in clon.iterdir():
                if elemento.name == ".git":
                    continue
                shutil.rmtree(elemento) if elemento.is_dir() else elemento.unlink()
        else:
            print(f"  La rama {RAMA} no existe todavía: se creará desde cero.")
            clon.mkdir(parents=True)
            ejecutar(["git", "init", "-b", RAMA], cwd=clon)
            ejecutar(["git", "remote", "add", "origin", REPO_LIBRO], cwd=clon)

        # Copia el build nuevo
        shutil.copytree(BUILD_DIR, clon, dirs_exist_ok=True)

        # Sin .nojekyll, GitHub Pages ignora las carpetas que empiezan por "_"
        (clon / ".nojekyll").touch()

        ejecutar(["git", "add", "--all"], cwd=clon)
        estado = subprocess.run(
            ["git", "status", "--porcelain"], cwd=clon,
            capture_output=True, text=True,
        ).stdout.strip()

        if not estado:
            print("  Sin cambios respecto a lo publicado. Nada que hacer.")
            return

        ejecutar(["git", "commit", "-m", "Actualizar libro compilado"], cwd=clon)
        ejecutar(["git", "push", "origin", RAMA], cwd=clon)

    print(f"\n  Publicado: {URL_SITIO}")
    if not existe:
        print(
            "\n  PRIMERA PUBLICACIÓN: activa GitHub Pages una sola vez en\n"
            f"  {REPO_LIBRO.removesuffix('.git')}/settings/pages\n"
            f"  -> Source: Deploy from a branch -> Branch: {RAMA} / (root) -> Save\n"
            "  El sitio tarda un par de minutos en quedar disponible."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solo-build", action="store_true",
        help="compila el libro sin publicarlo en gh-pages",
    )
    parser.add_argument(
        "--solo-publicar", action="store_true",
        help="publica el build que ya existe en jbook/_build/html, sin recompilar "
             "(útil para reintentar cuando falla la conexión con GitHub)",
    )
    parser.add_argument(
        "--forzar-ejecucion", action="store_true",
        help="reejecuta los notebooks en vez de usar las salidas guardadas",
    )
    argumentos = parser.parse_args()

    if argumentos.solo_publicar:
        if not (BUILD_DIR / "index.html").exists():
            sys.exit(
                f"[error] No hay un build previo en {BUILD_DIR}.\n"
                "        Compila primero: python scripts/publicar_libro.py --solo-build"
            )
        print(f"\n[1/1] Reutilizando el build de {BUILD_DIR}")
        publicar()
        return

    compilar(argumentos.forzar_ejecucion)
    if argumentos.solo_build:
        print("\nBuild listo. Ábrelo con:")
        print(f"  {BUILD_DIR / 'index.html'}")
        return
    publicar()


if __name__ == "__main__":
    main()
