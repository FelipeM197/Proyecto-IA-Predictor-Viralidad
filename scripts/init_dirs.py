"""
Script de Inicializacion Automatizada de Directorios del Proyecto.
Garantiza que todas las carpetas y archivos .gitkeep existan al clonar el repositorio.
"""

import sys
from pathlib import Path

# Agregar directorio raiz al path para importar settings
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DATA_RAW_DIR,
    DATA_MEDIA_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    LOGS_DIR,
    NOTEBOOKS_DIR,
    TESTS_DIR,
)

DIRECTORIOS = [
    DATA_RAW_DIR,
    DATA_MEDIA_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    LOGS_DIR,
    NOTEBOOKS_DIR,
    TESTS_DIR,
]

def inicializar_directorios() -> None:
    print("[+] Verificando e inicializando estructura de carpetas...")
    for d in DIRECTORIOS:
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        print(f"    [OK] {d.name} -> {d}")
    print("[+] Inicializacion completada con exito.")

if __name__ == "__main__":
    inicializar_directorios()
