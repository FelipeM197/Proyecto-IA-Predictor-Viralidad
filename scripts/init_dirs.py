from pathlib import Path
from config import settings


def inicializar_directorios(directorios: list[Path] | None = None) -> list[Path]:
    rutas = directorios if directorios is not None else settings.DIRECTORIOS_PROYECTO
    creados: list[Path] = []
    for ruta in rutas:
        ruta.mkdir(parents=True, exist_ok=True)
        marcador = ruta / ".gitkeep"
        if not marcador.exists():
            marcador.touch()
        creados.append(ruta)
    return creados


if __name__ == "__main__":
    rutas_listas = inicializar_directorios()
    print(f"Directorios listos: {len(rutas_listas)}")
