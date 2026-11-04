import shutil
import sys
import urllib.request

VERSION_MINIMA_PYTHON = (3, 10)
URL_OLLAMA_PING = "http://localhost:11434"


def auditar_entorno() -> dict[str, bool]:
    version_actual = sys.version_info[:2]
    python_valido = version_actual >= VERSION_MINIMA_PYTHON

    ffmpeg_disponible = shutil.which("ffmpeg") is not None

    ollama_activo = False
    try:
        with urllib.request.urlopen(URL_OLLAMA_PING, timeout=1.5) as respuesta:
            ollama_activo = respuesta.status == 200
    except Exception:
        ollama_activo = False

    return {
        "python": python_valido,
        "ffmpeg": ffmpeg_disponible,
        "ollama": ollama_activo,
    }


if __name__ == "__main__":
    resultado = auditar_entorno()
    print(f"Python >= 3.10: {'OK' if resultado['python'] else 'Falta actualizar'}")
    print(f"FFmpeg en PATH: {'OK' if resultado['ffmpeg'] else 'No detectado (modo respaldo activo)'}")
    print(f"Servidor Ollama: {'OK' if resultado['ollama'] else 'No detectado (modo contingencia activo)'}")
