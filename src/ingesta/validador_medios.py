from pathlib import Path
import soundfile as sf
from config import settings

DURACION_MINIMA_SEG = float(getattr(settings, "MIN_AUDIO_DURATION_SEC", 2.0))


def es_archivo_valido(ruta_audio: Path | str) -> tuple[bool, str]:
    ruta = Path(ruta_audio)
    if not ruta.exists() or not ruta.is_file():
        return False, "El archivo no existe en la ruta indicada"

    if ruta.stat().st_size <= 0:
        return False, "El archivo tiene un tamano de cero bytes"

    try:
        info = sf.info(str(ruta))
    except Exception as err:
        return False, f"Cabecera de audio corrupta o formato no decodificable: {err}"

    if info.frames <= 0 or info.samplerate <= 0 or info.channels < 1:
        return False, "El archivo no contiene muestras decodificables"

    if info.duration < DURACION_MINIMA_SEG:
        return (
            False,
            f"Duracion insuficiente ({info.duration:.2f}s < {DURACION_MINIMA_SEG}s)",
        )

    return True, "Archivo de audio integro y decodificable"
