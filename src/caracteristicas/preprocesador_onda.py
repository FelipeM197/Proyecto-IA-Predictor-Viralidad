from pathlib import Path
import librosa
import numpy as np
from config import settings


def normalizar_onda_pico(onda: np.ndarray) -> np.ndarray:
    if onda.size == 0:
        return np.zeros(settings.SAMPLE_RATE, dtype=np.float32)
    arreglo = np.nan_to_num(onda, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pico = float(np.max(np.abs(arreglo)))
    if pico > 1e-9:
        arreglo = arreglo / pico
    return np.clip(arreglo, -1.0, 1.0)


def cargar_y_estandarizar_onda(ruta: Path | str) -> tuple[np.ndarray, int]:
    ruta_archivo = Path(ruta)
    if not ruta_archivo.exists():
        raise FileNotFoundError(f"Archivo de audio no encontrado: {ruta_archivo}")

    onda, sr = librosa.load(
        str(ruta_archivo),
        sr=settings.SAMPLE_RATE,
        mono=True,
        duration=settings.MAX_AUDIO_DURATION_SEC,
    )
    max_muestras = int(settings.SAMPLE_RATE * settings.MAX_AUDIO_DURATION_SEC)
    if onda.shape[0] > max_muestras:
        onda = onda[:max_muestras]

    onda_normalizada = normalizar_onda_pico(onda)
    return onda_normalizada, int(sr)

