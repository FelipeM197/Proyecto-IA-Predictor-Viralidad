import librosa
import numpy as np
from config import settings

PERCENTIL_ALTO, PERCENTIL_BAJO = settings.PERCENTILES_UMBRALES[2] + 20, settings.PERCENTILES_UMBRALES[0] - 20


def extraer_caracteristicas_energia(
    onda: np.ndarray, sr: int = settings.SAMPLE_RATE
) -> dict[str, float]:
    total_muestras = len(onda)
    if total_muestras == 0:
        return {
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "dynamic_range": 0.0,
            "initial_silence_sec": 0.0,
            "silence_ratio": 1.0,
        }

    rms = librosa.feature.rms(
        y=onda, frame_length=settings.N_FFT, hop_length=settings.HOP_LENGTH
    )[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    p_alto = float(np.percentile(rms, PERCENTIL_ALTO))
    p_bajo = float(np.percentile(rms, PERCENTIL_BAJO))
    rango_dinamico = max(p_alto - p_bajo, 0.0)

    duracion_total = min(float(total_muestras / sr), settings.MAX_AUDIO_DURATION_SEC)
    if float(np.max(np.abs(onda))) <= 1e-7:
        return {
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "dynamic_range": 0.0,
            "initial_silence_sec": duracion_total,
            "silence_ratio": 1.0,
        }

    intervalos = librosa.effects.split(
        y=onda,
        top_db=settings.SILENCE_THRESHOLD_DB,
        frame_length=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
    )

    if len(intervalos) == 0:
        silencio_inicial = duracion_total
        proporcion_silencio = 1.0
    else:
        silencio_inicial = float(
            np.clip(intervalos[0, 0] / sr, 0.0, settings.MAX_AUDIO_DURATION_SEC)
        )
        muestras_activas = int(np.sum(intervalos[:, 1] - intervalos[:, 0]))
        proporcion_silencio = float(
            np.clip(1.0 - (muestras_activas / max(total_muestras, 1)), 0.0, 1.0)
        )

    return {
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "dynamic_range": rango_dinamico,
        "initial_silence_sec": silencio_inicial,
        "silence_ratio": proporcion_silencio,
    }
