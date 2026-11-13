import librosa
import numpy as np
from config import settings


def extraer_caracteristicas_ritmo(
    onda: np.ndarray, sr: int = settings.SAMPLE_RATE
) -> dict[str, float]:
    if onda.size == 0 or float(np.max(np.abs(onda))) <= 1e-9:
        return {
            "tempo_bpm": 0.0,
            "onset_strength_mean": 0.0,
            "onset_strength_std": 0.0,
            "beat_regularity": 0.0,
        }

    envolvente_ataque = librosa.onset.onset_strength(
        y=onda, sr=sr, hop_length=settings.HOP_LENGTH
    )
    tempo, pulsos = librosa.beat.beat_track(
        onset_envelope=envolvente_ataque, sr=sr, hop_length=settings.HOP_LENGTH
    )
    tempo_val = float(np.atleast_1d(tempo)[0])

    fuerza_media = float(np.mean(envolvente_ataque)) if envolvente_ataque.size > 0 else 0.0
    fuerza_std = float(np.std(envolvente_ataque)) if envolvente_ataque.size > 0 else 0.0

    # Regularidad temporal en funcion del coeficiente de variacion entre pulsos
    if len(pulsos) >= 3:
        tiempos_pulsos = librosa.frames_to_time(pulsos, sr=sr, hop_length=settings.HOP_LENGTH)
        intervalos = np.diff(tiempos_pulsos)
        media_int = float(np.mean(intervalos))
        std_int = float(np.std(intervalos))
        if media_int > 1e-6:
            regularidad = float(np.clip(1.0 / (1.0 + (std_int / media_int)), 0.0, 1.0))
        else:
            regularidad = 0.0
    elif len(pulsos) == 2:
        regularidad = 0.5
    else:
        regularidad = 0.0

    return {
        "tempo_bpm": tempo_val,
        "onset_strength_mean": fuerza_media,
        "onset_strength_std": fuerza_std,
        "beat_regularity": regularidad,
    }

