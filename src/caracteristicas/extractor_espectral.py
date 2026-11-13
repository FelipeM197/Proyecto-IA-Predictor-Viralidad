import librosa
import numpy as np
from config import settings

PROPORCION_ROLLOFF = settings.PERCENTIL_CORTE_VIRAL


def extraer_caracteristicas_espectrales(
    onda: np.ndarray, sr: int = settings.SAMPLE_RATE
) -> dict[str, float]:
    if onda.size == 0 or float(np.max(np.abs(onda))) <= 1e-9:
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flatness": 0.0,
            "zero_crossing_rate": 0.0,
        }

    # Piso de 1e-10 evita desbordamientos hacia -inf en tramas de silencio puro
    espectrograma = np.abs(
        librosa.stft(y=onda, n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH)
    ) + 1e-10

    centroide = librosa.feature.spectral_centroid(
        S=espectrograma, sr=sr, n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH
    )
    ancho_banda = librosa.feature.spectral_bandwidth(
        S=espectrograma, sr=sr, n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH
    )
    corte = librosa.feature.spectral_rolloff(
        S=espectrograma,
        sr=sr,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
        roll_percent=PROPORCION_ROLLOFF,
    )
    planitud = librosa.feature.spectral_flatness(
        S=espectrograma, n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH
    )
    cruces_cero = librosa.feature.zero_crossing_rate(
        y=onda, frame_length=settings.N_FFT, hop_length=settings.HOP_LENGTH
    )

    return {
        "spectral_centroid": float(np.mean(centroide)),
        "spectral_bandwidth": float(np.mean(ancho_banda)),
        "spectral_rolloff": float(np.mean(corte)),
        "spectral_flatness": float(np.clip(np.mean(planitud), 0.0, 1.0)),
        "zero_crossing_rate": float(np.clip(np.mean(cruces_cero), 0.0, 1.0)),
    }
