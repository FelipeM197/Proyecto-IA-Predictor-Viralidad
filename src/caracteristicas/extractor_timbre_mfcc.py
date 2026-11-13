import librosa
import numpy as np
from config import settings

CANTIDAD_MFCC = sum(1 for col in settings.COLUMNAS_ACUSTICAS if col.startswith("mfcc_"))


def extraer_caracteristicas_timbre(
    onda: np.ndarray, sr: int = settings.SAMPLE_RATE
) -> dict[str, float]:
    resultado: dict[str, float] = {}
    if onda.size == 0 or float(np.max(np.abs(onda))) <= 1e-9:
        for idx in range(1, CANTIDAD_MFCC + 1):
            resultado[f"mfcc_{idx}"] = 0.0
        resultado["chroma_stft_mean"] = 0.0
        return resultado

    matriz_mfcc = librosa.feature.mfcc(
        y=onda,
        sr=sr,
        n_mfcc=CANTIDAD_MFCC,
        n_fft=settings.N_FFT,
        hop_length=settings.HOP_LENGTH,
    )
    medias_mfcc = np.mean(matriz_mfcc, axis=1)
    for idx in range(CANTIDAD_MFCC):
        valor = float(medias_mfcc[idx])
        resultado[f"mfcc_{idx + 1}"] = valor if np.isfinite(valor) else 0.0

    espectrograma_potencia = (
        np.abs(librosa.stft(y=onda, n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH)) ** 2
        + 1e-10
    )
    chroma = librosa.feature.chroma_stft(
        S=espectrograma_potencia, sr=sr, n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH
    )
    chroma_val = float(np.mean(chroma))
    resultado["chroma_stft_mean"] = chroma_val if np.isfinite(chroma_val) else 0.0
    return resultado
