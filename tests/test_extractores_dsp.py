from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from config import settings
from src.caracteristicas.constructor_matriz import (
    extraer_vector_acustico,
    extraer_vector_acustico_desde_onda,
)
from src.caracteristicas.extractor_energia import extraer_caracteristicas_energia
from src.caracteristicas.preprocesador_onda import cargar_y_estandarizar_onda


def test_deteccion_silencio_inicial_tono_440hz() -> None:
    sr = settings.SAMPLE_RATE
    silencio_1s = np.zeros(sr, dtype=np.float32)
    t_tono = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
    tono_440hz = (0.8 * np.sin(2.0 * np.pi * 440.0 * t_tono)).astype(np.float32)
    senal = np.concatenate([silencio_1s, tono_440hz])

    metricas_energia = extraer_caracteristicas_energia(senal, sr=sr)
    diferencia = abs(metricas_energia["initial_silence_sec"] - 1.0)

    # Margen maximo permitido de 50 ms para el inicio del tono
    assert diferencia < 0.05, f"Silencio detectado: {metricas_energia['initial_silence_sec']}s"
    assert 0.20 <= metricas_energia["silence_ratio"] <= 0.30
    assert metricas_energia["rms_mean"] > 0.0


def test_vector_28_caracteristicas_finitas(tmp_path: Path) -> None:
    sr = settings.SAMPLE_RATE
    t = np.linspace(0.0, 4.0, sr * 4, endpoint=False)
    onda = (
        0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 880.0 * t)
    ).astype(np.float32)
    ruta_wav = tmp_path / "prueba_440.wav"
    sf.write(str(ruta_wav), onda, sr)

    vector = extraer_vector_acustico(ruta_wav)
    assert len(vector) == 28
    assert list(vector.keys()) == settings.COLUMNAS_ACUSTICAS
    for nombre, valor in vector.items():
        assert isinstance(valor, float), f"{nombre} no es float"
        assert np.isfinite(valor), f"{nombre} no es finito"


def test_cambio_duracion_maxima_en_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = settings.SAMPLE_RATE
    duracion_original = 20.0
    t = np.linspace(0.0, duracion_original, int(sr * duracion_original), endpoint=False)
    onda_larga = (0.7 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    ruta_wav = tmp_path / "audio_20s.wav"
    sf.write(str(ruta_wav), onda_larga, sr)

    # Recorte dinamico al reducir la ventana de analisis a 15 segundos
    monkeypatch.setattr(settings, "MAX_AUDIO_DURATION_SEC", 15.0)
    onda_recortada, sr_leido = cargar_y_estandarizar_onda(ruta_wav)

    assert sr_leido == sr
    assert len(onda_recortada) == int(15.0 * sr)
    assert float(np.max(np.abs(onda_recortada))) == pytest.approx(1.0, rel=1e-4)

    vector = extraer_vector_acustico_desde_onda(onda_recortada, sr=sr_leido)
    assert len(vector) == 28

