from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from config import settings
from src.ingesta.cargador_kaggle import (
    cargar_dataset_respaldo,
    cargar_dataset_tiktok,
    limpiar_valor_numerico,
)
from src.ingesta.cliente_ytdlp import descargar_audio_url
from src.ingesta.gestor_lotes import cargar_historial_fallos, ejecutar_ingesta_masiva
from src.ingesta.validador_medios import es_archivo_valido


def test_descargar_audio_url_inexistente_controla_excepcion(tmp_path: Path) -> None:
    url_rota = "https://www.tiktok.com/@usuario_inexistente_xyz/video/0000000000000000000"
    resultado = descargar_audio_url(url=url_rota, id_post="test_caido_01", directorio_destino=tmp_path)

    assert isinstance(resultado, dict)
    assert resultado["exito"] is False
    assert resultado["id_publicacion"] == "test_caido_01"
    assert resultado["ruta_audio"] is None
    assert isinstance(resultado["motivo_error"], str) and len(resultado["motivo_error"]) > 0


def test_validador_medios_rechaza_vacios_y_cortos(tmp_path: Path) -> None:
    archivo_vacio = tmp_path / "vacio.wav"
    archivo_vacio.touch()
    valido_vacio, _ = es_archivo_valido(archivo_vacio)
    assert valido_vacio is False

    sr = settings.SAMPLE_RATE
    onda_corta = np.sin(2.0 * np.pi * 440.0 * np.linspace(0.0, 1.0, sr, endpoint=False))
    archivo_corto = tmp_path / "corto.wav"
    sf.write(str(archivo_corto), onda_corta.astype(np.float32), sr)
    valido_corto, _ = es_archivo_valido(archivo_corto)
    assert valido_corto is False

    onda_valida = np.sin(2.0 * np.pi * 440.0 * np.linspace(0.0, 3.0, sr * 3, endpoint=False))
    archivo_ok = tmp_path / "valido.wav"
    sf.write(str(archivo_ok), onda_valida.astype(np.float32), sr)
    valido_ok, _ = es_archivo_valido(archivo_ok)
    assert valido_ok is True


def test_gestor_lotes_registra_fallos_y_activa_contingencia(tmp_path: Path) -> None:
    ruta_fallos = tmp_path / "fallos_descarga.csv"
    df_prueba = pd.DataFrame(
        [
            {"id_publicacion": "post_roto_1", "url_origen": "url_invalida_1"},
            {"id_publicacion": "post_roto_2", "url_origen": "htp://mal_formada"},
        ]
    )
    resumen = ejecutar_ingesta_masiva(
        df_urls=df_prueba,
        directorio_medios=tmp_path / "medios",
        ruta_fallos=ruta_fallos,
    )
    assert resumen["fallos"] == 2
    assert resumen["tasa_caidos"] == 1.0
    assert resumen["activar_respaldo"] is True

    historial = cargar_historial_fallos(ruta_fallos)
    assert len(historial) == 2
    assert set(historial.columns) == {"id_publicacion", "url_origen", "fecha_intento", "motivo_error"}


def test_homologacion_datasets_kaggle() -> None:
    assert limpiar_valor_numerico("1.2M") == 1200000.0
    assert limpiar_valor_numerico("45K") == 45000.0

    df_tk = cargar_dataset_tiktok()
    assert not df_tk.empty
    for col in settings.COLUMNAS_SOCIALES:
        assert col in df_tk.columns
        assert pd.api.types.is_float_dtype(df_tk[col])

    df_resp = cargar_dataset_respaldo(limite=50)
    assert len(df_resp) == 50
    for col in settings.COLUMNAS_SOCIALES + settings.COLUMNAS_ACUSTICAS:
        assert col in df_resp.columns
        assert np.isfinite(df_resp[col].to_numpy()).all()

