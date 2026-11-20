import json

import joblib
import numpy as np
import pandas as pd

from app.servicios.orquestador_inferencia import MotorInferencia
from config import settings
from src.modelos.explicabilidad_shap import explicar_prediccion_individual
from src.preprocesamiento import (
    acotar_outliers_iqr,
    calcular_engagement_rate,
    cargar_escalador,
    filtrar_vistas_minimas,
    imputar_faltantes_mediana,
    invertir_target_log,
    transformar_features,
    transformar_target_log,
)


def _vector_acustico_sintetico() -> dict[str, float]:
    return {
        "tempo_bpm": 124.0,
        "onset_strength_mean": 5.1,
        "onset_strength_std": 1.7,
        "beat_regularity": 0.85,
        "rms_mean": 0.24,
        "rms_std": 0.13,
        "dynamic_range": 0.54,
        "initial_silence_sec": 0.25,
        "silence_ratio": 0.012,
        "spectral_centroid": 3240.0,
        "spectral_bandwidth": 3140.0,
        "spectral_rolloff": 6075.0,
        "spectral_flatness": 0.135,
        "zero_crossing_rate": 0.101,
        "mfcc_1": -54.4,
        "mfcc_2": 4.8,
        "mfcc_3": 3.85,
        "mfcc_4": 2.28,
        "mfcc_5": 2.39,
        "mfcc_6": 2.32,
        "mfcc_7": 0.97,
        "mfcc_8": 1.57,
        "mfcc_9": 1.53,
        "mfcc_10": 0.93,
        "mfcc_11": 1.24,
        "mfcc_12": 1.45,
        "mfcc_13": 0.77,
        "chroma_stft_mean": 0.67,
    }


def test_calculo_e_inversion_target():
    vistas = [0, 50, 99, 100, 499, 500, 2500, np.nan]
    df_crudo = pd.DataFrame(
        {
            "views": vistas,
            "likes": [0, 2, 10, 15, 40, 50, 300, 5],
            "comments": [0, 0, 1, 2, 5, 10, 40, 1],
            "shares": [0, 0, 0, 1, 3, 5, 20, 0],
        }
    )

    df_filtrado = filtrar_vistas_minimas(df_crudo, min_views=settings.MIN_VIEWS)
    # Evalua contra settings.MIN_VIEWS para tolerar ajustes del umbral en defensa en vivo
    esperadas = {v for v in vistas if pd.notna(v) and v >= settings.MIN_VIEWS}
    assert set(df_filtrado["views"]) == esperadas
    assert not (df_filtrado["views"] < 100).any()
    assert not df_filtrado["views"].isna().any()

    df_target = calcular_engagement_rate(df_filtrado)
    df_log = transformar_target_log(df_target)
    recuperado = invertir_target_log(df_log[settings.COLUMNA_TARGET_LOG])
    assert np.allclose(recuperado, df_target[settings.COLUMNA_TARGET], atol=1e-6)

    # Incluye tasa nula para confirmar estabilidad en log1p(0) y expm1(0)
    df_tasas = pd.DataFrame({settings.COLUMNA_TARGET: [0.0, 0.015, 0.25, 1.8]})
    log_tasas = transformar_target_log(df_tasas)[settings.COLUMNA_TARGET_LOG].to_numpy()
    assert np.allclose(invertir_target_log(log_tasas), df_tasas[settings.COLUMNA_TARGET], atol=1e-6)


def test_limpieza_outliers_sin_nulos():
    rng = np.random.default_rng(settings.SEMILLA_ALEATORIA)
    datos = {
        col: rng.normal(loc=10.0, scale=2.0, size=24)
        for col in settings.COLUMNAS_ACUSTICAS
    }
    df_sucio = pd.DataFrame(datos)
    df_sucio.loc[0, "tempo_bpm"] = np.inf
    df_sucio.loc[1, "rms_mean"] = np.nan

    df_acotado = acotar_outliers_iqr(
        df_sucio,
        columnas=settings.COLUMNAS_ACUSTICAS,
        factor_iqr=settings.FACTOR_IQR,
        p_inf=settings.PERCENTIL_INFERIOR_OUTLIERS,
        p_sup=settings.PERCENTIL_SUPERIOR_OUTLIERS,
    )
    df_limpio, medianas = imputar_faltantes_mediana(
        df_acotado, columnas=settings.COLUMNAS_ACUSTICAS
    )

    matriz = df_limpio[settings.COLUMNAS_ACUSTICAS].to_numpy(dtype=float)
    assert matriz.shape == (24, len(settings.COLUMNAS_ACUSTICAS))
    assert len(medianas) == len(settings.COLUMNAS_ACUSTICAS)
    # acotar_outliers_iqr convierte +-inf en NaN antes de que la mediana los impute
    assert np.isfinite(matriz).all()


def _escalar_vector_sintetico(escalador) -> pd.DataFrame:
    df_entrada = pd.DataFrame(
        [_vector_acustico_sintetico()], columns=settings.COLUMNAS_ACUSTICAS
    )
    return transformar_features(
        df_entrada, escalador, settings.COLUMNAS_ACUSTICAS
    )[settings.COLUMNAS_ACUSTICAS]


def test_inferencia_vector_28_dimensiones_y_shap():
    assert settings.RUTA_ESCALADOR.is_file()
    assert settings.RUTA_MODELO_XGBOOST.is_file()

    escalador = cargar_escalador(settings.RUTA_ESCALADOR)
    modelo = joblib.load(settings.RUTA_MODELO_XGBOOST)
    df_escalado = _escalar_vector_sintetico(escalador)

    pred_log = float(modelo.predict(df_escalado)[0])
    pred_real = invertir_target_log(pred_log)

    assert isinstance(pred_real, float)
    assert np.isfinite(pred_real)
    assert pred_real > 0.0

    explicacion = explicar_prediccion_individual(modelo, df_escalado)
    contribuciones = explicacion["contribuciones"]
    favorables = explicacion["atributos_favorables"]
    perjudiciales = explicacion["atributos_perjudiciales"]

    assert len(contribuciones) == len(settings.COLUMNAS_ACUSTICAS)
    assert set(contribuciones.keys()) == set(settings.COLUMNAS_ACUSTICAS)
    assert all(np.isfinite(val) for val in contribuciones.values())

    n_positivos = sum(val > 0.0 for val in contribuciones.values())
    n_negativos = sum(val < 0.0 for val in contribuciones.values())
    assert len(favorables) == min(settings.TOP_K_ATRIBUTOS_SHAP, n_positivos)
    assert len(perjudiciales) == min(settings.TOP_K_ATRIBUTOS_SHAP, n_negativos)
    assert len(favorables) > 0 and len(perjudiciales) > 0
    assert all(item["impacto_shap"] > 0.0 for item in favorables)
    assert all(item["impacto_shap"] < 0.0 for item in perjudiciales)

    # Aditividad local exacta de TreeExplainer en escala logaritmica
    reconstruccion_log = float(explicacion["valor_base"]) + sum(contribuciones.values())
    assert np.isclose(reconstruccion_log, pred_log, atol=1e-4)


def test_artefactos_y_motor_inferencia():
    assert settings.RUTA_MODELO_MLP.is_file()
    assert settings.RUTA_REPORTE_MODELOS.is_file()
    assert settings.RUTA_UMBRALES_VIRALES.is_file()

    escalador = cargar_escalador(settings.RUTA_ESCALADOR)
    modelo_mlp = joblib.load(settings.RUTA_MODELO_MLP)
    df_escalado = _escalar_vector_sintetico(escalador)
    pred_mlp = modelo_mlp.predict(df_escalado)
    assert pred_mlp.shape == (1,)
    assert np.isfinite(pred_mlp).all()

    reporte = json.loads(settings.RUTA_REPORTE_MODELOS.read_text(encoding="utf-8"))
    metricas = reporte["metricas_test"]
    claves_modelos = {"ridge", "elasticnet", "mlp", "random_forest", "xgboost"}
    assert claves_modelos.issubset(metricas.keys())
    for clave in claves_modelos:
        assert all(np.isfinite(metricas[clave][m]) for m in ("MAE", "RMSE", "R2"))

    umbrales = json.loads(settings.RUTA_UMBRALES_VIRALES.read_text(encoding="utf-8"))
    assert len(umbrales) == len(settings.COLUMNAS_ACUSTICAS)
    assert set(umbrales.keys()) == set(settings.COLUMNAS_ACUSTICAS)
    for stats in umbrales.values():
        assert {"p25", "p50", "p75", "media"}.issubset(stats.keys())
        assert all(np.isfinite(stats[k]) for k in ("p25", "p50", "p75", "media"))
        assert stats["p25"] <= stats["p50"] <= stats["p75"]

    motor = MotorInferencia()
    assert motor.modo_demo is False
    assert motor.umbrales_oficiales is True
    assert motor.explicar is not None
