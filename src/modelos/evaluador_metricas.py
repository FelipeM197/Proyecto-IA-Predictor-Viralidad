import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from config import settings
from config.logging_config import configurar_registro
from src.ingesta.cargador_kaggle import cargar_dataset_respaldo
from src.modelos.modelo_arboles import entrenar_random_forest
from src.modelos.modelo_baseline import entrenar_baselines
from src.modelos.modelo_red_neuronal import RedNeuronalAcustica
from src.modelos.optimizador_hiperparametros import optimizar_xgboost_cv
from src.preprocesamiento import (
    acotar_outliers_iqr,
    ajustar_escalador,
    calcular_engagement_rate,
    dividir_datos,
    filtrar_vistas_minimas,
    imputar_faltantes_mediana,
    transformar_features,
    transformar_target_log,
)

logger = configurar_registro("src.modelos.evaluador")


def calcular_metricas_regresion(y_real, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(root_mean_squared_error(y_real, y_pred)),
        "R2": float(r2_score(y_real, y_pred)),
    }


def evaluar_modelos(
    modelos: dict[str, object], x_test, y_test
) -> dict[str, dict[str, float]]:
    metricas: dict[str, dict[str, float]] = {}
    for nombre, modelo in modelos.items():
        predicciones = (
            modelo.predecir(x_test)
            if hasattr(modelo, "predecir")
            else modelo.predict(x_test)
        )
        metricas[nombre] = calcular_metricas_regresion(y_test, predicciones)
    return metricas


def _generar_dataset_sintetico(n_muestras: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(settings.SEMILLA_ALEATORIA)
    tabla: dict[str, object] = {
        "id_publicacion": [f"sintetico_{idx}" for idx in range(n_muestras)],
    }

    for col in settings.COLUMNAS_ACUSTICAS:
        if col == "tempo_bpm":
            tabla[col] = rng.uniform(75.0, 170.0, n_muestras)
        elif col == "initial_silence_sec":
            tabla[col] = rng.uniform(0.0, 1.5, n_muestras)
        elif col == "beat_regularity":
            tabla[col] = rng.uniform(0.25, 0.95, n_muestras)
        elif col == "rms_mean":
            tabla[col] = rng.uniform(0.03, 0.35, n_muestras)
        else:
            tabla[col] = rng.normal(0.0, 1.0, n_muestras)

    senal = (
        0.45 * np.asarray(tabla["beat_regularity"])
        + 0.30 * np.asarray(tabla["rms_mean"])
        - 0.35 * np.asarray(tabla["initial_silence_sec"])
        - 0.002 * np.abs(np.asarray(tabla["tempo_bpm"]) - 122.0)
    )
    tasa = np.clip(
        0.04 + 0.12 / (1.0 + np.exp(-3.0 * senal)) + rng.normal(0.0, 0.01, n_muestras),
        0.005,
        0.40,
    )
    vistas = np.round(rng.uniform(settings.MIN_VIEWS * 5, 500000.0, n_muestras))
    interacciones = vistas * tasa

    tabla["views"] = vistas.astype(np.float64)
    tabla["likes"] = np.round(interacciones * 0.80).astype(np.float64)
    tabla["comments"] = np.round(interacciones * 0.12).astype(np.float64)
    tabla["shares"] = np.round(interacciones * 0.08).astype(np.float64)
    return pd.DataFrame(tabla)


def cargar_datos_entrenamiento(
    ruta_parquet: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    ruta = settings.RUTA_MATRIZ_CRUDA if ruta_parquet is None else Path(ruta_parquet)
    if ruta.is_file():
        return pd.read_parquet(ruta), "matriz_acustica_cruda"

    try:
        return cargar_dataset_respaldo(limite=800), "respaldo_tiktok_csv"
    except FileNotFoundError:
        logger.warning("Sin parquet ni dataset de respaldo; se genera conjunto sintetico.")
        return _generar_dataset_sintetico(), "sintetico_determinista"


def preparar_particiones_modelado(
    df_crudo: pd.DataFrame, guardar_escalador: bool = False
) -> dict[str, object]:
    df_filtrado = filtrar_vistas_minimas(df_crudo, min_views=settings.MIN_VIEWS)
    df_target = calcular_engagement_rate(df_filtrado)
    df_log = transformar_target_log(df_target)

    df_train, df_val, df_test = dividir_datos(
        df_log,
        col_target=settings.COLUMNA_TARGET_LOG,
        prop_train=settings.PROPORCION_TRAIN,
        prop_val=settings.PROPORCION_VAL,
        prop_test=settings.PROPORCION_TEST,
        semilla=settings.SEMILLA_ALEATORIA,
        n_bins=settings.BINS_ESTRATIFICACION,
    )

    # Limites IQR, medianas y escalador se ajustan solo en entrenamiento para evitar fuga de datos
    df_train_iqr, limites_iqr = acotar_outliers_iqr(
        df_train,
        columnas=settings.COLUMNAS_ACUSTICAS,
        factor_iqr=settings.FACTOR_IQR,
        p_inf=settings.PERCENTIL_INFERIOR_OUTLIERS,
        p_sup=settings.PERCENTIL_SUPERIOR_OUTLIERS,
        retornar_limites=True,
    )
    df_val_iqr = acotar_outliers_iqr(
        df_val, columnas=settings.COLUMNAS_ACUSTICAS, limites=limites_iqr
    )
    df_test_iqr = acotar_outliers_iqr(
        df_test, columnas=settings.COLUMNAS_ACUSTICAS, limites=limites_iqr
    )

    df_train_limpio, medianas_train = imputar_faltantes_mediana(
        df_train_iqr, columnas=settings.COLUMNAS_ACUSTICAS
    )
    df_val_limpio, _ = imputar_faltantes_mediana(
        df_val_iqr,
        columnas=settings.COLUMNAS_ACUSTICAS,
        valores_imputacion=medianas_train,
    )
    df_test_limpio, _ = imputar_faltantes_mediana(
        df_test_iqr,
        columnas=settings.COLUMNAS_ACUSTICAS,
        valores_imputacion=medianas_train,
    )

    ruta_escalador = settings.RUTA_ESCALADOR if guardar_escalador else None
    escalador = ajustar_escalador(
        df_train_limpio,
        columnas_features=settings.COLUMNAS_ACUSTICAS,
        ruta_salida=ruta_escalador,
    )

    df_train_esc = transformar_features(
        df_train_limpio, escalador, settings.COLUMNAS_ACUSTICAS
    )
    df_val_esc = transformar_features(
        df_val_limpio, escalador, settings.COLUMNAS_ACUSTICAS
    )
    df_test_esc = transformar_features(
        df_test_limpio, escalador, settings.COLUMNAS_ACUSTICAS
    )

    return {
        "df_train_limpio": df_train_limpio,
        "df_train_esc": df_train_esc,
        "df_val_esc": df_val_esc,
        "df_test_esc": df_test_esc,
        "escalador": escalador,
    }


def guardar_reporte_comparativo(
    reporte: dict[str, object],
    ruta_salida: str | Path = settings.RUTA_REPORTE_MODELOS,
) -> Path:
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ruta


def ejecutar_pipeline_entrenamiento(
    ruta_parquet: str | Path | None = None,
    guardar_artefactos: bool = True,
) -> dict[str, object]:
    df_crudo, fuente_datos = cargar_datos_entrenamiento(ruta_parquet=ruta_parquet)
    particiones = preparar_particiones_modelado(
        df_crudo, guardar_escalador=guardar_artefactos
    )

    cols = settings.COLUMNAS_ACUSTICAS
    col_target = settings.COLUMNA_TARGET_LOG

    x_train = particiones["df_train_esc"][cols]
    y_train = particiones["df_train_esc"][col_target].to_numpy(dtype=float)
    x_val = particiones["df_val_esc"][cols]
    y_val = particiones["df_val_esc"][col_target].to_numpy(dtype=float)
    x_test = particiones["df_test_esc"][cols]
    y_test = particiones["df_test_esc"][col_target].to_numpy(dtype=float)

    baselines = entrenar_baselines(x_train, y_train)
    red_mlp = RedNeuronalAcustica()
    red_mlp.entrenar(x_train, y_train, x_val, y_val)
    modelo_rf = entrenar_random_forest(x_train, y_train)
    modelo_xgb, resumen_cv = optimizar_xgboost_cv(x_train, y_train)

    modelos_evaluacion = {
        "ridge": baselines["ridge"],
        "elasticnet": baselines["elasticnet"],
        "mlp": red_mlp.modelo,
        "random_forest": modelo_rf,
        "xgboost": modelo_xgb,
    }
    metricas_test = evaluar_modelos(modelos_evaluacion, x_test, y_test)
    mejor_modelo_test = min(
        metricas_test, key=lambda nombre: metricas_test[nombre]["RMSE"]
    )

    # Selecciona dinamicamente el modelo con menor RMSE en test para produccion
    reporte: dict[str, object] = {
        "modelo_seleccionado_produccion": mejor_modelo_test,
        "mejor_modelo_test": mejor_modelo_test,
        "metricas_test": metricas_test,
        "optimizacion_cv": resumen_cv,
        "fuente_datos": fuente_datos,
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
    }

    if guardar_artefactos:
        red_mlp.guardar(settings.RUTA_MODELO_MLP)
        settings.RUTA_MODELO_XGBOOST.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelos_evaluacion[mejor_modelo_test], settings.RUTA_MODELO_XGBOOST)
        guardar_reporte_comparativo(reporte, settings.RUTA_REPORTE_MODELOS)

    logger.info(
        "Entrenamiento finalizado (%s) | n=%d/%d/%d | produccion=%s | RMSE=%.4f | R2=%.4f",
        fuente_datos,
        len(x_train),
        len(x_val),
        len(x_test),
        mejor_modelo_test,
        metricas_test[mejor_modelo_test]["RMSE"],
        metricas_test[mejor_modelo_test]["R2"],
    )
    return reporte


if __name__ == "__main__":
    ejecutar_pipeline_entrenamiento(guardar_artefactos=True)
