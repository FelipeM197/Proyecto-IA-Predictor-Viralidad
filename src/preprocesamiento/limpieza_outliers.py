"""Tratamiento estadistico de valores atipicos e imputacion robusta.

Proporciona funciones para acotar outliers mediante rango intercuartilico (Tukey)
combinado con percentiles extremos (Winsorization) e imputacion por mediana,
asegurando consistencia entre entrenamiento y evaluacion sin fuga de datos.
"""

from typing import Union
import numpy as np
import pandas as pd


def acotar_outliers_iqr(
    df: pd.DataFrame,
    columnas: list[str],
    factor_iqr: float = 1.5,
    p_inf: float = 0.01,
    p_sup: float = 0.99,
    retornar_limites: bool = False,
    limites: dict[str, tuple[float, float]] | None = None,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[str, tuple[float, float]]]]:
    """Acota valores atipicos en columnas numericas usando criterios IQR y percentiles.

    Sustituye valores infinitos por nulos y aplica Winsorization combinando
    las cotas de Tukey con percentiles de seguridad para evitar distorsiones
    en distribuciones asimetricas.
    """
    columnas_faltantes = [col for col in columnas if col not in df.columns]
    if columnas_faltantes:
        raise KeyError(f"Columnas ausentes en el DataFrame: {columnas_faltantes}")

    df_acotado = df.copy()
    df_acotado[columnas] = df_acotado[columnas].replace([np.inf, -np.inf], np.nan)

    limites_calculados: dict[str, tuple[float, float]] = {}

    for col in columnas:
        if limites is not None and col in limites:
            cota_inf, cota_sup = limites[col]
        else:
            serie = df_acotado[col].dropna()
            if serie.empty:
                cota_inf, cota_sup = np.nan, np.nan
            else:
                q1 = float(serie.quantile(0.25))
                q3 = float(serie.quantile(0.75))
                iqr = q3 - q1

                cota_iqr_inf = q1 - factor_iqr * iqr
                cota_iqr_sup = q3 + factor_iqr * iqr

                percentil_inf = float(serie.quantile(p_inf))
                percentil_sup = float(serie.quantile(p_sup))

                cota_inf = max(cota_iqr_inf, percentil_inf)
                cota_sup = min(cota_iqr_sup, percentil_sup)

                if cota_inf > cota_sup:
                    cota_inf, cota_sup = cota_iqr_inf, cota_iqr_sup

        limites_calculados[col] = (cota_inf, cota_sup)

        if not np.isnan(cota_inf) and not np.isnan(cota_sup):
            df_acotado[col] = df_acotado[col].clip(lower=cota_inf, upper=cota_sup)

    if retornar_limites:
        return df_acotado, limites_calculados

    return df_acotado


def imputar_faltantes_mediana(
    df: pd.DataFrame,
    columnas: list[str],
    valores_imputacion: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Imputa valores ausentes en columnas numericas utilizando la mediana.

    Permite calcular las medianas sobre el conjunto de entrenamiento o
    reutilizar un diccionario precomputado para evitar fuga de informacion.
    """
    columnas_faltantes = [col for col in columnas if col not in df.columns]
    if columnas_faltantes:
        raise KeyError(f"Columnas ausentes en el DataFrame: {columnas_faltantes}")

    df_imputado = df.copy()
    valores_usados: dict[str, float] = {}

    if valores_imputacion is None:
        for col in columnas:
            mediana = float(df_imputado[col].median())
            valores_usados[col] = mediana
            df_imputado[col] = df_imputado[col].fillna(mediana)
    else:
        valores_usados = dict(valores_imputacion)
        for col in columnas:
            if col in valores_usados:
                df_imputado[col] = df_imputado[col].fillna(valores_usados[col])

    return df_imputado, valores_usados
