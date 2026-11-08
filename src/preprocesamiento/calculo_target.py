"""Calculo y transformacion matematica de la variable objetivo de engagement.

Este modulo implementa las funciones para filtrar publicaciones con baja
exposicion, calcular la tasa de engagement relativa sobre las visualizaciones y
aplicar las transformaciones directas e inversas en escala logaritmica.
"""

from typing import Union
import numpy as np
import pandas as pd


def filtrar_vistas_minimas(df: pd.DataFrame, min_views: int = 100) -> pd.DataFrame:
    """Descarta registros con visualizaciones inferiores al umbral minimo especificado.

    Eliminar registros con un volumen muy bajo de vistas previene ruido estadistico
    y valores de ratio desproporcionados causados por tamanos muestrales reducidos.

    Args:
        df: DataFrame de entrada con datos de publicaciones. Requiere la columna 'views'.
        min_views: Conteo minimo de visualizaciones requerido. Por defecto es 100.

    Returns:
        Copia limpia e independiente del DataFrame conteniendo unicamente
        los registros que satisfacen la condicion de vistas minimas.
    """
    if "views" not in df.columns:
        raise KeyError("La columna 'views' no esta presente en el DataFrame.")

    condicion = df["views"].notna() & (df["views"] >= min_views)
    df_filtrado = df.loc[condicion].copy().reset_index(drop=True)
    return df_filtrado


def calcular_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la tasa relativa de engagement por publicacion.

    Aplica la formulacion estandar de tasa de interaccion:
        engagement_rate = (likes + comments + shares) / views

    Gestiona casos limite de denominadores nulos, iguales a cero o valores
    faltantes en las metricas de interaccion, asignando un ratio de 0.0
    para evitar indeterminaciones, divisiones por cero o infinitos.

    Args:
        df: DataFrame que contiene las columnas 'views', 'likes', 'comments' y 'shares'.

    Returns:
        Copia del DataFrame original agregando la columna 'engagement_rate'.
    """
    columnas_necesarias = ["views", "likes", "comments", "shares"]
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if columnas_faltantes:
        raise KeyError(
            f"Faltan columnas requeridas para el calculo del engagement: {columnas_faltantes}"
        )

    df_resultado = df.copy()

    interacciones = (
        df_resultado["likes"].fillna(0)
        + df_resultado["comments"].fillna(0)
        + df_resultado["shares"].fillna(0)
    )

    vistas = df_resultado["views"]
    mascara_vistas_validas = vistas.notna() & (vistas > 0)

    tasa = pd.Series(0.0, index=df_resultado.index, dtype=float)
    tasa.loc[mascara_vistas_validas] = (
        interacciones.loc[mascara_vistas_validas]
        / vistas.loc[mascara_vistas_validas]
    )

    df_resultado["engagement_rate"] = tasa.fillna(0.0).astype(float)
    return df_resultado


def transformar_target_log(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica una transformacion logaritmica natural ln(1 + x) sobre la tasa de engagement.

    En el analisis de redes sociales y plataformas de contenido, la variable
    'engagement_rate' exhibe una distribucion fuertemente sesgada a la derecha
    (asimetria positiva extrema con colas pesadas), donde la mayoria de los
    elementos tienen baja interaccion y unos pocos fenomenos virales alcanzan
    magnitudes atipicas.

    Se adopta formalmente la transformacion ln(1 + x) mediante numpy.log1p
    por los siguientes fundamentos tecnicos:
    1. Estabilidad matematica en cero: Para publicaciones sin ninguna interaccion
       (engagement_rate = 0), ln(1 + 0) = ln(1) = 0, eludiendo la indeterminacion
       asintotica a menos infinito originada por el logaritmo estandar ln(0).
    2. Reduccion de la asimetria positiva extrema: Al comprimir monotonicamente
       el rango dinamico superior, contrae las distancias en la cola derecha,
       homogeneiza la varianza residual (homocedasticidad) y aproxima los residuos
       a una distribucion gaussiana, optimizando la convergencia en modelos lineales,
       redes neuronales y ensambles basados en gradiente.

    Args:
        df: DataFrame que contiene la columna 'engagement_rate'.

    Returns:
        Copia del DataFrame con la nueva columna 'engagement_rate_log'.
    """
    if "engagement_rate" not in df.columns:
        raise KeyError("La columna 'engagement_rate' no esta presente en el DataFrame.")

    df_resultado = df.copy()
    valores_no_negativos = df_resultado["engagement_rate"].clip(lower=0.0).fillna(0.0)
    df_resultado["engagement_rate_log"] = np.log1p(
        valores_no_negativos.to_numpy(dtype=float)
    )
    return df_resultado


def invertir_target_log(
    y_log: Union[np.ndarray, pd.Series, float, int]
) -> Union[np.ndarray, pd.Series, float]:
    """Revierte la transformacion logaritmica aplicando exp(y) - 1.

    Constituye la funcion inversa formal de ln(1 + x) calculada a traves de
    numpy.expm1:
        f^(-1)(y) = exp(y) - 1

    Permite restituir las estimaciones o predicciones del modelo desde la escala
    logaritmica comprimida a la escala natural y directamente interpretable del
    engagement rate, acotando el resultado inferiormente en 0.0 para prevenir
    imprecisiones numericas de punto flotante.

    Args:
        y_log: Valor escalar, array unidimensional o Serie de pandas en escala logaritmica.

    Returns:
        Valores reconvertidos a la escala original de tasa de engagement.
    """
    if isinstance(y_log, pd.Series):
        return pd.Series(
            np.clip(np.expm1(y_log.astype(float)), a_min=0.0, a_max=None),
            index=y_log.index,
            name=y_log.name,
        )
    elif isinstance(y_log, (np.ndarray, list, tuple)):
        arreglo = np.asarray(y_log, dtype=float)
        return np.clip(np.expm1(arreglo), a_min=0.0, a_max=None)
    else:
        valor_invertido = float(np.expm1(float(y_log)))
        return max(0.0, valor_invertido)
