"""Pruebas unitarias para las funciones de calculo y transformacion del target."""

import numpy as np
import pandas as pd
import pytest

from src.preprocesamiento import (
    calcular_engagement_rate,
    filtrar_vistas_minimas,
    invertir_target_log,
    transformar_target_log,
)


def test_filtrar_vistas_minimas():
    df = pd.DataFrame(
        {
            "views": [10, 99, 100, 500, None, 0],
            "likes": [1, 2, 3, 4, 5, 6],
        }
    )
    df_filtrado = filtrar_vistas_minimas(df, min_views=100)
    assert len(df_filtrado) == 2
    assert (df_filtrado["views"] >= 100).all()
    assert list(df_filtrado["views"]) == [100.0, 500.0]


def test_filtrar_vistas_minimas_sin_columna_views():
    df = pd.DataFrame({"otras_metricas": [1, 2, 3]})
    with pytest.raises(KeyError):
        filtrar_vistas_minimas(df)


def test_calcular_engagement_rate():
    df = pd.DataFrame(
        {
            "views": [1000, 500, 0, None],
            "likes": [100, 50, 10, 5],
            "comments": [20, 10, 2, 1],
            "shares": [10, 5, 1, 0],
        }
    )
    resultado = calcular_engagement_rate(df)

    assert "engagement_rate" in resultado.columns
    # Fila 0: (100 + 20 + 10) / 1000 = 130 / 1000 = 0.13
    assert np.isclose(resultado.loc[0, "engagement_rate"], 0.13)
    # Fila 1: (50 + 10 + 5) / 500 = 65 / 500 = 0.13
    assert np.isclose(resultado.loc[1, "engagement_rate"], 0.13)
    # Denominador 0 o nulo
    assert resultado.loc[2, "engagement_rate"] == 0.0
    assert resultado.loc[3, "engagement_rate"] == 0.0


def test_calcular_engagement_rate_columnas_faltantes():
    df = pd.DataFrame({"views": [100], "likes": [10]})
    with pytest.raises(KeyError):
        calcular_engagement_rate(df)


def test_transformar_target_log():
    df = pd.DataFrame({"engagement_rate": [0.0, 0.15, 1.0, 2.5]})
    resultado = transformar_target_log(df)

    assert "engagement_rate_log" in resultado.columns
    assert resultado.loc[0, "engagement_rate_log"] == 0.0
    assert np.isclose(resultado.loc[1, "engagement_rate_log"], np.log1p(0.15))
    assert np.isclose(resultado.loc[2, "engagement_rate_log"], np.log1p(1.0))
    assert np.isclose(resultado.loc[3, "engagement_rate_log"], np.log1p(2.5))


def test_invertir_target_log():
    valores_originales = np.array([0.0, 0.05, 0.18, 0.45, 1.2])
    valores_log = np.log1p(valores_originales)

    # Inversion de escalar
    assert np.isclose(invertir_target_log(valores_log[1]), valores_originales[1])

    # Inversion de numpy array
    invertidos_array = invertir_target_log(valores_log)
    assert np.allclose(invertidos_array, valores_originales)

    # Inversion de pandas Series
    serie_log = pd.Series(valores_log, name="target_log")
    invertidos_serie = invertir_target_log(serie_log)
    assert isinstance(invertidos_serie, pd.Series)
    assert np.allclose(invertidos_serie.values, valores_originales)
