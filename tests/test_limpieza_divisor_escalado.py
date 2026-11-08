"""Pruebas unitarias para limpieza de outliers, division estratificada y escalado."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocesamiento import (
    acotar_outliers_iqr,
    ajustar_escalador,
    cargar_escalador,
    dividir_datos,
    imputar_faltantes_mediana,
    transformar_features,
)


def test_acotar_outliers_iqr_basico():
    datos = {
        "metrica": [-100.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 500.0, np.inf, -np.inf],
    }
    df = pd.DataFrame(datos)

    df_acotado, limites = acotar_outliers_iqr(
        df,
        columnas=["metrica"],
        factor_iqr=1.5,
        p_inf=0.01,
        p_sup=0.99,
        retornar_limites=True,
    )

    # Verifica que inf y -inf se convirtieron a NaN y no explotan
    assert "metrica" in limites
    cota_inf, cota_sup = limites["metrica"]
    assert cota_inf < cota_sup

    # Los valores extremos deben haber quedado dentro de las cotas
    valores_validos = df_acotado["metrica"].dropna()
    assert valores_validos.min() >= cota_inf
    assert valores_validos.max() <= cota_sup


def test_acotar_outliers_iqr_reutilizacion_limites():
    df_train = pd.DataFrame({"valor": [1.0, 2.0, 3.0, 4.0, 100.0]})
    _, limites_train = acotar_outliers_iqr(
        df_train, columnas=["valor"], retornar_limites=True
    )

    df_test = pd.DataFrame({"valor": [0.0, 2.0, 200.0]})
    df_test_acotado = acotar_outliers_iqr(
        df_test, columnas=["valor"], limites=limites_train, retornar_limites=False
    )

    assert df_test_acotado["valor"].max() == limites_train["valor"][1]


def test_acotar_outliers_iqr_columnas_faltantes():
    df = pd.DataFrame({"col1": [1, 2, 3]})
    with pytest.raises(KeyError):
        acotar_outliers_iqr(df, columnas=["col_inexistente"])


def test_imputar_faltantes_mediana_calculo_y_reutilizacion():
    df_train = pd.DataFrame({"a": [10.0, 20.0, 30.0, np.nan], "b": [1.0, np.nan, 3.0, 5.0]})

    df_imputado, valores_imputacion = imputar_faltantes_mediana(
        df_train, columnas=["a", "b"]
    )

    # Mediana de 'a' sin nan es 20.0; mediana de 'b' sin nan es 3.0
    assert valores_imputacion["a"] == 20.0
    assert valores_imputacion["b"] == 3.0
    assert df_imputado["a"].isna().sum() == 0
    assert df_imputado["b"].isna().sum() == 0
    assert df_imputado.loc[3, "a"] == 20.0
    assert df_imputado.loc[1, "b"] == 3.0

    # Test con valores precomputados (evitar data leakage)
    df_test = pd.DataFrame({"a": [np.nan, 50.0], "b": [2.0, np.nan]})
    df_test_imputado, _ = imputar_faltantes_mediana(
        df_test, columnas=["a", "b"], valores_imputacion=valores_imputacion
    )
    assert df_test_imputado.loc[0, "a"] == 20.0
    assert df_test_imputado.loc[1, "b"] == 3.0


def test_imputar_faltantes_mediana_error_columna():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError):
        imputar_faltantes_mediana(df, columnas=["no_existe"])


def test_dividir_datos_estratificado():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "target": np.linspace(0.01, 10.0, n),
            "feature": np.random.randn(n),
        }
    )

    df_train, df_val, df_test = dividir_datos(
        df,
        col_target="target",
        prop_train=0.70,
        prop_val=0.15,
        prop_test=0.15,
        semilla=42,
        n_bins=4,
    )

    assert len(df_train) == 70
    assert len(df_val) == 15
    assert len(df_test) == 15
    assert len(df_train) + len(df_val) + len(df_test) == n

    # Verificar que las tres particiones cubren el rango dinamico del target
    assert df_train["target"].min() < 1.0 and df_train["target"].max() > 9.0
    assert df_val["target"].min() < 2.0 and df_val["target"].max() > 8.0
    assert df_test["target"].min() < 2.0 and df_test["target"].max() > 8.0


def test_dividir_datos_validacion_proporciones():
    df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError):
        dividir_datos(df, col_target="target", prop_train=0.5, prop_val=0.5, prop_test=0.5)


def test_dividir_datos_columna_inexistente():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        dividir_datos(df, col_target="target_no_existe")


def test_escalado_features_flujo_completo(tmp_path: Path):
    df_train = pd.DataFrame(
        {
            "f1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "f2": [100.0, 200.0, 300.0, 400.0, 500.0],
            "otra": ["a", "b", "c", "d", "e"],
        }
    )
    df_test = pd.DataFrame(
        {
            "f1": [15.0, 35.0],
            "f2": [150.0, 350.0],
            "otra": ["x", "y"],
        }
    )

    ruta_scaler = tmp_path / "escalador.joblib"
    columnas = ["f1", "f2"]

    escalador = ajustar_escalador(df_train, columnas, ruta_salida=ruta_scaler)
    assert isinstance(escalador, StandardScaler)
    assert ruta_scaler.exists()

    # Cargar y verificar
    escalador_cargado = cargar_escalador(ruta_scaler)
    np.testing.assert_allclose(escalador.mean_, escalador_cargado.mean_)
    np.testing.assert_allclose(escalador.scale_, escalador_cargado.scale_)

    # Transformar train
    df_train_esc = transformar_features(df_train, escalador, columnas)
    assert np.isclose(df_train_esc["f1"].mean(), 0.0, atol=1e-7)
    assert np.isclose(df_train_esc["f1"].std(ddof=0), 1.0, atol=1e-7)

    # Transformar test usando media y escala de train
    df_test_esc = transformar_features(df_test, escalador_cargado, columnas)
    media_f1_train = 30.0
    desv_f1_train = np.std([10.0, 20.0, 30.0, 40.0, 50.0])
    esperado_f1_test_0 = (15.0 - media_f1_train) / desv_f1_train
    assert np.isclose(df_test_esc.loc[0, "f1"], esperado_f1_test_0)

    # Verificar que la columna no escalada permanece intacta
    assert list(df_train_esc["otra"]) == ["a", "b", "c", "d", "e"]


def test_cargar_escalador_no_encontrado():
    with pytest.raises(FileNotFoundError):
        cargar_escalador("ruta/inexistente/scaler.joblib")
