"""Pruebas unitarias y de integracion para el modulo de preprocesamiento.

Valida el filtrado de vistas minimas, la continuidad y reversibilidad de la
transformacion logaritmica del target, la particion estratificada sin solapamiento,
el tratamiento de outliers mediante acotamiento IQR con imputacion por mediana,
y la ausencia estricta de fuga de informacion (data leakage) en el escalado Z-score.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocesamiento import (
    acotar_outliers_iqr,
    ajustar_escalador,
    calcular_engagement_rate,
    cargar_escalador,
    dividir_datos,
    filtrar_vistas_minimas,
    imputar_faltantes_mediana,
    invertir_target_log,
    transformar_features,
    transformar_target_log,
)


# -----------------------------------------------------------------------------
# Pruebas de filtrado y calculo del target
# -----------------------------------------------------------------------------


def test_filtrado_vistas_minimas_umbral_y_descarte():
    """Verifica el descarte estricto de views < 100 y la preservacion de views >= 100."""
    df_prueba = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "views": [0, 50, 99, 100, 101, 2500, np.nan],
            "likes": [0, 5, 10, 20, 30, 400, 15],
        }
    )

    df_resultado = filtrar_vistas_minimas(df_prueba, min_views=100)

    # Solo deben permanecer registros con views >= 100
    assert len(df_resultado) == 3
    assert set(df_resultado["id"]) == {4, 5, 6}
    assert (df_resultado["views"] >= 100).all()
    assert not df_resultado["views"].isna().any()

    # Los valores originales de las columnas preservadas no deben alterarse
    assert df_resultado.loc[df_resultado["id"] == 4, "views"].iloc[0] == 100
    assert df_resultado.loc[df_resultado["id"] == 5, "views"].iloc[0] == 101
    assert df_resultado.loc[df_resultado["id"] == 6, "views"].iloc[0] == 2500


def test_filtrado_vistas_minimas_columna_faltante():
    """Verifica que se lance KeyError si el DataFrame carece de la columna 'views'."""
    df_sin_views = pd.DataFrame({"reproducciones": [150, 200]})
    with pytest.raises(KeyError):
        filtrar_vistas_minimas(df_sin_views, min_views=100)


def test_continuidad_y_reversibilidad_transformacion_logaritmica():
    """Evalua continuidad en cero, monotonia estricta y reversibilidad exacta de ln(1+x)."""
    # 1. Continuidad y anulacion identica en el origen
    df_cero = pd.DataFrame({"engagement_rate": [0.0]})
    df_cero_trans = transformar_target_log(df_cero)
    assert df_cero_trans["engagement_rate_log"].iloc[0] == 0.0

    # 2. Monotonia estricta sobre valores no negativos crecientes
    tasas = np.array([0.0, 1e-6, 0.01, 0.05, 0.25, 0.8, 1.5, 5.0])
    df_tasas = pd.DataFrame({"engagement_rate": tasas})
    df_tasas_trans = transformar_target_log(df_tasas)
    valores_log = df_tasas_trans["engagement_rate_log"].to_numpy()

    diferencias = np.diff(valores_log)
    assert (diferencias > 0).all()

    # 3. Reversibilidad exacta mediante expm1
    tasas_recuperadas = invertir_target_log(valores_log)
    assert np.allclose(tasas_recuperadas, tasas, atol=1e-12)

    # 4. Reversibilidad soportando pd.Series y escalares float
    serie_log = pd.Series(valores_log, name="target_log")
    serie_recuperada = invertir_target_log(serie_log)
    assert isinstance(serie_recuperada, pd.Series)
    assert np.allclose(serie_recuperada.values, tasas, atol=1e-12)

    escalar_log = float(np.log1p(0.35))
    escalar_recuperado = invertir_target_log(escalar_log)
    assert np.isclose(escalar_recuperado, 0.35, atol=1e-12)


def test_calculo_engagement_rate_robustez_division_cero():
    """Verifica que el calculo del ratio maneje adecuadamente ceros y nulos sin infinitos."""
    df_interacciones = pd.DataFrame(
        {
            "views": [500, 1000, 0, np.nan],
            "likes": [25, 50, 10, 5],
            "comments": [5, 10, 2, np.nan],
            "shares": [0, 5, 0, 1],
        }
    )

    df_resultado = calcular_engagement_rate(df_interacciones)
    assert "engagement_rate" in df_resultado.columns

    # Registro 0: (25 + 5 + 0) / 500 = 30 / 500 = 0.06
    assert np.isclose(df_resultado.loc[0, "engagement_rate"], 0.06)
    # Registro 1: (50 + 10 + 5) / 1000 = 65 / 1000 = 0.065
    assert np.isclose(df_resultado.loc[1, "engagement_rate"], 0.065)
    # Vistas 0 y NaN no deben producir inf ni nan, sino 0.0
    assert df_resultado.loc[2, "engagement_rate"] == 0.0
    assert df_resultado.loc[3, "engagement_rate"] == 0.0


# -----------------------------------------------------------------------------
# Pruebas de particion estratificada y consistencia dimensional
# -----------------------------------------------------------------------------


def test_invarianza_y_consistencia_dimensiones_particion_70_15_15():
    """Valida dimensiones exactas, suma consistente y disyuncion estricta de indices."""
    n_total = 200
    rng = np.random.default_rng(42)

    df_completo = pd.DataFrame(
        {
            "id_unico": np.arange(n_total),
            "target": rng.uniform(0.001, 1.5, size=n_total),
            "feature_x": rng.normal(10.0, 2.5, size=n_total),
        }
    )

    df_train, df_val, df_test = dividir_datos(
        df_completo,
        col_target="target",
        prop_train=0.70,
        prop_val=0.15,
        prop_test=0.15,
        semilla=42,
        n_bins=5,
    )

    # 1. Longitudes esperadas exactas
    assert len(df_train) == 140
    assert len(df_val) == 30
    assert len(df_test) == 30

    # 2. Invarianza de conservacion de registros
    assert len(df_train) + len(df_val) + len(df_test) == n_total

    # 3. Conjuntos de identificadores mutuamente disjuntos (sin solapamiento)
    ids_train = set(df_train["id_unico"])
    ids_val = set(df_val["id_unico"])
    ids_test = set(df_test["id_unico"])

    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)
    assert (ids_train | ids_val | ids_test) == set(df_completo["id_unico"])

    # 4. Preservacion de todas las columnas originales
    columnas_esperadas = list(df_completo.columns)
    assert list(df_train.columns) == columnas_esperadas
    assert list(df_val.columns) == columnas_esperadas
    assert list(df_test.columns) == columnas_esperadas


def test_dividir_datos_validaciones_parametros():
    """Verifica el rechazo de proporciones inconsistentes o target con valores nulos."""
    df = pd.DataFrame({"target": [0.1, 0.2, 0.3, 0.4]})

    # Suma de proporciones distinta de 1.0
    with pytest.raises(ValueError):
        dividir_datos(df, col_target="target", prop_train=0.8, prop_val=0.1, prop_test=0.2)

    # Presencia de NaN en target
    df_nulos = pd.DataFrame({"target": [0.1, np.nan, 0.3, 0.4]})
    with pytest.raises(ValueError):
        dividir_datos(df_nulos, col_target="target")


# -----------------------------------------------------------------------------
# Pruebas de tratamiento de outliers e imputacion
# -----------------------------------------------------------------------------


def test_preservacion_mediana_imputacion_sin_fuga():
    """Comprueba que la imputacion por mediana calcule en train y replique en test sin fuga."""
    df_train = pd.DataFrame(
        {
            "metrica_a": [10.0, 20.0, 30.0, 40.0, np.nan],
            "metrica_b": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    df_train_imp, valores_mediana = imputar_faltantes_mediana(
        df_train, columnas=["metrica_a", "metrica_b"]
    )

    # Mediana de metrica_a [10, 20, 30, 40] es 25.0
    # Mediana de metrica_b [1, 2, 4, 5] es 3.0
    assert valores_mediana["metrica_a"] == 25.0
    assert valores_mediana["metrica_b"] == 3.0

    # Valores nulos en train sustituidos exactamente por las medianas aprendidas
    assert df_train_imp.loc[4, "metrica_a"] == 25.0
    assert df_train_imp.loc[2, "metrica_b"] == 3.0

    # Preservacion estricta de valores preexistentes no nulos
    assert df_train_imp.loc[0, "metrica_a"] == 10.0
    assert df_train_imp.loc[1, "metrica_b"] == 2.0

    # Imputacion en test utilizando unicamente el vector precalculado de train
    df_test = pd.DataFrame(
        {
            "metrica_a": [np.nan, 100.0],
            "metrica_b": [50.0, np.nan],
        }
    )

    df_test_imp, _ = imputar_faltantes_mediana(
        df_test,
        columnas=["metrica_a", "metrica_b"],
        valores_imputacion=valores_mediana,
    )

    # Si hubiera data leakage con test, la mediana de 'metrica_a' seria 100.0, pero debe ser 25.0
    assert df_test_imp.loc[0, "metrica_a"] == 25.0
    assert df_test_imp.loc[1, "metrica_b"] == 3.0
    assert df_test_imp.loc[1, "metrica_a"] == 100.0


def test_comportamiento_clip_acotamiento_iqr():
    """Valida el acotamiento exacto de valores atipicos superiores e inferiores."""
    df_datos = pd.DataFrame(
        {
            "valor": [-500.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 5000.0],
        }
    )

    df_acotado, limites = acotar_outliers_iqr(
        df_datos,
        columnas=["valor"],
        factor_iqr=1.5,
        p_inf=0.01,
        p_sup=0.99,
        retornar_limites=True,
    )

    cota_inf, cota_sup = limites["valor"]
    assert cota_inf < cota_sup

    # Los valores que excedian los limites deben quedar exactamente en la cota
    assert df_acotado["valor"].min() == pytest.approx(cota_inf)
    assert df_acotado["valor"].max() == pytest.approx(cota_sup)

    # Los valores intermedios deben permanecer invariantes
    assert df_acotado.loc[df_datos["valor"] == 12.0, "valor"].iloc[0] == 12.0

    # Verificacion de reutilizacion de cotas en evaluacion
    df_nuevo = pd.DataFrame({"valor": [-9999.0, 12.5, 9999.0]})
    df_nuevo_acotado = acotar_outliers_iqr(
        df_nuevo, columnas=["valor"], limites=limites
    )
    assert df_nuevo_acotado.loc[0, "valor"] == pytest.approx(cota_inf)
    assert df_nuevo_acotado.loc[1, "valor"] == 12.5
    assert df_nuevo_acotado.loc[2, "valor"] == pytest.approx(cota_sup)


# -----------------------------------------------------------------------------
# Verificacion matematica estricta de ausencia de data leakage
# -----------------------------------------------------------------------------


def test_verificacion_matematica_ausencia_data_leakage(tmp_path: Path):
    """Demuestra matematicamente que el escalador solo absorbe estadisticos de train.

    Tras el ajuste en train:
    - Train estandarizado exhibe media cercana a 0 y desviacion estandar cercana a 1.
    - Val y Test transformados con el escalador de train NO son forzados a media 0
      ni desviacion estandar 1, certificando la independencia de los estadisticos.
    """
    rng = np.random.default_rng(2026)

    # Train con distribucion N(mu=50, sigma=10)
    mu_train, sigma_train = 50.0, 10.0
    train_vals = rng.normal(loc=mu_train, scale=sigma_train, size=500)

    # Val con distribucion desplazada intencionalmente N(mu=65, sigma=15)
    mu_val, sigma_val = 65.0, 15.0
    val_vals = rng.normal(loc=mu_val, scale=sigma_val, size=150)

    # Test con distribucion desplazada intencionalmente N(mu=35, sigma=8)
    mu_test, sigma_test = 35.0, 8.0
    test_vals = rng.normal(loc=mu_test, scale=sigma_test, size=150)

    df_train = pd.DataFrame({"feature": train_vals})
    df_val = pd.DataFrame({"feature": val_vals})
    df_test = pd.DataFrame({"feature": test_vals})

    ruta_archivo_scaler = tmp_path / "escalador_produccion.joblib"
    columnas_a_escalar = ["feature"]

    # 1. Ajuste exclusivo sobre train
    escalador = ajustar_escalador(
        df_train, columnas_a_escalar, ruta_salida=ruta_archivo_scaler
    )

    # Los estadisticos internos del escalador deben corresponder a train
    assert np.isclose(escalador.mean_[0], df_train["feature"].mean(), atol=1e-7)
    assert np.isclose(escalador.scale_[0], df_train["feature"].std(ddof=0), atol=1e-7)

    # 2. Transformacion de train: media ~ 0 y std ~ 1
    df_train_esc = transformar_features(df_train, escalador, columnas_a_escalar)
    media_train_esc = float(df_train_esc["feature"].mean())
    std_train_esc = float(df_train_esc["feature"].std(ddof=0))

    assert np.isclose(media_train_esc, 0.0, atol=1e-7)
    assert np.isclose(std_train_esc, 1.0, atol=1e-7)

    # 3. Transformacion de val y test usando el escalador de train cargado de disco
    escalador_cargado = cargar_escalador(ruta_archivo_scaler)
    df_val_esc = transformar_features(df_val, escalador_cargado, columnas_a_escalar)
    df_test_esc = transformar_features(df_test, escalador_cargado, columnas_a_escalar)

    media_val_esc = float(df_val_esc["feature"].mean())
    std_val_esc = float(df_val_esc["feature"].std(ddof=0))

    media_test_esc = float(df_test_esc["feature"].mean())
    std_test_esc = float(df_test_esc["feature"].std(ddof=0))

    # Demostracion matematica: En val y test, la media y desviacion estandar
    # NO son forzadas a 0 ni a 1, probando ausencia de refit o fuga de datos.
    assert not np.isclose(media_val_esc, 0.0, atol=0.2)
    assert not np.isclose(std_val_esc, 1.0, atol=0.2)

    assert not np.isclose(media_test_esc, 0.0, atol=0.2)
    assert not np.isclose(std_test_esc, 1.0, atol=0.2)

    # Validacion analitica de la transformacion lineal esperada z = (x - mu_train) / sigma_train
    valor_arbitrario = df_val.loc[0, "feature"]
    valor_escalado_esperado = (valor_arbitrario - escalador.mean_[0]) / escalador.scale_[0]
    assert np.isclose(df_val_esc.loc[0, "feature"], valor_escalado_esperado, atol=1e-7)
