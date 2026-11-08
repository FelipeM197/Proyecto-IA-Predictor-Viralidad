"""Particion estratificada de conjuntos de datos con variable objetivo continua.

Implementa la particion en entrenamiento, validacion y prueba mediante la
discretizacion en cuantiles de la variable continua, garantizando distribuciones
homogeneas entre particiones.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dividir_datos(
    df: pd.DataFrame,
    col_target: str,
    prop_train: float = 0.70,
    prop_val: float = 0.15,
    prop_test: float = 0.15,
    semilla: int = 42,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide un DataFrame en entrenamiento, validacion y prueba con estratificacion por cuantiles.

    Estratifica la variable objetivo continua mediante qcut para preservar su distribucion
    marginal en cada una de las tres particiones resultantes.
    """
    if col_target not in df.columns:
        raise KeyError(f"La columna objetivo '{col_target}' no existe en el DataFrame.")

    suma_props = prop_train + prop_val + prop_test
    if not np.isclose(suma_props, 1.0, atol=1e-4):
        raise ValueError(
            f"Las proporciones de division deben sumar 1.0. Suma actual: {suma_props:.4f}"
        )

    if prop_train <= 0 or prop_test <= 0 or prop_val < 0:
        raise ValueError("Las proporciones de train y test deben ser positivas.")

    if df[col_target].isna().any():
        raise ValueError(
            f"La columna objetivo '{col_target}' contiene valores nulos. Deben filtrarse previamente."
        )

    bins = pd.qcut(df[col_target], q=n_bins, labels=False, duplicates="drop")

    prop_resto = prop_val + prop_test

    # Si cada estrato tiene suficientes muestras, estratificamos; de lo contrario division simple
    conteo_minimo_estratos = bins.value_counts().min()
    estratificar_primera = bins if conteo_minimo_estratos >= 2 else None

    df_train, df_resto, _, bins_resto = train_test_split(
        df,
        bins,
        test_size=prop_resto,
        random_state=semilla,
        stratify=estratificar_primera,
    )

    if prop_val == 0.0:
        df_val = df_resto.iloc[0:0].copy()
        df_test = df_resto.copy()
    else:
        prop_test_relativa = prop_test / prop_resto
        conteo_minimo_resto = bins_resto.value_counts().min()
        estratificar_segunda = bins_resto if conteo_minimo_resto >= 2 else None

        df_val, df_test = train_test_split(
            df_resto,
            test_size=prop_test_relativa,
            random_state=semilla,
            stratify=estratificar_segunda,
        )

    return (
        df_train.copy().reset_index(drop=True),
        df_val.copy().reset_index(drop=True),
        df_test.copy().reset_index(drop=True),
    )
