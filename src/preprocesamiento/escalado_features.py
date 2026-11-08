"""Estandarizacion de caracteristicas numericas sin fuga de datos.

Implementa el ajuste de StandardScaler sobre la particion de entrenamiento,
su persistencia en disco y la aplicacion coherente sobre conjuntos de validacion,
prueba o inferencia.
"""

from pathlib import Path
from typing import Union
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def ajustar_escalador(
    df_train: pd.DataFrame,
    columnas_features: list[str],
    ruta_salida: Union[str, Path, None] = None,
) -> StandardScaler:
    """Ajusta un StandardScaler exclusivamente sobre las caracteristicas de entrenamiento.

    Garantiza que la media y la desviacion estandar provengan unicamente del conjunto
    de entrenamiento para impedir cualquier tipo de fuga de datos.
    """
    columnas_faltantes = [col for col in columnas_features if col not in df_train.columns]
    if columnas_faltantes:
        raise KeyError(
            f"Columnas ausentes en el DataFrame de entrenamiento: {columnas_faltantes}"
        )

    escalador = StandardScaler()
    escalador.fit(df_train[columnas_features])

    if ruta_salida is not None:
        ruta = Path(ruta_salida)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(escalador, ruta)

    return escalador


def transformar_features(
    df: pd.DataFrame,
    escalador: StandardScaler,
    columnas_features: list[str],
) -> pd.DataFrame:
    """Transforma las columnas indicadas utilizando un StandardScaler previamente ajustado.

    Retorna una copia independiente del DataFrame con las columnas especificadas
    estandarizadas a media cero y varianza unitaria.
    """
    columnas_faltantes = [col for col in columnas_features if col not in df.columns]
    if columnas_faltantes:
        raise KeyError(f"Columnas ausentes en el DataFrame: {columnas_faltantes}")

    df_transformado = df.copy()
    valores_escalados = escalador.transform(df_transformado[columnas_features])
    df_transformado[columnas_features] = valores_escalados

    return df_transformado


def cargar_escalador(ruta_escalador: Union[str, Path]) -> StandardScaler:
    """Carga un objeto StandardScaler previamente serializado con joblib."""
    ruta = Path(ruta_escalador)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontro el archivo del escalador en: {ruta}")

    escalador: StandardScaler = joblib.load(ruta)
    return escalador
