"""Modulo de preprocesamiento de datos del proyecto."""

from src.preprocesamiento.calculo_target import (
    calcular_engagement_rate,
    filtrar_vistas_minimas,
    invertir_target_log,
    transformar_target_log,
)
from src.preprocesamiento.divisor_dataset import dividir_datos
from src.preprocesamiento.escalado_features import (
    ajustar_escalador,
    cargar_escalador,
    transformar_features,
)
from src.preprocesamiento.limpieza_outliers import (
    acotar_outliers_iqr,
    imputar_faltantes_mediana,
)

__all__ = [
    "filtrar_vistas_minimas",
    "calcular_engagement_rate",
    "transformar_target_log",
    "invertir_target_log",
    "acotar_outliers_iqr",
    "imputar_faltantes_mediana",
    "dividir_datos",
    "ajustar_escalador",
    "transformar_features",
    "cargar_escalador",
]
