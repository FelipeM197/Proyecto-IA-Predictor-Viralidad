from importlib import import_module

_EXPORTACIONES = {
    "entrenar_ridge": "src.modelos.modelo_baseline",
    "entrenar_elasticnet": "src.modelos.modelo_baseline",
    "entrenar_baselines": "src.modelos.modelo_baseline",
    "RedNeuronalAcustica": "src.modelos.modelo_red_neuronal",
    "crear_random_forest": "src.modelos.modelo_arboles",
    "entrenar_random_forest": "src.modelos.modelo_arboles",
    "crear_xgboost": "src.modelos.modelo_arboles",
    "entrenar_xgboost": "src.modelos.modelo_arboles",
    "optimizar_xgboost_cv": "src.modelos.optimizador_hiperparametros",
    "calcular_metricas_regresion": "src.modelos.evaluador_metricas",
    "evaluar_modelos": "src.modelos.evaluador_metricas",
    "cargar_datos_entrenamiento": "src.modelos.evaluador_metricas",
    "preparar_particiones_modelado": "src.modelos.evaluador_metricas",
    "guardar_reporte_comparativo": "src.modelos.evaluador_metricas",
    "ejecutar_pipeline_entrenamiento": "src.modelos.evaluador_metricas",
    "generar_umbrales_virales": "src.modelos.explicabilidad_shap",
    "explicar_prediccion_individual": "src.modelos.explicabilidad_shap",
    "calcular_importancia_global_shap": "src.modelos.explicabilidad_shap",
}

__all__ = list(_EXPORTACIONES.keys())


# Carga diferida para evitar RuntimeWarning de runpy al ejecutar python -m src.modelos.evaluador_metricas
def __getattr__(nombre: str):
    if nombre in _EXPORTACIONES:
        modulo = import_module(_EXPORTACIONES[nombre])
        valor = getattr(modulo, nombre)
        globals()[nombre] = valor
        return valor
    raise AttributeError(f"El paquete 'src.modelos' no exporta '{nombre}'.")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
