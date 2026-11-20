import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from config import settings


def _normalizar_vector_individual(
    vector_escalado,
    nombres_columnas: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    if isinstance(vector_escalado, pd.DataFrame):
        if len(vector_escalado) != 1:
            raise ValueError("Se esperaba exactamente una fila en vector_escalado.")
        if nombres_columnas is None:
            return vector_escalado, list(vector_escalado.columns)
        nombres = list(nombres_columnas)
        return vector_escalado[nombres], nombres

    if isinstance(vector_escalado, pd.Series):
        arreglo = vector_escalado.to_numpy(dtype=float).reshape(1, -1)
        if nombres_columnas is not None:
            nombres = list(nombres_columnas)
        elif isinstance(vector_escalado.index, pd.RangeIndex) and len(vector_escalado) == len(
            settings.COLUMNAS_ACUSTICAS
        ):
            # Si la serie carece de etiquetas acusticas, se asume el orden canonico de 28 atributos
            nombres = list(settings.COLUMNAS_ACUSTICAS)
        else:
            nombres = [str(idx) for idx in vector_escalado.index]
        return pd.DataFrame(arreglo, columns=nombres), nombres

    arreglo = np.asarray(vector_escalado, dtype=float)
    if arreglo.ndim == 1:
        arreglo = arreglo.reshape(1, -1)
    elif arreglo.ndim != 2 or arreglo.shape[0] != 1:
        raise ValueError("vector_escalado debe ser 1D o 2D con una sola fila.")

    nombres = (
        list(settings.COLUMNAS_ACUSTICAS)
        if nombres_columnas is None
        else list(nombres_columnas)
    )
    if len(nombres) != arreglo.shape[1]:
        raise ValueError(
            f"Dimension incompatible: {arreglo.shape[1]} atributos para {len(nombres)} columnas."
        )
    return pd.DataFrame(arreglo, columns=nombres), nombres


def generar_umbrales_virales(
    df_entrenamiento: pd.DataFrame,
    col_target: str = settings.COLUMNA_TARGET,
    columnas: list[str] | None = None,
    percentil_corte: float = settings.PERCENTIL_CORTE_VIRAL,
    percentiles: tuple[int, ...] | list[int] = settings.PERCENTILES_UMBRALES,
    ruta_salida: str | Path | None = settings.RUTA_UMBRALES_VIRALES,
) -> dict[str, dict[str, float]]:
    if df_entrenamiento.empty:
        raise ValueError("El conjunto de entrenamiento esta vacio.")
    if col_target not in df_entrenamiento.columns:
        raise KeyError(f"Columna objetivo ausente: '{col_target}'.")
    if not 0.0 <= percentil_corte <= 1.0:
        raise ValueError("percentil_corte debe ubicarse en [0, 1].")

    cols = list(settings.COLUMNAS_ACUSTICAS) if columnas is None else list(columnas)
    faltantes = [col for col in cols if col not in df_entrenamiento.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas acusticas: {faltantes}")

    objetivo = df_entrenamiento[col_target].dropna()
    if objetivo.empty:
        raise ValueError(f"La columna objetivo '{col_target}' no contiene valores validos.")

    corte_viral = float(objetivo.quantile(percentil_corte))
    df_viral = df_entrenamiento.loc[df_entrenamiento[col_target] >= corte_viral]

    umbrales: dict[str, dict[str, float]] = {}
    for col in cols:
        serie = df_viral[col].dropna().astype(float)
        if serie.empty:
            raise ValueError(f"Sin datos validos para '{col}' en el segmento viral.")
        estadisticos = {f"p{int(p)}": float(np.percentile(serie, p)) for p in percentiles}
        estadisticos["media"] = float(serie.mean())
        umbrales[col] = estadisticos

    if ruta_salida is not None:
        ruta = Path(ruta_salida)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(umbrales, indent=2, ensure_ascii=False), encoding="utf-8")

    return umbrales


def explicar_prediccion_individual(
    modelo,
    vector_escalado,
    nombres_columnas: list[str] | None = None,
    top_k: int = settings.TOP_K_ATRIBUTOS_SHAP,
) -> dict[str, object]:
    df_entrada, nombres = _normalizar_vector_individual(vector_escalado, nombres_columnas)

    explicador = shap.TreeExplainer(modelo)
    valores_shap = np.asarray(explicador.shap_values(df_entrada), dtype=float).reshape(-1)
    # shap actualiza expected_value tras calcular shap_values en XGBoost y unifica escalar/arreglo
    valor_base = float(np.asarray(explicador.expected_value).reshape(-1)[0])
    if len(valores_shap) != len(nombres):
        raise ValueError(
            f"TreeExplainer retorno {len(valores_shap)} valores para {len(nombres)} atributos."
        )

    contribuciones = {col: float(val) for col, val in zip(nombres, valores_shap)}
    limite = max(0, int(top_k))
    ordenados = sorted(contribuciones.items(), key=lambda par: par[1], reverse=True)

    atributos_favorables = [
        {"variable": col, "impacto_shap": val}
        for col, val in ordenados
        if val > 0.0
    ][:limite]
    atributos_perjudiciales = [
        {"variable": col, "impacto_shap": val}
        for col, val in reversed(ordenados)
        if val < 0.0
    ][:limite]

    return {
        "valor_base": valor_base,
        "contribuciones": contribuciones,
        "atributos_favorables": atributos_favorables,
        "atributos_perjudiciales": atributos_perjudiciales,
    }


def calcular_importancia_global_shap(modelo, x_muestra) -> pd.DataFrame:
    if len(x_muestra) == 0:
        raise ValueError("La muestra para calcular importancia global SHAP esta vacia.")

    if isinstance(x_muestra, pd.DataFrame):
        nombres = list(x_muestra.columns)
        entrada = x_muestra
    else:
        arreglo = np.asarray(x_muestra, dtype=float)
        if arreglo.ndim == 1:
            arreglo = arreglo.reshape(1, -1)
        n_cols = arreglo.shape[1]
        nombres = (
            list(settings.COLUMNAS_ACUSTICAS)
            if n_cols == len(settings.COLUMNAS_ACUSTICAS)
            else [f"feature_{i}" for i in range(n_cols)]
        )
        entrada = pd.DataFrame(arreglo, columns=nombres)

    explicador = shap.TreeExplainer(modelo)
    valores_shap = np.asarray(explicador.shap_values(entrada), dtype=float)
    if valores_shap.ndim == 1:
        valores_shap = valores_shap.reshape(1, -1)

    resumen = pd.DataFrame(
        {
            "variable": nombres,
            "importancia_media_abs": np.mean(np.abs(valores_shap), axis=0),
            "impacto_medio_con_signo": np.mean(valores_shap, axis=0),
        }
    )
    return resumen.sort_values("importancia_media_abs", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Importacion diferida para no arrastrar el stack de entrenamiento en inferencia
    from src.modelos.evaluador_metricas import (
        cargar_datos_entrenamiento,
        preparar_particiones_modelado,
    )

    df_crudo, _ = cargar_datos_entrenamiento()
    particiones = preparar_particiones_modelado(df_crudo, guardar_escalador=False)
    generar_umbrales_virales(particiones["df_train_limpio"])
