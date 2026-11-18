"""Deteccion matematica de desviaciones frente al perfil viral (Fase 7, Tarea 7.1).

Logica 100 % determinista: el LLM nunca decide que esta mal, solo redacta.
Regla de anclaje SHAP (contrato con Persona 2): una variable es deficiencia
prioritaria si su valor fisico cae fuera de [p25, p75] Y su impacto SHAP es
negativo. Si aun no hay SHAP disponible, basta con estar fuera de rango.
"""
from config import settings

# Nombre legible y unidad fisica de las variables que se muestran al usuario
ETIQUETAS = {
    "tempo_bpm": ("Tempo", "BPM"),
    "onset_strength_mean": ("Fuerza de ataques", ""),
    "beat_regularity": ("Regularidad del pulso", ""),
    "rms_mean": ("Energia media (RMS)", ""),
    "dynamic_range": ("Rango dinamico", ""),
    "initial_silence_sec": ("Silencio inicial", "s"),
    "silence_ratio": ("Proporcion de silencio", ""),
    "spectral_centroid": ("Brillo (centroide espectral)", "Hz"),
    "spectral_rolloff": ("Frecuencia de caida", "Hz"),
    "spectral_flatness": ("Planitud espectral", ""),
    "zero_crossing_rate": ("Cruces por cero", ""),
}

# Variables donde "menos" nunca es un problema (no se reporta deficit)
VARIABLES_SOLO_EXCESO = {"initial_silence_sec", "silence_ratio"}
ORDEN_GRAVEDAD = {"critica": 0, "moderada": 1}


def etiqueta_de(variable: str) -> tuple[str, str]:
    return ETIQUETAS.get(variable, (variable.replace("_", " ").capitalize(), ""))


def _clasificar_gravedad(variable: str, valor: float) -> str:
    if variable == "initial_silence_sec" and valor > settings.UMBRAL_SILENCIO_CRITICO_SEC:
        return "critica"  # el espectador abandona en los primeros segundos
    return "moderada"


def evaluar_desviaciones(
    vector_usuario: dict[str, float],
    umbrales: dict[str, dict[str, float]],
    valores_shap: dict[str, float] | None = None,
) -> list[dict]:
    """Compara la pista contra los percentiles virales y devuelve las fallas ordenadas.

    Cada elemento: {variable, etiqueta, unidad, valor, p25, p50, p75,
                    direccion ("exceso"|"deficit"), gravedad, impacto_shap}
    """
    deficiencias: list[dict] = []
    for variable, rango in umbrales.items():
        if variable not in vector_usuario:
            continue
        if valores_shap is None and variable not in ETIQUETAS:
            continue  # sin SHAP solo se juzgan variables fisicamente interpretables
        valor = float(vector_usuario[variable])
        p25, p75 = float(rango["p25"]), float(rango["p75"])

        if valor > p75:
            direccion = "exceso"
        elif valor < p25 and variable not in VARIABLES_SOLO_EXCESO:
            direccion = "deficit"
        else:
            continue  # dentro del rango viral

        impacto = None if valores_shap is None else float(valores_shap.get(variable, 0.0))
        if impacto is not None and impacto >= 0:
            continue  # fuera de rango pero ayudo a la prediccion: no se penaliza

        nombre, unidad = etiqueta_de(variable)
        deficiencias.append(
            {
                "variable": variable,
                "etiqueta": nombre,
                "unidad": unidad,
                "valor": round(valor, 4),
                "p25": round(p25, 4),
                "p50": round(float(rango.get("p50", (p25 + p75) / 2)), 4),
                "p75": round(p75, 4),
                "direccion": direccion,
                "gravedad": _clasificar_gravedad(variable, valor),
                "impacto_shap": None if impacto is None else round(impacto, 4),
            }
        )

    # Primero las criticas; dentro de cada nivel, el SHAP mas negativo primero
    deficiencias.sort(
        key=lambda d: (ORDEN_GRAVEDAD[d["gravedad"]], d["impacto_shap"] or 0.0)
    )
    return deficiencias
