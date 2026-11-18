"""Filtro de respuestas del LLM y motor de respaldo sin conexion (Fase 7, Tarea 7.3).

redactar_diagnostico() es el punto de entrada del agente: intenta Ollama y,
si falla o responde mal, genera consejos por reglas para que la app nunca se detenga.
"""
import re

from src.agente.cliente_ollama import ClienteOllama
from src.agente.constructor_prompts import (
    NUM_RECOMENDACIONES,
    armar_prompt_diagnostico,
    armar_prompt_sistema,
)

# Consejos deterministas por variable y direccion de la desviacion
CONSEJOS = {
    ("initial_silence_sec", "exceso"): "Recorta el silencio inicial de {valor} a menos de {p75}: "
    "el sonido debe arrancar en el primer cuadro para evitar que el espectador deslice.",
    ("silence_ratio", "exceso"): "Reduce los huecos de silencio internos (ahora {valor}) "
    "con transiciones o un colchon musical continuo.",
    ("tempo_bpm", "exceso"): "Baja el tempo de {valor} hacia {p50} con time-stretch "
    "sin alterar el tono, para entrar en el rango viral ({p25} - {p75}).",
    ("tempo_bpm", "deficit"): "Sube el tempo de {valor} hacia {p50} con time-stretch "
    "sin alterar el tono, para entrar en el rango viral ({p25} - {p75}).",
    ("rms_mean", "deficit"): "Aumenta la sonoridad (RMS {valor} frente a {p50}) con un "
    "compresor suave y un limitador final a -1 dBTP.",
    ("rms_mean", "exceso"): "Reduce la sobrecompresion (RMS {valor}); deja respirar los "
    "transitorios bajando la ganancia del limitador.",
    ("spectral_centroid", "deficit"): "El audio suena opaco (centroide {valor}); realza "
    "agudos con un shelving de +2 a +3 dB sobre 6 kHz.",
    ("spectral_centroid", "exceso"): "El audio suena estridente (centroide {valor}); "
    "atenua 3-5 kHz con un EQ de banda ancha.",
    ("beat_regularity", "deficit"): "Refuerza un pulso estable (regularidad {valor}): "
    "cuantiza la percusion o anade un bombo constante.",
}
CONSEJOS_GENERALES = [
    "Coloca el momento mas pegadizo del audio en los primeros 2 segundos del video.",
    "Normaliza la sonoridad a unos -14 LUFS para que las plataformas no la atenuen.",
    "Evita fundidos de entrada largos: inicia con un ataque claro (voz, golpe o drop).",
    "Mantén la voz por encima de la musica con una reduccion de 6-8 dB en el fondo.",
]


def _fmt(valor: float, unidad: str) -> str:
    return f"{valor:.2f} {unidad}".strip() if abs(valor) >= 1 else f"{valor:.4f} {unidad}".strip()


def generar_recomendacion_respaldo(
    deficiencias: list[dict], n: int = NUM_RECOMENDACIONES
) -> list[str]:
    """Consejos en texto plano basados en reglas cuando el LLM no responde."""
    consejos: list[str] = []
    for d in deficiencias:
        plantilla = CONSEJOS.get((d["variable"], d["direccion"]))
        if plantilla:
            u = d["unidad"]
            consejos.append(
                plantilla.format(
                    valor=_fmt(d["valor"], u), p25=_fmt(d["p25"], u),
                    p50=_fmt(d["p50"], u), p75=_fmt(d["p75"], u),
                )
            )
    for general in CONSEJOS_GENERALES:  # completa hasta n con consejos universales
        if len(consejos) >= n:
            break
        consejos.append(general)
    return consejos[:n]


def limpiar_respuesta_llm(texto: str, n: int = NUM_RECOMENDACIONES) -> list[str]:
    """Extrae solo los puntos numerados/viñetas, descartando introducciones."""
    puntos = []
    for linea in texto.splitlines():
        coincidencia = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)", linea)
        if coincidencia:
            puntos.append(coincidencia.group(1).replace("**", "").strip())
    return puntos[:n]


def redactar_diagnostico(
    deficiencias: list[dict],
    cliente: ClienteOllama | None = None,
    engagement_estimado: float | None = None,
    atributos_favorables: list[dict] | None = None,
) -> dict:
    """Devuelve {"recomendaciones": list[str], "fuente": "ollama"|"respaldo", "motivo": str|None}."""
    cliente = cliente or ClienteOllama()
    respuesta = cliente.generar_respuesta(
        armar_prompt_diagnostico(deficiencias, engagement_estimado, atributos_favorables),
        armar_prompt_sistema(),
    )
    if respuesta["exito"]:
        puntos = limpiar_respuesta_llm(respuesta["texto"])
        if len(puntos) == NUM_RECOMENDACIONES:
            return {"recomendaciones": puntos, "fuente": "ollama", "motivo": None,
                    "modelo": respuesta["modelo"], "duracion_seg": respuesta["duracion_seg"]}
        motivo = f"El LLM devolvio {len(puntos)} puntos en lugar de {NUM_RECOMENDACIONES}"
    else:
        motivo = respuesta["motivo_error"]

    return {"recomendaciones": generar_recomendacion_respaldo(deficiencias),
            "fuente": "respaldo", "motivo": motivo}
