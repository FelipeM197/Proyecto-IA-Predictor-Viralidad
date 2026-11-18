"""Pruebas automatizadas para el modulo del agente de diagnostico (Fase 10, Tarea 10.1).

Verifica:
1. Deteccion matematica de desviaciones y regla de consistencia de signo SHAP.
2. Ensamblaje de prompts con anclaje numerico duro y respeto de NUM_RECOMENDACIONES.
3. Tolerancia a fallos de ClienteOllama ante puerto inactivo o desconexion.
4. Activacion automatica del motor de respaldo basado en reglas cuando Ollama falla.
"""

from config import settings
from src.agente.cliente_ollama import ClienteOllama
from src.agente.constructor_prompts import (
    NUM_RECOMENDACIONES,
    armar_prompt_diagnostico,
    armar_prompt_sistema,
)
from src.agente.detector_deficiencias import evaluar_desviaciones
from src.agente.formateador_respuesta import (
    generar_recomendacion_respaldo,
    limpiar_respuesta_llm,
    redactar_diagnostico,
)


def _umbrales_prueba() -> dict[str, dict[str, float]]:
    return {
        "initial_silence_sec": {"p25": 0.04, "p50": 0.12, "p75": 0.28, "media": 0.16},
        "tempo_bpm": {"p25": 115.0, "p50": 122.0, "p75": 132.0, "media": 122.0},
        "rms_mean": {"p25": 0.12, "p50": 0.18, "p75": 0.25, "media": 0.18},
    }


def test_detector_deficiencias_prioriza_silencio_critico_y_filtra_shap_positivo() -> None:
    umbrales = _umbrales_prueba()
    vector_usuario = {
        "initial_silence_sec": 0.85,  # > 0.35s (critica) y > p75
        "tempo_bpm": 145.0,           # > p75 (exceso), pero con SHAP positivo -> debe ignorarse
        "rms_mean": 0.05,             # < p25 (deficit) con SHAP negativo -> moderada
    }
    valores_shap = {
        "initial_silence_sec": -0.045,
        "tempo_bpm": 0.021,           # contribuyo positivamente a la prediccion
        "rms_mean": -0.018,
    }

    deficiencias = evaluar_desviaciones(vector_usuario, umbrales, valores_shap)

    variables_detectadas = [d["variable"] for d in deficiencias]
    assert "tempo_bpm" not in variables_detectadas
    assert variables_detectadas == ["initial_silence_sec", "rms_mean"]
    assert deficiencias[0]["gravedad"] == "critica"
    assert deficiencias[0]["direccion"] == "exceso"
    assert deficiencias[1]["gravedad"] == "moderada"
    assert deficiencias[1]["direccion"] == "deficit"


def test_detector_no_penaliza_silencio_inicial_bajo() -> None:
    umbrales = _umbrales_prueba()
    vector_usuario = {
        "initial_silence_sec": 0.0,  # < p25, pero menos silencio nunca es deficiencia
        "tempo_bpm": 122.0,
        "rms_mean": 0.18,
    }
    deficiencias = evaluar_desviaciones(vector_usuario, umbrales, valores_shap=None)
    assert deficiencias == []


def test_constructor_prompts_anclaje_numerico_y_cantidad_recomendaciones() -> None:
    umbrales = _umbrales_prueba()
    vector = {"initial_silence_sec": 1.20, "tempo_bpm": 98.0, "rms_mean": 0.18}
    shap_vals = {"initial_silence_sec": -0.031, "tempo_bpm": -0.012, "rms_mean": 0.005}
    deficiencias = evaluar_desviaciones(vector, umbrales, shap_vals)

    prompt_sis = armar_prompt_sistema(n_recomendaciones=2)
    assert "EXACTAMENTE 2 recomendaciones" in prompt_sis

    prompt_usr = armar_prompt_diagnostico(
        deficiencias=deficiencias,
        engagement_estimado=0.0845,
        atributos_favorables=[{"variable": "rms_mean", "impacto_shap": 0.005}],
        n_recomendaciones=2,
    )
    assert "8.45 %" in prompt_usr
    assert "1.20 s" in prompt_usr
    assert "98.00 BPM" in prompt_usr
    assert "rms_mean" in prompt_usr
    assert "Redacta 2 recomendaciones" in prompt_usr


def test_cliente_ollama_controla_puerto_inactivo_sin_excepcion() -> None:
    # Puerto local reservado sin servicio activo y timeout minimo para ejecucion rapida
    cliente_offline = ClienteOllama(
        host="http://127.0.0.1:65530",
        modelo=settings.OLLAMA_MODEL,
        timeout_sec=0.3,
    )

    assert cliente_offline.esta_disponible() is False
    assert cliente_offline.modelo_instalado() is False

    respuesta = cliente_offline.generar_respuesta("Prueba sin conexion")
    assert isinstance(respuesta, dict)
    assert respuesta["exito"] is False
    assert respuesta["texto"] == ""
    assert isinstance(respuesta["motivo_error"], str) and len(respuesta["motivo_error"]) > 0


def test_formateador_activa_motor_respaldo_cuando_ollama_no_responde() -> None:
    umbrales = _umbrales_prueba()
    vector = {"initial_silence_sec": 0.90, "tempo_bpm": 100.0, "rms_mean": 0.04}
    deficiencias = evaluar_desviaciones(vector, umbrales, valores_shap=None)

    cliente_offline = ClienteOllama(host="http://127.0.0.1:65530", timeout_sec=0.3)
    diagnostico = redactar_diagnostico(
        deficiencias=deficiencias,
        cliente=cliente_offline,
        engagement_estimado=0.05,
    )

    assert diagnostico["fuente"] == "respaldo"
    assert len(diagnostico["recomendaciones"]) == NUM_RECOMENDACIONES
    assert any("silencio inicial" in r.lower() for r in diagnostico["recomendaciones"])
    assert diagnostico["motivo"] is not None


def test_limpiar_respuesta_llm_y_respaldo_directo() -> None:
    texto_sucio = (
        "Claro, aqui tienes mis consejos:\n"
        "1. **Recorta** el silencio inicial a menos de 0.28 s.\n"
        "2. Sube el tempo hacia 122 BPM.\n"
        "3. Aumenta la compresion RMS.\n"
        "4. Punto extra que debe recortarse.\n"
    )
    limpios = limpiar_respuesta_llm(texto_sucio, n=3)
    assert len(limpios) == 3
    assert limpios[0] == "Recorta el silencio inicial a menos de 0.28 s."

    respaldo_vacio = generar_recomendacion_respaldo([], n=2)
    assert len(respaldo_vacio) == 2
