"""Punto de entrada de la interfaz Streamlit (Fase 8, Tarea 8.2).

Ejecutar desde la raiz del proyecto: streamlit run app/main.py
"""
import sys
from pathlib import Path

import streamlit as st

# Streamlit agrega app/ al sys.path; se anade la raiz para importar config, src y app
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from config import settings  # noqa: E402
from app.componentes.graficos_plotly import grafico_barras_shap, grafico_radar, grafico_velocimetro  # noqa: E402
from app.componentes.panel_entrada import mostrar_panel_entrada  # noqa: E402
from app.componentes.panel_recomendaciones import mostrar_aviso_demo, mostrar_panel_recomendaciones  # noqa: E402
from app.componentes.reproductor_onda import mostrar_reproductor  # noqa: E402
from app.servicios.orquestador_inferencia import MotorInferencia, ResultadoInferencia  # noqa: E402
from src.agente.cliente_ollama import ClienteOllama  # noqa: E402

RUTA_CSS = Path(__file__).parent / "estilos.css"


def cargar_estilos(css: Path = RUTA_CSS) -> None:
    """Inyecta la hoja de estilos dentro de etiquetas <style> para que no se imprima."""
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def inicializar_estado() -> None:
    """Variables persistentes entre recargas de Streamlit."""
    st.session_state.setdefault("audio_actual", None)
    st.session_state.setdefault("resultado", None)


@st.cache_resource(show_spinner=False)
def obtener_cliente_llm() -> ClienteOllama:
    """Una sola instancia del cliente compartida por todas las sesiones."""
    return ClienteOllama()


@st.cache_resource(show_spinner="Cargando modelos en memoria...")
def obtener_motor() -> MotorInferencia:
    """Escalador, XGBoost y umbrales se leen del disco una unica vez."""
    return MotorInferencia(obtener_cliente_llm())


@st.cache_data(ttl=30, show_spinner=False)
def estado_ollama() -> tuple[bool, bool]:
    cliente = obtener_cliente_llm()
    activo = cliente.esta_disponible()
    return activo, activo and cliente.modelo_instalado()


def _insignia(ok: bool, texto_ok: str, texto_error: str) -> None:
    clase, texto = ("ok", texto_ok) if ok else ("error", texto_error)
    st.markdown(f"<span class='insignia insignia-{clase}'>{texto}</span>", unsafe_allow_html=True)


def dibujar_barra_lateral() -> None:
    motor = obtener_motor()
    with st.sidebar:
        st.markdown("### ⚙️ Estado del sistema")
        activo, modelo_ok = estado_ollama()
        _insignia(activo, "Ollama en linea", "Ollama desconectado")
        st.caption(f"Modelo: `{settings.OLLAMA_MODEL}` · {settings.OLLAMA_HOST}")
        if activo and not modelo_ok:
            st.warning(f"Modelo no descargado. Ejecuta: `ollama pull {settings.OLLAMA_MODEL}`")
        if not activo:
            st.info("Se usaran recomendaciones de respaldo basadas en reglas.")
        _insignia(not motor.modo_demo, "Modelo XGBoost cargado", "Modo demostracion")
        _insignia(motor.umbrales_oficiales, "Umbrales virales oficiales", "Umbrales de referencia")
        _insignia(motor.explicar is not None, "SHAP disponible", "SHAP pendiente")
        st.divider()
        st.caption(
            f"Muestreo: {settings.SAMPLE_RATE} Hz · Ventana: {int(settings.MAX_AUDIO_DURATION_SEC)} s"
        )


def procesar_solicitud(solicitud: dict) -> None:
    motor = obtener_motor()
    with st.spinner("Analizando: senal acustica → modelo → SHAP → agente LLM..."):
        if solicitud["tipo"] == "archivo":
            st.session_state["audio_actual"] = solicitud
            resultado = motor.analizar_bytes(solicitud["bytes"], solicitud["nombre"])
        else:
            audio, resultado = motor.analizar_url(solicitud["url"])
            st.session_state["audio_actual"] = audio
    st.session_state["resultado"] = resultado


def _grafico(figura) -> None:
    st.plotly_chart(figura, use_container_width=True, config={"displayModeBar": False})


def dibujar_resultados(r: ResultadoInferencia) -> None:
    st.divider()
    st.markdown("<div class='tarjeta-titulo'>3. Resultados del analisis</div>", unsafe_allow_html=True)
    if not r.exito:
        st.error(r.mensaje_error)
        return
    if r.modo_demo:
        mostrar_aviso_demo()

    col_gauge, col_radar = st.columns(2, gap="large")
    with col_gauge:
        if r.engagement_estimado is not None:
            _grafico(grafico_velocimetro(r.engagement_estimado * 100))
        m1, m2, m3 = st.columns(3)
        m1.metric("Tempo", f"{r.vector['tempo_bpm']:.0f} BPM")
        m2.metric("Silencio inicial", f"{r.vector['initial_silence_sec']:.2f} s")
        m3.metric("Energia RMS", f"{r.vector['rms_mean']:.3f}")
        st.caption(" · ".join(f"{k}: {v:.2f} s" for k, v in r.tiempos.items()))
    with col_radar:
        _grafico(grafico_radar(r.vector, r.umbrales))

    if r.explicacion_shap and r.explicacion_shap.get("contribuciones"):
        _grafico(grafico_barras_shap(r.explicacion_shap["contribuciones"]))
    mostrar_panel_recomendaciones(r.deficiencias, r.diagnostico)


def main() -> None:
    st.set_page_config(page_title="Predictor de Viralidad", page_icon="🎧", layout="wide")
    cargar_estilos()
    inicializar_estado()
    dibujar_barra_lateral()

    st.markdown("<div class='titulo-app'>🎧 Predictor de Viralidad por Audio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitulo-app'>Estima la tasa de interaccion de tu contenido a partir de "
        "sus senales acusticas y recibe consejos de edicion sonora.</div>",
        unsafe_allow_html=True,
    )

    col_entrada, col_audio = st.columns([1, 1.3], gap="large")
    with col_entrada:
        solicitud = mostrar_panel_entrada()
    if solicitud is not None:
        procesar_solicitud(solicitud)
    with col_audio:
        mostrar_reproductor(st.session_state["audio_actual"])

    if st.session_state["resultado"] is not None:
        dibujar_resultados(st.session_state["resultado"])


if __name__ == "__main__":
    main()
