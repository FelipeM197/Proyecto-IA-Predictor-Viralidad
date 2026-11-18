"""Componente de reproduccion e informacion basica del audio (Fase 8)."""
import io

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from config import settings

MIME_POR_FORMATO = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
}


@st.cache_data(show_spinner=False)
def leer_metadatos_audio(datos: bytes) -> dict:
    """Extrae duracion, frecuencia, canales y una onda reducida para graficar.

    Se cachea por contenido para no releer el audio en cada interaccion.
    """
    meta = {"decodificable": False, "tamano_mb": len(datos) / (1024 * 1024)}
    try:
        onda, sr = sf.read(io.BytesIO(datos), dtype="float32", always_2d=True)
    except Exception as err:  # formatos como m4a no siempre los decodifica libsndfile
        meta["motivo"] = str(err)[:120]
        return meta

    mono = onda.mean(axis=1)
    paso = max(1, mono.size // settings.PUNTOS_ONDA_UI)
    meta.update(
        decodificable=True,
        duracion_seg=mono.size / sr,
        sample_rate=int(sr),
        canales=int(onda.shape[1]),
        pico=float(np.max(np.abs(mono))) if mono.size else 0.0,
        onda_reducida=mono[::paso],
        paso_seg=paso / sr,
    )
    return meta


def _grafico_onda(onda: np.ndarray, paso_seg: float) -> go.Figure:
    tiempo = np.arange(onda.size) * paso_seg
    figura = go.Figure(
        go.Scatter(x=tiempo, y=onda, mode="lines", line=dict(color="#8b5cf6", width=1))
    )
    # Franja verde: ventana que realmente analiza el modelo
    figura.add_vrect(
        x0=0,
        x1=min(settings.MAX_AUDIO_DURATION_SEC, float(tiempo[-1]) if tiempo.size else 0),
        fillcolor="#22c55e",
        opacity=0.08,
        line_width=0,
    )
    figura.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Tiempo (s)", color="#9ca3af", gridcolor="#2a2f3d"),
        yaxis=dict(title="Amplitud", color="#9ca3af", gridcolor="#2a2f3d"),
        showlegend=False,
    )
    return figura


def mostrar_reproductor(audio: dict | None) -> dict | None:
    """Reproductor + tarjetas de metricas basicas. Devuelve los metadatos leidos.

    audio: {"nombre", "bytes", "formato"} o None si aun no se cargo nada.
    """
    if not audio:
        st.markdown(
            "<div class='tarjeta'><div class='tarjeta-titulo'>Esperando audio</div>"
            "<div class='tarjeta-texto'>Aqui veras el reproductor y la forma de onda.</div></div>",
            unsafe_allow_html=True,
        )
        return None
    nombre, datos, formato = audio["nombre"], audio["bytes"], audio["formato"]
    st.markdown(
        "<div class='tarjeta tarjeta-verde'><div class='tarjeta-titulo'>2. Tu audio</div>"
        f"<div class='tarjeta-texto'>{nombre}</div></div>",
        unsafe_allow_html=True,
    )
    st.audio(datos, format=MIME_POR_FORMATO.get(formato, "audio/mpeg"))

    meta = leer_metadatos_audio(datos)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tamano", f"{meta['tamano_mb']:.2f} MB")
    col2.metric("Formato", formato.upper())

    if not meta["decodificable"]:
        col3.metric("Duracion", "—")
        col4.metric("Frecuencia", "—")
        st.info("No se pudo leer la onda en el navegador; el analisis completo la procesara.")
        return meta

    col3.metric("Duracion", f"{meta['duracion_seg']:.1f} s")
    col4.metric("Frecuencia", f"{meta['sample_rate']:,} Hz".replace(",", " "))
    st.caption(
        f"Canales: {meta['canales']} · Pico de amplitud: {meta['pico']:.2f} · "
        f"Ventana analizada: primeros {int(settings.MAX_AUDIO_DURATION_SEC)} s (franja verde)"
    )
    st.plotly_chart(
        _grafico_onda(meta["onda_reducida"], meta["paso_seg"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    return meta
