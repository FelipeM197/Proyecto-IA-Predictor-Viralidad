"""Componente de entrada: subida de archivo de audio o enlace web (Fase 8)."""
import streamlit as st

from config import settings


def _tarjeta(titulo: str, texto: str, variante: str = "violeta") -> None:
    st.markdown(
        f"<div class='tarjeta tarjeta-{variante}'>"
        f"<div class='tarjeta-titulo'>{titulo}</div>"
        f"<div class='tarjeta-texto'>{texto}</div></div>",
        unsafe_allow_html=True,
    )


def _es_url_valida(url: str) -> bool:
    return url.strip().startswith(("http://", "https://"))


def mostrar_panel_entrada() -> dict | None:
    """Dibuja las dos pestanas de entrada y devuelve la solicitud del usuario.

    Retorna None si el usuario aun no envio nada, o un diccionario:
      {"tipo": "archivo", "nombre": str, "bytes": bytes, "formato": str}
      {"tipo": "url", "url": str}
    """
    _tarjeta(
        "1. Carga tu audio",
        "Sube una pista local o pega el enlace de un video de TikTok, YouTube o Instagram. "
        f"Se analizaran los primeros {int(settings.MAX_AUDIO_DURATION_SEC)} segundos.",
    )

    pestana_archivo, pestana_url = st.tabs(["🎵 Archivo local", "🔗 Enlace web"])

    with pestana_archivo:
        archivo = st.file_uploader(
            "Selecciona un archivo de audio",
            type=settings.FORMATOS_AUDIO_UI,
            help=f"Formatos: {', '.join(settings.FORMATOS_AUDIO_UI)}. "
            f"Tamano maximo: {settings.MAX_UPLOAD_MB} MB.",
            key="entrada_archivo",
        )
        if st.button("Analizar archivo", key="btn_archivo", use_container_width=True):
            if archivo is None:
                st.warning("Primero selecciona un archivo de audio.")
            elif archivo.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"El archivo supera el limite de {settings.MAX_UPLOAD_MB} MB.")
            else:
                return {
                    "tipo": "archivo",
                    "nombre": archivo.name,
                    "bytes": archivo.getvalue(),
                    "formato": archivo.name.rsplit(".", 1)[-1].lower(),
                }

    with pestana_url:
        url = st.text_input(
            "Enlace del video o audio",
            placeholder="https://www.tiktok.com/@usuario/video/...",
            key="entrada_url",
        )
        if st.button("Analizar enlace", key="btn_url", use_container_width=True):
            if not url or not _es_url_valida(url):
                st.warning("Ingresa un enlace valido que empiece por http:// o https://")
            else:
                return {"tipo": "url", "url": url.strip()}

    return None
