import os
import shutil
from pathlib import Path
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError
from config import settings
from config.logging_config import configurar_registro

logger = configurar_registro("ingesta.ytdlp")

TIMEOUT_RED_SEC = int(os.getenv("YTDLP_SOCKET_TIMEOUT", "10"))
REINTENTOS_RED = int(os.getenv("YTDLP_MAX_RETRIES", "2"))
CALIDAD_MP3_KBPS = "192"


def descargar_audio_url(
    url: str, id_post: str, directorio_destino: Path | None = None
) -> dict[str, str | bool | None]:
    id_limpio = str(id_post).strip()
    if not url or not isinstance(url, str) or not url.strip().startswith(("http://", "https://")):
        return {
            "exito": False,
            "id_publicacion": id_limpio,
            "url_origen": str(url),
            "ruta_audio": None,
            "motivo_error": "URL invalida o con esquema no soportado",
        }

    carpeta = directorio_destino if directorio_destino is not None else settings.DATOS_MEDIOS_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    plantilla_salida = str(carpeta / f"{id_limpio}.%(ext)s")
    ruta_mp3 = carpeta / f"{id_limpio}.mp3"

    ffmpeg_activo = shutil.which("ffmpeg") is not None
    opciones: dict = {
        "format": "bestaudio/best",
        "outtmpl": plantilla_salida,
        "noplaylist": True,
        "socket_timeout": TIMEOUT_RED_SEC,
        "retries": REINTENTOS_RED,
        "quiet": True,
        "no_warnings": True,
    }

    if ffmpeg_activo:
        opciones["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": CALIDAD_MP3_KBPS,
            }
        ]

    try:
        with yt_dlp.YoutubeDL(opciones) as cliente:
            cliente.extract_info(url.strip(), download=True)

        if ruta_mp3.exists():
            ruta_final = ruta_mp3
        else:
            coincidencias = list(carpeta.glob(f"{id_limpio}.*"))
            if not coincidencias:
                raise DownloadError("No se genero ningun archivo de audio en disco")
            ruta_final = coincidencias[0]

        logger.info("Audio descargado correctamente para publicacion %s", id_limpio)
        return {
            "exito": True,
            "id_publicacion": id_limpio,
            "url_origen": url.strip(),
            "ruta_audio": str(ruta_final),
            "motivo_error": None,
        }
    except (DownloadError, ExtractorError) as err:
        motivo = f"Error de extraccion yt-dlp: {str(err)[:160]}"
        logger.warning("Enlace caido o inaccesible (%s): %s", id_limpio, motivo)
        return {
            "exito": False,
            "id_publicacion": id_limpio,
            "url_origen": url.strip(),
            "ruta_audio": None,
            "motivo_error": motivo,
        }
    except Exception as err:
        motivo = f"Excepcion inesperada de red/sistema: {str(err)[:160]}"
        logger.error("Fallo al descargar %s: %s", id_limpio, motivo)
        return {
            "exito": False,
            "id_publicacion": id_limpio,
            "url_origen": url.strip(),
            "ruta_audio": None,
            "motivo_error": motivo,
        }
