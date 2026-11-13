from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from config import settings
from config.logging_config import configurar_registro
from src.ingesta.cliente_ytdlp import descargar_audio_url
from src.ingesta.validador_medios import es_archivo_valido

logger = configurar_registro("ingesta.lotes")

RUTA_FALLOS_DEFECTO = settings.DATOS_DIR / "fallos_descarga.csv"
UMBRAL_CAIDOS_MAX = float(getattr(settings, "UMBRAL_TASA_CAIDOS", 0.30))
COLUMNAS_FALLOS = ["id_publicacion", "url_origen", "fecha_intento", "motivo_error"]


def cargar_historial_fallos(ruta_fallos: Path | None = None) -> pd.DataFrame:
    archivo = ruta_fallos if ruta_fallos is not None else RUTA_FALLOS_DEFECTO
    if not archivo.exists():
        return pd.DataFrame(columns=COLUMNAS_FALLOS)
    try:
        df = pd.read_csv(archivo, dtype=str)
        for col in COLUMNAS_FALLOS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNAS_FALLOS]
    except Exception:
        return pd.DataFrame(columns=COLUMNAS_FALLOS)


def registrar_fallo_descarga(
    id_post: str, url: str, motivo: str, ruta_fallos: Path | None = None
) -> None:
    archivo = ruta_fallos if ruta_fallos is not None else RUTA_FALLOS_DEFECTO
    archivo.parent.mkdir(parents=True, exist_ok=True)
    nuevo = pd.DataFrame(
        [
            {
                "id_publicacion": str(id_post),
                "url_origen": str(url),
                "fecha_intento": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "motivo_error": str(motivo),
            }
        ]
    )
    escribir_cabecera = not archivo.exists() or archivo.stat().st_size == 0
    nuevo.to_csv(archivo, mode="a", header=escribir_cabecera, index=False, encoding="utf-8")


def ejecutar_ingesta_masiva(
    df_urls: pd.DataFrame,
    limite: int | None = None,
    directorio_medios: Path | None = None,
    ruta_fallos: Path | None = None,
) -> dict[str, int | float | bool | list[dict]]:
    carpeta_medios = directorio_medios if directorio_medios is not None else settings.DATOS_MEDIOS_DIR
    archivo_fallos = ruta_fallos if ruta_fallos is not None else RUTA_FALLOS_DEFECTO
    carpeta_medios.mkdir(parents=True, exist_ok=True)

    historial = cargar_historial_fallos(archivo_fallos)
    ids_fallidos = set(historial["id_publicacion"].astype(str).tolist())

    muestra = df_urls.head(limite).copy() if limite is not None else df_urls.copy()
    resultados: list[dict] = []
    exitos = 0
    fallos = 0

    for _, fila in muestra.iterrows():
        id_post = str(fila["id_publicacion"]).strip()
        url = str(fila.get("url_origen", "")).strip()

        if id_post in ids_fallidos:
            fallos += 1
            continue

        ruta_existente = carpeta_medios / f"{id_post}.mp3"
        if ruta_existente.exists():
            valido, _ = es_archivo_valido(ruta_existente)
            if valido:
                exitos += 1
                resultados.append(
                    {"id_publicacion": id_post, "ruta_audio": str(ruta_existente), "exito": True}
                )
                continue

        respuesta = descargar_audio_url(url=url, id_post=id_post, directorio_destino=carpeta_medios)
        if respuesta["exito"] and respuesta["ruta_audio"]:
            valido, motivo_val = es_archivo_valido(Path(str(respuesta["ruta_audio"])))
            if valido:
                exitos += 1
                resultados.append(respuesta)
            else:
                fallos += 1
                registrar_fallo_descarga(id_post, url, motivo_val, archivo_fallos)
        else:
            fallos += 1
            registrar_fallo_descarga(
                id_post, url, str(respuesta.get("motivo_error", "Error desconocido")), archivo_fallos
            )

    total = max(exitos + fallos, 1)
    tasa_caidos = fallos / total
    activar_respaldo = tasa_caidos > UMBRAL_CAIDOS_MAX
    logger.info(
        "Lote finalizado: %d exitos, %d fallos (tasa caidos: %.1f%%, respaldo: %s)",
        exitos,
        fallos,
        tasa_caidos * 100.0,
        activar_respaldo,
    )
    return {
        "total_procesados": exitos + fallos,
        "exitos": exitos,
        "fallos": fallos,
        "tasa_caidos": tasa_caidos,
        "activar_respaldo": activar_respaldo,
        "resultados": resultados,
    }
