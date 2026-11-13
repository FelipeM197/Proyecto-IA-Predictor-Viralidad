from pathlib import Path
import numpy as np
import pandas as pd
from config import settings

CARPETA_TIKTOK = settings.DATOS_CRUDOS_DIR / "tiktok"
RUTA_TIKTOK_VIDEOS = CARPETA_TIKTOK / "trending_videos.csv"
RUTA_TIKTOK_RESPALDO = CARPETA_TIKTOK / "tiktok.csv"
CANTIDAD_MFCC = sum(1 for col in settings.COLUMNAS_ACUSTICAS if col.startswith("mfcc_"))


def limpiar_valor_numerico(valor: object) -> float:
    if pd.isna(valor):
        return 0.0
    if isinstance(valor, (int, float, np.number)):
        return float(max(valor, 0.0))
    texto = str(valor).strip().upper().replace(",", "")
    if not texto:
        return 0.0
    mult = 1.0
    if texto.endswith("M"):
        mult, texto = 1e6, texto[:-1]
    elif texto.endswith("K"):
        mult, texto = 1e3, texto[:-1]
    elif texto.endswith("B"):
        mult, texto = 1e9, texto[:-1]
    try:
        return max(float(texto) * mult, 0.0)
    except ValueError:
        return 0.0


def _normalizar_tabla_tiktok(df: pd.DataFrame) -> pd.DataFrame:
    mapeo = {
        "video_id": "id_publicacion",
        "post_id": "id_publicacion",
        "video_link": "url_origen",
        "n_plays": "views",
        "video_playCount": "views",
        "n_likes": "likes",
        "n_comments": "comments",
        "video_commentCount": "comments",
        "n_shares": "shares",
        "video_shareCount": "shares",
    }
    df_norm = df.rename(columns={k: v for k, v in mapeo.items() if k in df.columns}).copy()
    if "id_publicacion" not in df_norm.columns:
        df_norm["id_publicacion"] = [f"post_{i}" for i in range(len(df_norm))]
    df_norm["id_publicacion"] = df_norm["id_publicacion"].astype(str).str.strip()
    df_norm["url_origen"] = df_norm.get("url_origen", pd.Series("", index=df_norm.index)).fillna("").astype(str).str.strip()

    for col in settings.COLUMNAS_SOCIALES:
        df_norm[col] = df_norm[col].apply(limpiar_valor_numerico).astype(np.float64) if col in df_norm.columns else 0.0

    columnas_salida = ["id_publicacion", "url_origen"] + settings.COLUMNAS_SOCIALES
    return df_norm[columnas_salida].drop_duplicates(subset=["id_publicacion"]).reset_index(drop=True)


def cargar_dataset_tiktok(ruta_csv: Path | None = None) -> pd.DataFrame:
    if ruta_csv is not None:
        return _normalizar_tabla_tiktok(pd.read_csv(ruta_csv))

    archivos = [
        RUTA_TIKTOK_VIDEOS,
        CARPETA_TIKTOK / "tiktok_collected_videos.csv",
        CARPETA_TIKTOK / "tiktok_collected_liked_videos.csv",
    ]
    tablas = [_normalizar_tabla_tiktok(pd.read_csv(f)) for f in archivos if f.exists()]
    if not tablas:
        raise FileNotFoundError(f"No se encontro el dataset de TikTok en {CARPETA_TIKTOK}")
    return pd.concat(tablas, ignore_index=True).drop_duplicates(subset=["id_publicacion"]).reset_index(drop=True)


def cargar_dataset_respaldo(ruta_csv: Path | None = None, limite: int | None = 800) -> pd.DataFrame:
    archivo = ruta_csv if ruta_csv is not None else RUTA_TIKTOK_RESPALDO
    df_raw = pd.read_csv(archivo).drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    if limite is not None:
        df_raw = df_raw.head(limite).copy()

    rng = np.random.default_rng(settings.SEMILLA_ALEATORIA)
    n = len(df_raw)

    duracion_sec = df_raw["duration"].apply(limpiar_valor_numerico).to_numpy() / 1000.0
    tempo = np.clip(df_raw["tempo"].apply(limpiar_valor_numerico).to_numpy(), 60.0, 200.0)
    energia = np.clip(df_raw["energy"].apply(limpiar_valor_numerico).to_numpy(), 0.05, 1.0)
    dance = np.clip(df_raw["danceability"].apply(limpiar_valor_numerico).to_numpy(), 0.05, 1.0)
    speech = np.clip(df_raw["speechiness"].apply(limpiar_valor_numerico).to_numpy(), 0.01, 1.0)
    acoust = np.clip(df_raw["acousticness"].apply(limpiar_valor_numerico).to_numpy(), 0.01, 1.0)
    valence = np.clip(df_raw["valence"].apply(limpiar_valor_numerico).to_numpy(), 0.05, 1.0)
    pop = np.clip(df_raw["popularity"].apply(limpiar_valor_numerico).to_numpy(), 1.0, 100.0)

    silencio_ini = np.clip(
        0.04 + (1.0 - energia) * 0.45 + acoust * 0.35 + rng.uniform(0.0, 0.15, n),
        0.0,
        settings.MAX_AUDIO_DURATION_SEC,
    )
    beat_reg = np.clip(0.45 + 0.50 * dance - 0.10 * acoust + rng.normal(0.0, 0.03, n), 0.1, 0.99)
    rms_m = np.clip(0.03 + 0.28 * energia + rng.normal(0.0, 0.01, n), 0.005, 0.48)

    datos: dict[str, object] = {
        "id_publicacion": df_raw["track_id"].astype(str).str.strip(),
        "duration_sec": duracion_sec.astype(np.float64),
        "tempo_bpm": tempo.astype(np.float64),
        "onset_strength_mean": np.clip(0.8 + 4.2 * energia + 1.5 * dance, 0.1, 10.0),
        "onset_strength_std": np.clip(0.3 + 1.8 * energia + 0.6 * speech, 0.05, 5.0),
        "beat_regularity": beat_reg,
        "rms_mean": rms_m,
        "rms_std": np.clip(rms_m * (0.35 + 0.25 * dance), 0.002, 0.24),
        "dynamic_range": np.clip(rms_m * 2.1 + 0.04 * (1.0 - acoust), 0.02, 0.78),
        "initial_silence_sec": silencio_ini,
        "silence_ratio": np.clip(silencio_ini / settings.MAX_AUDIO_DURATION_SEC + 0.05 * acoust, 0.0, 0.85),
        "spectral_centroid": np.clip(1100.0 + 2800.0 * energia + 1200.0 * speech - 800.0 * acoust, 300.0, 7500.0),
        "spectral_bandwidth": np.clip(1400.0 + 1900.0 * energia + 600.0 * valence, 400.0, 5800.0),
        "spectral_rolloff": np.clip(2400.0 + 4800.0 * energia + 1100.0 * speech, 800.0, 9800.0),
        "spectral_flatness": np.clip(0.02 + 0.22 * speech + 0.10 * (1.0 - acoust), 0.001, 0.85),
        "zero_crossing_rate": np.clip(0.03 + 0.16 * speech + 0.07 * energia, 0.008, 0.32),
    }

    for idx in range(1, CANTIDAD_MFCC + 1):
        base_mfcc = -95.0 + 55.0 * energia if idx == 1 else (18.0 / idx) * (valence - 0.3 * acoust)
        datos[f"mfcc_{idx}"] = (base_mfcc + rng.normal(0.0, 2.5, n)).astype(np.float64)

    datos["chroma_stft_mean"] = np.clip(0.28 + 0.38 * valence + 0.18 * dance, 0.05, 0.95)

    score_acustico = (
        0.38 * beat_reg
        + 0.32 * energia
        - 0.45 * np.clip(silencio_ini, 0.0, 1.5)
        - 0.18 * np.abs(tempo - 122.0) / 60.0
        + 0.15 * (pop / 100.0)
    )
    tasa_latente = np.clip(0.035 + 0.14 * (1.0 / (1.0 + np.exp(-3.2 * score_acustico))) + rng.normal(0.0, 0.012, n), 0.008, 0.45)
    views = np.round(np.exp(rng.uniform(np.log(500.0), np.log(2500000.0), n)))
    interacciones = views * tasa_latente

    datos["views"] = views.astype(np.float64)
    datos["likes"] = np.round(interacciones * 0.82).astype(np.float64)
    datos["comments"] = np.round(interacciones * 0.10).astype(np.float64)
    datos["shares"] = np.round(interacciones * 0.08).astype(np.float64)

    return pd.DataFrame(datos)
