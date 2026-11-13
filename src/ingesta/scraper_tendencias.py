import pandas as pd
from config import settings
from src.ingesta.cargador_kaggle import limpiar_valor_numerico

RUTA_TIKTOK_RESPALDO = settings.DATOS_CRUDOS_DIR / "tiktok" / "tiktok.csv"
RUTA_INSTAGRAM_REELS = settings.DATOS_CRUDOS_DIR / "instagram" / "instagram_reels_clean.csv"


def consultar_tendencias_locales(top_n: int = 20) -> pd.DataFrame:
    registros: list[dict[str, object]] = []

    if RUTA_TIKTOK_RESPALDO.exists():
        df_tk = pd.read_csv(RUTA_TIKTOK_RESPALDO).head(top_n)
        for _, fila in df_tk.iterrows():
            registros.append(
                {
                    "plataforma": "TikTok",
                    "id_referencia": str(fila.get("track_id", "")),
                    "titulo": str(fila.get("track_name", "")),
                    "autor": str(fila.get("artist_name", "")),
                    "tempo_bpm": limpiar_valor_numerico(fila.get("tempo", 120.0)),
                    "popularidad": limpiar_valor_numerico(fila.get("popularity", 50.0)),
                }
            )

    if RUTA_INSTAGRAM_REELS.exists():
        df_ig = pd.read_csv(RUTA_INSTAGRAM_REELS).head(top_n)
        for _, fila in df_ig.iterrows():
            registros.append(
                {
                    "plataforma": "Instagram Reels",
                    "id_referencia": str(fila.get("post_id", "")),
                    "titulo": str(fila.get("niche", "Reel")),
                    "autor": str(fila.get("creator_handle", "")),
                    "tempo_bpm": 122.0,
                    "popularidad": limpiar_valor_numerico(fila.get("virality_score", 50.0)),
                }
            )

    return pd.DataFrame(registros)
