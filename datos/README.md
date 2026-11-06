# Fuentes de Datos y Guia de Replicabilidad

Este directorio almacena los datos necesarios para el entrenamiento y evaluacion del predictor de interaccion. Dado que los archivos de datos brutos superan los limites de tamano de Git, estan excluidos en el archivo `.gitignore`. Cualquier persona que clone este repositorio puede reconstruir el entorno de datos descargando los paquetes desde sus fuentes publicas y ubicandolos segun la siguiente estructura.

---

## 1. Estructura Requerida de Carpetas

```text
datos/
├── crudos/
│   ├── instagram/
│   ├── tiktok/
│   └── youtube/
├── medios/          (Audios MP3 descargados temporalmente)
└── procesados/      (Matriz consolidada en formato Parquet)
```

---

## 2. Enlaces de Descarga y Contenido por Carpeta

### A. Subcarpeta `datos/crudos/tiktok/`

1. **Popular TikTok Videos, Authors, and Musics**
   - **Plataforma:** Kaggle
   - **Autor:** `thedevastator`
   - **Enlace directo:** [Kaggle Dataset](https://www.kaggle.com/datasets/thedevastator/popular-tiktok-videos-authors-and-musics)
   - **Archivos extraidos:** `trending_videos.csv`, `trending_authors.csv`, `tiktok_collected_videos.csv`, `tiktok_collected_liked_videos.csv`, `tiktok_funny_hashtag_videos.csv`.
   - **Columnas clave:** `video_link`, `video_length`, `n_plays`, `n_likes`, `n_comments`, `n_shares`.
   - **Uso en el proyecto:** Provee los enlaces directos a las publicaciones de TikTok y los cuatro valores numericos que alimentan la formula del Engagement Rate.

2. **TikTok Trending Tracks**
   - **Plataforma:** Kaggle
   - **Autor:** `yamqwe`
   - **Enlace directo:** [Kaggle Dataset](https://www.kaggle.com/datasets/yamqwe/tiktok-trending-tracks)
   - **Archivo extraido:** `tiktok.csv`.
   - **Columnas clave:** `track_name`, `artist_name`, `tempo`, `danceability`, `energy`, `loudness`, `duration`.
   - **Uso en el proyecto:** Dataset de contingencia y respaldo tabular con atributos acusticos precalculados.

---

### B. Subcarpeta `datos/crudos/instagram/`

1. **Instagram Reels: Virality & Engagement Trends 2026**
   - **Plataforma:** Kaggle
   - **Autor:** `harishyadav0506`
   - **Enlace directo:** [Kaggle Dataset](https://www.kaggle.com/datasets/harishyadav0506/instagram-reels-virality-and-engagement-trends-2026)
   - **Archivos extraidos:** `instagram_reels_clean.csv`, `instagram_reels_clean.parquet`, `dataset_summary.csv`, `data_dictionary.csv`, reportes estadisticos.
   - **Columnas clave:** `post_id`, `views`, `likes`, `comments`, `video_duration_sec`, `engagement_rate_pct`, `virality_score`.
   - **Uso en el proyecto:** Evaluacion multiplataforma del modelo predictivo sobre publicaciones de Instagram Reels.

---

### C. Subcarpeta `datos/crudos/youtube/`

1. **Trending YouTube Video Statistics**
   - **Plataforma:** Kaggle
   - **Autor:** `datasnaek`
   - **Enlace directo:** [Kaggle Dataset](https://www.kaggle.com/datasets/datasnaek/youtube-new)
   - **Archivos extraidos:** Archivos regionales de tendencias (`MXvideos.csv`, `USvideos.csv`, `CAvideos.csv`, etc.) y sus catalogos JSON de categorias.
   - **Columnas clave:** `video_id`, `views`, `likes`, `comment_count`, `title`, `description`.
   - **Uso en el proyecto:** Analisis de escala y validacion cruzada sobre videos en formato corto y tendencias globales.

---

## 3. Instrucciones Paso a Paso para Nuevas Instalaciones

1. Descargar los archivos comprimidos `.zip` desde los enlaces indicados arriba.
2. Descomprimir el contenido de cada archivo.
3. Copiar los archivos resultantes en su respectiva subcarpeta dentro de `datos/crudos/` (`tiktok`, `instagram`, `youtube`).
4. Ejecutar el pipeline de preprocesamiento para compilar la matriz unificada:
   ```powershell
   .\.venv\Scripts\python.exe -m src.modelos.entrenador_pipeline
   ```
