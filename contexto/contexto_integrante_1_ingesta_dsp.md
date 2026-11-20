# Contexto y Pendientes: Integrante 1 (Ingesta de Medios y DSP Acústico)

> **Documento de trabajo individual para el Integrante 1 (`Persona 1`)**  
> Basado en la arquitectura formal de **`plan_maestro_detallado.md`** (Fases 2, 3, 9 y 10).

---

## 1. Responsabilidad Formal en el Proyecto
El Integrante 1 es el responsable del pipeline de adquisición de datos crudos, descarga de audio mediante `yt-dlp`, validación de integridad física de medios y extracción de las **28 características acústicas** con `librosa`, compilándolas en la matriz columnar de entrada para el modelo predictivo.

---

## 2. Lo que ya está completado y verificado (Commits `a122035`, `afbfb7f`, `4aafe1c`)

* **Módulo de Ingesta (`src/ingesta/` — Fase 2):**
  - `cliente_ytdlp.py`: Descargador con timeout estricto de 10 s y extracción forzada a MP3 mediante FFmpeg, con captura controlada de `DownloadError` y `ExtractorError`.
  - `validador_medios.py`: Inspección de integridad con `soundfile` (> 0 bytes y > 2 s de duración).
  - `gestor_lotes.py`: Procesamiento por lotes y registro de URLs rotas en `datos/fallos_descarga.csv`.
  - `cargador_kaggle.py`: Normalización de tablas de TikTok (`trending_videos.csv` y `tiktok.csv` de Kaggle).
  - `scraper_tendencias.py`: Extractor ligero de listas virales.
* **Procesamiento de Señales (`src/caracteristicas/` — Fase 3):**
  - `preprocesador_onda.py`: Conversión a mono, remuestreo a 22 050 Hz y normalización de amplitud a $[-1.0, 1.0]$.
  - `extractor_ritmo.py`: Tempo en BPM (`beat_track`), fuerza de ataque (`onset_strength`) y regularidad.
  - `extractor_energia.py`: Energía RMS, dinámica y cálculo exacto del silencio inicial (`initial_silence_sec`).
  - `extractor_espectral.py`: Centroide espectral (brillo), ancho de banda, caída (*rolloff*) y cruces por cero.
  - `extractor_timbre_mfcc.py`: 13 coeficientes MFCC y contenido armónico Chroma.
  - `constructor_matriz.py`: Compilador del vector de 28 variables acústicas + 4 métricas sociales.
* **Cuadernos y Pruebas (Fases 9 y 10):**
  - `notebooks/01_auditoria_ingesta_y_caidos.ipynb`: Supervivencia de enlaces.
  - `notebooks/02_laboratorio_senales_dsp.ipynb`: Gráficas paso a paso de espectrogramas Mel y ondas.
  - `tests/test_ingesta_ytdlp.py` y `tests/test_extractores_dsp.py`: Pruebas de unidad automatizadas aprobadas al 100 %.

---

## 3. Lo que falta exactamente por hacer (Lista de Pendientes Críticos)

Aunque el código está subido y pasa pruebas sintéticas, quedan pendientes las siguientes tareas operativas indispensables para la sustentación:

1. **Generar físicamente la matriz en disco (`matriz_acustica_cruda.parquet`):**
   - El script `src/caracteristicas/constructor_matriz.py` no se ha ejecutado para guardar el archivo en `datos/procesados/matriz_acustica_cruda.parquet`. Actualmente la carpeta `datos/procesados/` solo tiene el `.gitkeep`.
   - **Acción requerida:** Ejecutar en terminal:
     ```powershell
     .\.venv\Scripts\python.exe -m src.caracteristicas.constructor_matriz
     ```
     Verificar que se cree el archivo Parquet con al menos 600 filas y las 33 columnas de contrato (id_publicacion + 4 sociales + 28 acústicas).
2. **Descarga de audios reales en `datos/medios/`:**
   - La carpeta `datos/medios/` está vacía (0 bytes). Aunque el código incluye un generador de tonos sintéticos de respaldo, para la defensa ante el tribunal deben existir archivos MP3 reales descargados desde las URLs de TikTok para demostrar que `yt-dlp` y `FFmpeg` funcionan en vivo.
3. **Preparación para la Defensa Oral en Vivo (Tarea 10.2 del Plan Maestro):**
   - **Pregunta y prueba que le pedirá el jurado en vivo:**
     *«Cambie la duración máxima de análisis en `config/settings.py` de 30 a 15 segundos y demuestre que el preprocesador recorta la señal de inmediato sin errores.»*
   - **Cómo responder y ejecutar:**
     1. Abrir `config/settings.py` y modificar `MAX_AUDIO_DURATION_SEC = 15.0`.
     2. Ejecutar `tests/test_extractores_dsp.py` para demostrar que la onda se ajusta a $15 \times 22\,050 = 330\,750$ muestras.
     3. Argumentar: *«Al reducir la ventana a 15 segundos, nos concentramos en el gancho auditivo inicial del video, reduciendo el costo computacional de la STFT a la mitad a cambio de perder la evolución tímbrica de la segunda mitad de la pista.»*

---

## 4. Contratos de Entrega hacia el resto del equipo
* **Entregable hacia Persona 2 (Estadística y Modelos):**
  - Archivo `datos/procesados/matriz_acustica_cruda.parquet` libre de valores infinitos o columnas corruptas.
* **Entregable hacia Persona 3 (Streamlit):**
  - Función de extracción rápida `extraer_vector_acustico_desde_onda()` que `app/servicios/orquestador_inferencia.py` utiliza cuando el usuario sube un archivo a la app.
