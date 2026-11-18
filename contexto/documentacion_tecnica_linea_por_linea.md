# Documentación Técnica Exhaustiva Línea por Línea (Fase 1)

Este documento detalla la lógica, motivo de diseño y restricciones de cada línea de código escrita en los módulos de configuración, infraestructura y diagnóstico del proyecto.

---

## 1. Módulo `config/__init__.py`

```python
1: from config import settings
2: 
3: __all__ = ["settings"]
```

### Análisis línea por línea:
* **Línea 1 (`from config import settings`):** Importa el módulo de configuración dentro del paquete `config`. Esto permite que cualquier parte del proyecto pueda escribir `from config import settings` o simplemente importar el módulo directamente.
* **Línea 3 (`__all__ = ["settings"]`):** Variable especial de Python a nivel de módulo (PEP 8). Define la interfaz pública del paquete cuando alguien ejecuta `from config import *`, restringiendo la exportación exclusivamente a `settings` para evitar polución del espacio de nombres con importaciones internas.

---

## 2. Módulo `config/settings.py`

Este archivo centraliza todas las constantes del proyecto para cumplir la **Regla de Oro de Parametrización Centralizada** (cero números mágicos en el código).

### Bloque A: Importaciones y Rutas Dinámicas (Líneas 1 a 19)

```python
1: import os
2: from pathlib import Path
3: 
4: # Anclaje dinamico para independizar la ejecucion del directorio de trabajo actual
5: BASE_DIR = Path(__file__).resolve().parent.parent
```
* **Línea 1 (`import os`):** Importa la librería estándar de sistema operativo para leer variables de entorno locales si existen (`os.getenv`).
* **Línea 2 (`from pathlib import Path`):** Importa la clase orientada a objetos `Path`, que gestiona rutas de archivos de forma nativa e independiente del sistema operativo (usando separadores `/` o `\` según corresponda).
* **Línea 4 (`# Anclaje dinamico...`):** Comentario técnico que explica el porqué del diseño.
* **Línea 5 (`BASE_DIR = Path(__file__).resolve().parent.parent`):** Obtiene la ruta física del archivo `settings.py` (`__file__`), resuelve cualquier enlace simbólico (`.resolve()`), sube un nivel a la carpeta `config` (`.parent`) y sube otro nivel a la raíz del repositorio (`.parent`). Esto garantiza que los scripts funcionen igual si se ejecutan desde la raíz, desde `scripts/` o desde `notebooks/`.

```python
7: DATOS_DIR = BASE_DIR / "datos"
8: DATOS_CRUDOS_DIR = DATOS_DIR / "crudos"
9: DATOS_MEDIOS_DIR = DATOS_DIR / "medios"
10: DATOS_PROCESADOS_DIR = DATOS_DIR / "procesados"
11: MODELOS_DIR = BASE_DIR / "modelos_guardados"
12: LOGS_DIR = BASE_DIR / "logs"
```
* **Líneas 7 a 12:** Construcción de las rutas absolutas del proyecto usando el operador `/` de `pathlib`.
  - `DATOS_DIR`: Directorio principal de datos.
  - `DATOS_CRUDOS_DIR`: Ubicación de los archivos originales de TikTok, Instagram y YouTube.
  - `DATOS_MEDIOS_DIR`: Ubicación de los archivos de audio MP3 descargados temporalmente.
  - `DATOS_PROCESADOS_DIR`: Destino de la matriz tabular unificada en formato Parquet.
  - `MODELOS_DIR`: Destino de los binarios serializados (`.joblib`) y reportes de métricas.
  - `LOGS_DIR`: Carpeta para la persistencia del archivo rotativo de eventos.

```python
14: DIRECTORIOS_PROYECTO = [
15:     DATOS_CRUDOS_DIR,
16:     DATOS_MEDIOS_DIR,
17:     DATOS_PROCESADOS_DIR,
18:     MODELOS_DIR,
19:     LOGS_DIR,
20: ]
```
* **Líneas 14 a 20:** Lista de objetos `Path` que representa la infraestructura física de almacenamiento. Es consumida por `scripts/init_dirs.py` para asegurar que las carpetas existan en cualquier clon limpio del repositorio.

---

### Bloque B: Rutas de Artefactos de Salida (Líneas 22 a 27)

```python
22: RUTA_MATRIZ_CRUDA = DATOS_PROCESADOS_DIR / "matriz_acustica_cruda.parquet"
23: RUTA_ESCALADOR = MODELOS_DIR / "escalador_acustico.joblib"
24: RUTA_MODELO_MLP = MODELOS_DIR / "modelo_mlp_final.joblib"
25: RUTA_MODELO_XGBOOST = MODELOS_DIR / "modelo_xgboost_final.joblib"
26: RUTA_REPORTE_MODELOS = MODELOS_DIR / "reporte_comparativo_modelos.json"
27: RUTA_UMBRALES_VIRALES = MODELOS_DIR / "umbrales_virales.json"
```
* **Líneas 22 a 27:** Contratos deterministas de persistencia. Define exactamente dónde se lee y dónde se escribe cada archivo binario o JSON entre las fases:
  - `RUTA_MATRIZ_CRUDA`: Entrega de Persona 1 a Persona 2.
  - `RUTA_ESCALADOR` y `RUTA_MODELO_XGBOOST`: Entregas de Persona 2 a Persona 3 para la inferencia en tiempo real en la interfaz de Streamlit.
  - `RUTA_REPORTE_MODELOS`: Evidencia estadística de comparación entre los modelos para la defensa.
  - `RUTA_UMBRALES_VIRALES`: Valores de referencia que consume el agente diagnóstico de Llama 3 en la Fase 7.

---

### Bloque C: Constantes de Procesamiento Digital de Señales (Líneas 29 a 33)

```python
29: SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
30: MAX_AUDIO_DURATION_SEC = float(os.getenv("MAX_AUDIO_DURATION_SEC", "30.0"))
31: N_FFT = 2048
32: HOP_LENGTH = 512
33: SILENCE_THRESHOLD_DB = 30
```
* **Línea 29 (`SAMPLE_RATE`):** Tasa de muestreo acústica fijada en $22\,050\text{ Hz}$, con opción de sobreescritura mediante variable de entorno `.env`. Según el Teorema de Nyquist, permite capturar frecuencias hasta $11\,025\text{ Hz}$, suficiente para el espectro de voz y música en redes sociales.
* **Línea 30 (`MAX_AUDIO_DURATION_SEC`):** Ventana de corte temporal de 30 segundos. Estandariza la longitud de análisis de cualquier audio largo.
* **Línea 31 (`N_FFT = 2048`):** Tamaño de ventana de la Transformada Rápida de Fourier (FFT). Equivale a una duración temporal de $2048 / 22050 \approx 92.88\text{ ms}$ y una resolución frecuencial de $10.77\text{ Hz}$ por bin.
* **Línea 32 (`HOP_LENGTH = 512`):** Salto temporal entre ventanas consecutivas de la STFT. Equivale a $512 / 22050 \approx 23.22\text{ ms}$, generando un traslape del 75 % que previene discontinuidades de fase.
* **Línea 33 (`SILENCE_THRESHOLD_DB = 30`):** Umbral relativo de potencia en decibelios para clasificar una trama como silencio acústico mediante `librosa.effects.split`.

---

### Bloque D: Parámetros de Ingeniería Estadística y Partición (Líneas 35 a 45)

```python
35: SEMILLA_ALEATORIA = 42
36: MIN_VIEWS = 100
37: PERCENTIL_INFERIOR_OUTLIERS = 0.01
38: PERCENTIL_SUPERIOR_OUTLIERS = 0.99
39: FACTOR_IQR = 1.5
40: PROPORCION_TRAIN = 0.70
41: PROPORCION_VAL = 0.15
42: PROPORCION_TEST = 0.15
43: BINS_ESTRATIFICACION = 5
```
* **Línea 35 (`SEMILLA_ALEATORIA = 42`):** Semilla global que fija la inicialización de generadores de números pseudoaleatorios en `numpy`, `scikit-learn` y `xgboost`, garantizando reproducibilidad exacta de resultados.
* **Línea 36 (`MIN_VIEWS = 100`):** Filtro de alcance mínimo para estabilizar la varianza del cociente de interacción.
* **Líneas 37 a 39:** Constantes para el tratamiento de valores atípicos mediante el criterio de Tukey ($1.5 \cdot \text{IQR}$) acotado hacia los percentiles $P_1$ y $P_{99}$.
* **Líneas 40 a 42:** Proporciones del particionamiento de datos: 70 % para ajuste de modelos, 15 % para monitoreo de parada temprana y validación cruzada, y 15 % para evaluación ciega final.
* **Línea 43 (`BINS_ESTRATIFICACION = 5`):** Cantidad de cuantiles para discretizar la variable objetivo continua al ejecutar `train_test_split(..., stratify=...)`.

---

### Bloque E: Esquema de Columnas y Variables (Líneas 45 a 81)

```python
45: COLUMNAS_SOCIALES = ["views", "likes", "comments", "shares"]
46: COLUMNA_TARGET = "engagement_rate"
47: COLUMNA_TARGET_LOG = "engagement_rate_log"
```
* **Línea 45:** Nombres unificados de las cuatro métricas sociales que alimentan la fórmula de rendimiento.
* **Líneas 46 y 47:** Identificadores para la columna de tasa en escala real y en escala logarítmica $\ln(1+x)$.

```python
49: COLUMNAS_ACUSTICAS = [
50:     "tempo_bpm",
51:     "onset_strength_mean",
52:     "onset_strength_std",
53:     "beat_regularity",
54:     "rms_mean",
55:     "rms_std",
56:     "dynamic_range",
57:     "initial_silence_sec",
58:     "silence_ratio",
59:     "spectral_centroid",
60:     "spectral_bandwidth",
61:     "spectral_rolloff",
62:     "spectral_flatness",
63:     "zero_crossing_rate",
64:     "mfcc_1",
...
77:     "mfcc_13",
78:     "chroma_stft_mean",
79: ]
```
* **Líneas 49 a 79:** Lista de las 28 variables acústicas extraídas por `librosa`. Fija el orden estricto de columnas para que el escalador `StandardScaler` y los modelos predictivos reciban siempre la matriz con la misma dimensionalidad y alineación.

---

### Bloque F: Hiperparámetros de Modelos y Explicabilidad (Líneas 81 a 105)

```python
81: ALPHAS_RIDGE = [0.01, 0.1, 1.0, 10.0, 100.0]
82: ALPHAS_ELASTICNET = [0.001, 0.01, 0.1, 1.0]
83: L1_RATIOS_ELASTICNET = [0.2, 0.5, 0.8]
84: 
85: CAPAS_OCULTAS_MLP = (128, 64, 32)
86: ALPHA_L2_MLP = 0.001
87: LR_INICIAL_MLP = 0.003
88: BATCH_SIZE_MLP = 64
89: MAX_ITER_MLP = 600
90: PACIENCIA_MLP = 15
```
* **Líneas 81 a 83:** Espacios de búsqueda para regularización $L_2$ (`RidgeCV`) y balance $L_1/L_2$ (`ElasticNetCV`).
* **Líneas 85 a 90:** Topología del Perceptrón Multicapa (3 capas de 128, 64 y 32 neuronas), penalización de pesos $L_2$, tasa de aprendizaje inicial ($0.003$), tamaño de lote ($64$), máximo de iteraciones ($600$) y paciencia de parada temprana ($15$ épocas sin mejora).

```python
92: N_ESTIMADORES_RF = 200
93: MAX_DEPTH_RF = 10
94: SUBSAMPLE_ARBOLES = 0.8
95: 
96: PARAM_GRID_XGB = {
97:     "max_depth": [3, 5, 7],
98:     "learning_rate": [0.03, 0.08, 0.15],
99:     "n_estimators": [100, 200],
100: }
101: PLIEGUES_CV = 5
```
* **Líneas 92 a 94:** Configuración de Random Forest (200 árboles de profundidad máxima 10 con submuestreo de 80 %).
* **Líneas 96 a 100:** Rejilla para `GridSearchCV` de XGBoost (18 combinaciones evaluadas bajo validación cruzada de 5 pliegues).

```python
103: PERCENTIL_CORTE_VIRAL = 0.85
104: PERCENTILES_UMBRALES = (25, 50, 75)
105: TOP_K_ATRIBUTOS_SHAP = 3
```
* **Líneas 103 a 105:** Umbral del 15 % superior para segmentar publicaciones virales, cálculo de percentiles intercuartiles ($P_{25}, P_{50}, P_{75}$) y selección de los 3 atributos con mayor impacto positivo y negativo en SHAP.

---

## 3. Módulo `config/logging_config.py`

Provee un registrador de eventos estructurado que reemplaza llamadas a `print()`.

```python
1: import logging
2: import sys
3: from logging.handlers import RotatingFileHandler
4: from config import settings
```
* **Líneas 1 a 4:** Importación del módulo de registro estándar, salida de sistema, manejador de archivos rotativos y la configuración central.

```python
7: def configurar_registro(nombre: str = "sistema") -> logging.Logger:
8:     logger = logging.getLogger(nombre)
9:     if logger.handlers:
10:         return logger
```
* **Línea 7:** Definición de la función con anotaciones de tipo (`str` y retorno `logging.Logger`).
* **Líneas 8 a 10:** Obtiene el registrador por nombre. Si ya posee manejadores asociados (`handlers`), lo retorna de inmediato para evitar que un mensaje se imprima múltiples veces si la función se invoca repetidamente.

```python
12:     logger.setLevel(logging.INFO)
13:     settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
14:     ruta_log = settings.LOGS_DIR / "sistema.log"
```
* **Línea 12:** Fija el nivel de severidad mínimo en `INFO`, ignorando mensajes de depuración verbosa (`DEBUG`).
* **Línea 13:** Asegura físicamente la existencia de la carpeta `logs/`.
* **Línea 14:** Define la ruta del archivo de registro `logs/sistema.log`.

```python
16:     formato = logging.Formatter(
17:         "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
18:         datefmt="%Y-%m-%d %H:%M:%S",
19:     )
```
* **Líneas 16 a 19:** Formato estandarizado de cada línea: fecha y hora en formato ISO simplificado, nivel de severidad justificado a 7 caracteres, nombre del módulo emisor y cuerpo del mensaje.

```python
21:     consola = logging.StreamHandler(sys.stdout)
22:     consola.setFormatter(formato)
23:     consola.setLevel(logging.INFO)
24:     logger.addHandler(consola)
```
* **Líneas 21 a 24:** Configuración del primer destino: la consola estándar (`stdout`). Permite monitorear el progreso en vivo durante ejecuciones en terminal o pruebas con `pytest`.

```python
26:     # Rotacion a 5 MB para evitar saturacion de disco en ejecuciones extensas
27:     archivo = RotatingFileHandler(
28:         ruta_log, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
29:     )
30:     archivo.setFormatter(formato)
31:     archivo.setLevel(logging.INFO)
32:     logger.addHandler(archivo)
33: 
34:     return logger
```
* **Líneas 26 a 32:** Configuración del segundo destino: el archivo rotativo `logs/sistema.log`. Al alcanzar los 5 MB ($5 \times 1024 \times 1024\text{ bytes}$), se archiva y se crea uno nuevo, conservando un máximo de 3 respaldos históricos (`backupCount=3`). Esto previene la saturación del disco duro ante descargas masivas o reintentos en bucle.
* **Línea 34:** Retorna la instancia configurada del registrador.

---

## 4. Módulo `scripts/init_dirs.py`

Entregable de la Persona 2 (Fase 1, Tarea 1.3) para crear la jerarquía del proyecto.

```python
1: from pathlib import Path
2: from config import settings
```
* **Líneas 1 y 2:** Importa `Path` y las constantes de directorios desde `config.settings`.

```python
5: def inicializar_directorios(directorios: list[Path] | None = None) -> list[Path]:
6:     rutas = directorios if directorios is not None else settings.DIRECTORIOS_PROYECTO
7:     creados: list[Path] = []
8:     for ruta in rutas:
9:         ruta.mkdir(parents=True, exist_ok=True)
10:         marcador = ruta / ".gitkeep"
11:         if not marcador.exists():
12:             marcador.touch()
13:         creados.append(ruta)
14:     return creados
```
* **Línea 5:** Función que acepta una lista opcional de rutas y retorna las rutas procesadas.
* **Línea 6:** Si no se especifican rutas, toma por defecto `settings.DIRECTORIOS_PROYECTO`.
* **Líneas 8 y 9:** Itera sobre cada directorio y ejecuta `.mkdir(parents=True, exist_ok=True)`. `parents=True` crea cualquier carpeta superior que falte; `exist_ok=True` evita errores si la carpeta ya existía.
* **Líneas 10 a 12:** Crea el archivo `.gitkeep` con `.touch()` si la carpeta es nueva, asegurando que Git reconozca y conserve la carpeta en el repositorio aunque esté vacía de datos.
* **Línea 14:** Retorna la lista de rutas creadas o verificadas.

```python
17: if __name__ == "__main__":
18:     rutas_listas = inicializar_directorios()
19:     print(f"Directorios listos: {len(rutas_listas)}")
```
* **Líneas 17 a 19:** Bloque de ejecución principal. Solo se ejecuta si se corre el script directamente desde la terminal (`python scripts/init_dirs.py`), imprimiendo una confirmación sobria con el conteo de carpetas verificadas.

---

## 5. Módulo `scripts/check_environment.py`

Entregable de diagnóstico del sistema (Fase 1, Tarea 1.3).

```python
1: import shutil
2: import sys
3: import urllib.request
4: 
5: VERSION_MINIMA_PYTHON = (3, 10)
6: URL_OLLAMA_PING = "http://localhost:11434"
```
* **Líneas 1 a 3:** Importa `shutil` (para buscar ejecutables del sistema), `sys` (para inspeccionar la versión del intérprete) y `urllib.request` (librería nativa para peticiones HTTP ligeras sin requerir dependencias externas como `requests`).
* **Línea 5:** Tupla con la versión mínima exigida de Python (`3.10`).
* **Línea 6:** Dirección URL del puerto local donde escucha el servidor de Ollama.

```python
9: def auditar_entorno() -> dict[str, bool]:
10:     version_actual = sys.version_info[:2]
11:     python_valido = version_actual >= VERSION_MINIMA_PYTHON
12: 
13:     ffmpeg_disponible = shutil.which("ffmpeg") is not None
```
* **Líneas 10 y 11:** Evalúa si la versión actual de Python (`major, minor`) cumple o supera la mínima requerida.
* **Línea 13:** Emplea `shutil.which("ffmpeg")` para buscar el binario en la variable de entorno `PATH` de Windows. Si existe retorna la ruta al ejecutable, validando que la condición no sea `None`.

```python
15:     ollama_activo = False
16:     try:
17:         with urllib.request.urlopen(URL_OLLAMA_PING, timeout=1.5) as respuesta:
18:             ollama_activo = respuesta.status == 200
19:     except Exception:
20:         ollama_activo = False
21: 
22:     return {
23:         "python": python_valido,
24:         "ffmpeg": ffmpeg_disponible,
25:         "ollama": ollama_activo,
26:     }
```
* **Líneas 15 a 20:** Realiza un sondeo HTTP no bloqueante hacia el puerto `11434` con un tiempo límite de $1.5\text{ segundos}$. Si el servidor responde con código de estado HTTP 200, marca `ollama_activo = True`. Si el servidor está apagado o no responde, atrapa la excepción y establece `False` de forma segura sin abortar la ejecución del script.
* **Líneas 22 a 26:** Retorna un diccionario booleano con el estado de los tres componentes.

```python
29: if __name__ == "__main__":
30:     resultado = auditar_entorno()
31:     print(f"Python >= 3.10: {'OK' if resultado['python'] else 'Falta actualizar'}")
32:     print(f"FFmpeg en PATH: {'OK' if resultado['ffmpeg'] else 'No detectado (modo respaldo activo)'}")
33:     print(f"Servidor Ollama: {'OK' if resultado['ollama'] else 'No detectado (modo contingencia activo)'}")
```
* **Líneas 29 a 33:** Ejecución directa que muestra en pantalla un reporte claro y sobrio del estado de cada servicio.

---

## 6. Módulo `.gitignore`

Define las exclusiones para mantener el repositorio limpio y proteger información sensible.

```text
1: # Entorno virtual y cache de Python
2: .venv/
3: venv/
4: env/
5: __pycache__/
6: *.py[cod]
7: *$py.class
8: *.egg-info/
9: build/
10: dist/
```
* **Líneas 1 a 10:** Excluye el entorno virtual aislado, el código de bytes precompilado por Python (`.pyc`) y los directorios temporales generados al ejecutar `pip install -e .`.

```text
12: # Configuraciones y carpetas de agentes
13: .agents/
14: .antigravity/
```
* **Líneas 12 a 14:** Excluye los directorios locales del sistema multiagente de Google Antigravity para que las configuraciones de sesión no se mezclen con el código fuente del proyecto.

```text
16: # Datos pesados y audios descargados (NO subir a Git)
17: datos/crudos/*
18: !datos/crudos/.gitkeep
19: datos/medios/*
20: !datos/medios/.gitkeep
21: datos/procesados/*
22: !datos/procesados/.gitkeep
```
* **Líneas 16 a 22:** Exclusión de la capa de datos. Bloquea la subida de archivos pesados (CSV, Parquet, MP3, WAV) en todas las subcarpetas de `datos/crudos/`, pero conserva la existencia de las carpetas en el repositorio gracias a la excepción `!*.gitkeep`.

```text
24: # Modelos entrenados pesados y logs
25: modelos_guardados/*.joblib
26: modelos_guardados/*.pkl
27: !modelos_guardados/.gitkeep
28: logs/*.log
29: !logs/.gitkeep
30: .env
31: .DS_Store
32: Thumbs.db
```
* **Líneas 24 a 32:** Excluye los pesos binarios de los modelos entrenados (`.joblib`, `.pkl`), los archivos de registro (`.log`) y las variables de entorno locales (`.env`), evitando filtraciones de contraseñas o tokens.

```text
34: # Archivos de planeacion interna local (no subir a GitHub)
35: plan_maestro_detallado.md
36: plan_*.md
```
* **Líneas 34 a 36:** Excluye los documentos de planeación de trabajo para mantener el repositorio enfocado estrictamente en la entrega de ingeniería de software.
