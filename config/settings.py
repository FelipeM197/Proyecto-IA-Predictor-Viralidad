import os
from pathlib import Path

# Anclaje dinamico para independizar la ejecucion del directorio de trabajo actual
BASE_DIR = Path(__file__).resolve().parent.parent

DATOS_DIR = BASE_DIR / "datos"
DATOS_CRUDOS_DIR = DATOS_DIR / "crudos"
DATOS_MEDIOS_DIR = DATOS_DIR / "medios"
DATOS_PROCESADOS_DIR = DATOS_DIR / "procesados"
MODELOS_DIR = BASE_DIR / "modelos_guardados"
LOGS_DIR = BASE_DIR / "logs"

DIRECTORIOS_PROYECTO = [
    DATOS_CRUDOS_DIR,
    DATOS_MEDIOS_DIR,
    DATOS_PROCESADOS_DIR,
    MODELOS_DIR,
    LOGS_DIR,
]

RUTA_MATRIZ_CRUDA = DATOS_PROCESADOS_DIR / "matriz_acustica_cruda.parquet"
RUTA_ESCALADOR = MODELOS_DIR / "escalador_acustico.joblib"
RUTA_MODELO_MLP = MODELOS_DIR / "modelo_mlp_final.joblib"
RUTA_MODELO_XGBOOST = MODELOS_DIR / "modelo_xgboost_final.joblib"
RUTA_REPORTE_MODELOS = MODELOS_DIR / "reporte_comparativo_modelos.json"
RUTA_UMBRALES_VIRALES = MODELOS_DIR / "umbrales_virales.json"

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
MAX_AUDIO_DURATION_SEC = float(os.getenv("MAX_AUDIO_DURATION_SEC", "30.0"))
N_FFT = 2048
HOP_LENGTH = 512
SILENCE_THRESHOLD_DB = 30

SEMILLA_ALEATORIA = 42
MIN_VIEWS = 100
PERCENTIL_INFERIOR_OUTLIERS = 0.01
PERCENTIL_SUPERIOR_OUTLIERS = 0.99
FACTOR_IQR = 1.5
PROPORCION_TRAIN = 0.70
PROPORCION_VAL = 0.15
PROPORCION_TEST = 0.15
BINS_ESTRATIFICACION = 5

COLUMNAS_SOCIALES = ["views", "likes", "comments", "shares"]
COLUMNA_TARGET = "engagement_rate"
COLUMNA_TARGET_LOG = "engagement_rate_log"

COLUMNAS_ACUSTICAS = [
    "tempo_bpm",
    "onset_strength_mean",
    "onset_strength_std",
    "beat_regularity",
    "rms_mean",
    "rms_std",
    "dynamic_range",
    "initial_silence_sec",
    "silence_ratio",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "zero_crossing_rate",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_8",
    "mfcc_9",
    "mfcc_10",
    "mfcc_11",
    "mfcc_12",
    "mfcc_13",
    "chroma_stft_mean",
]

ALPHAS_RIDGE = [0.01, 0.1, 1.0, 10.0, 100.0]
ALPHAS_ELASTICNET = [0.001, 0.01, 0.1, 1.0]
L1_RATIOS_ELASTICNET = [0.2, 0.5, 0.8]

CAPAS_OCULTAS_MLP = (128, 64, 32)
ALPHA_L2_MLP = 0.001
LR_INICIAL_MLP = 0.003
BATCH_SIZE_MLP = 64
MAX_ITER_MLP = 600
PACIENCIA_MLP = 15

N_ESTIMADORES_RF = 200
MAX_DEPTH_RF = 10
SUBSAMPLE_ARBOLES = 0.8

PARAM_GRID_XGB = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.08, 0.15],
    "n_estimators": [100, 200],
}
PLIEGUES_CV = 5

PERCENTIL_CORTE_VIRAL = 0.85
PERCENTILES_UMBRALES = (25, 50, 75)
TOP_K_ATRIBUTOS_SHAP = 3


# Agente LLM local (Fase 7): llama3.2:3b instalado en Ollama local
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TEMPERATURA = float(os.getenv("OLLAMA_TEMPERATURA", "0.2"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "400"))
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "60"))
OLLAMA_TIMEOUT_PING_SEC = float(os.getenv("OLLAMA_TIMEOUT_PING_SEC", "3"))

# Interfaz Streamlit (Fase 8)
FORMATOS_AUDIO_UI = ["mp3", "wav", "ogg", "flac", "m4a"]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
PUNTOS_ONDA_UI = 2000

# Agente: reglas deterministas de deteccion (Tarea 7.1)
UMBRAL_SILENCIO_CRITICO_SEC = 0.35
# Rangos de referencia del plan maestro, usados mientras no exista umbrales_virales.json
UMBRALES_REFERENCIA = {
    "tempo_bpm": {"p25": 115.0, "p50": 122.0, "p75": 132.0, "media": 122.0},
    "initial_silence_sec": {"p25": 0.0, "p50": 0.12, "p75": 0.35, "media": 0.16},
}

# Graficos de resultados (Tarea 8.2)
ENGAGEMENT_MAX_GAUGE_PCT = 20.0
NUM_BARRAS_SHAP = 10
RANGOS_RADAR = {  # (minimo, maximo) fisicos para normalizar cada eje a [0, 1]
    "tempo_bpm": (60.0, 180.0),
    "rms_mean": (0.0, 0.4),
    "spectral_centroid": (500.0, 5000.0),
    "beat_regularity": (0.0, 1.0),
    "initial_silence_sec": (0.0, 2.0),
}
