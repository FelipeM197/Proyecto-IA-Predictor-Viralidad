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
