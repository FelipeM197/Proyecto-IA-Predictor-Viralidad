"""
Modulo Central de Configuracion y Parametros del Sistema.
Define rutas dinamicas, constantes DSP y configuracion de modelos.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Cargar variables de entorno si existe archivo .env
load_dotenv()

# ==============================================================================
# 1. RUTAS DINAMICAS DEL SISTEMA
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Carpetas de Datos
DATA_DIR: Path = BASE_DIR / "datos"
DATA_RAW_DIR: Path = DATA_DIR / "crudos"
DATA_MEDIA_DIR: Path = DATA_DIR / "medios"
DATA_PROCESSED_DIR: Path = DATA_DIR / "procesados"
FAILED_URLS_LOG: Path = DATA_DIR / "fallos_descarga.csv"

# Carpetas de Artefactos y Modelos
MODELS_DIR: Path = BASE_DIR / "modelos_guardados"
LOGS_DIR: Path = BASE_DIR / "logs"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
TESTS_DIR: Path = BASE_DIR / "tests"

# ==============================================================================
# 2. PARAMETROS DE PROCESAMIENTO DIGITAL DE SENALES (DSP - Librosa)
# ==============================================================================
# Frecuencia de muestreo (22050 Hz captura frecuencias hasta 11 kHz segun Nyquist)
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", 22050))

# Duracion maxima en segundos analizada por pista
MAX_DURATION_SEC: float = float(os.getenv("MAX_AUDIO_DURATION_SEC", 30.0))

# Ventanas de la Transformada Rapida de Fourier (STFT)
N_FFT: int = 2048
HOP_LENGTH: int = 512

# Parametros timbricos y de silencio
N_MFCC: int = 13
SILENCE_TOP_DB: int = 30  # Umbral en dB por debajo del pico para detectar silencio

# ==============================================================================
# 3. PARAMETROS DE MODELADO PREDICTIVO (Machine Learning)
# ==============================================================================
RANDOM_SEED: int = 42
MIN_VIEWS_FILTER: int = 100  # Descartar publicaciones con vistas insignificantes

# ==============================================================================
# 4. CONFIGURACION DEL AGENTE GENERATIVO (Ollama Local)
# ==============================================================================
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TEMPERATURE: float = 0.2  # Respuestas directas, tecnicas y sin alucinacion
