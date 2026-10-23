"""
Configuracion del Sistema de Logging Centralizado.
Escribe simultaneamente en la consola y en 'logs/sistema.log'.
"""

import logging
import sys
from pathlib import Path
from config.settings import LOGS_DIR

def setup_logger(name: str = "predictor_engagement") -> logging.Logger:
    """
    Configura y devuelve un logger con manejadores de consola y archivo rotativo.
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers si ya esta configurado
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Asegurar que exista la carpeta de logs
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "sistema.log"
    
    # Formato profesional con timestamp, nivel y modulo
    formato = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formato)
    logger.addHandler(console_handler)
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formato)
    logger.addHandler(file_handler)
    
    return logger
