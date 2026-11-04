import logging
import sys
from logging.handlers import RotatingFileHandler
from config import settings


def configurar_registro(nombre: str = "sistema") -> logging.Logger:
    logger = logging.getLogger(nombre)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_log = settings.LOGS_DIR / "sistema.log"

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formato)
    consola.setLevel(logging.INFO)
    logger.addHandler(consola)

    # Rotacion a 5 MB para evitar saturacion de disco en ejecuciones extensas
    archivo = RotatingFileHandler(
        ruta_log, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    archivo.setFormatter(formato)
    archivo.setLevel(logging.INFO)
    logger.addHandler(archivo)

    return logger
