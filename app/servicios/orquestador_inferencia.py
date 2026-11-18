"""Orquestador del flujo de inferencia en tiempo real (Fase 8, Tarea 8.1).

Secuencia: audio -> validacion -> DSP (28 variables) -> escalado -> XGBoost -> inversion
log -> SHAP -> deficiencias -> diagnostico LLM. Los temporales se borran siempre.
Sin los artefactos de Persona 2 (.joblib, umbrales, SHAP) opera en modo demostracion.
"""
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from config import settings
from config.logging_config import configurar_registro
from src.agente.cliente_ollama import ClienteOllama
from src.agente.detector_deficiencias import evaluar_desviaciones
from src.agente.formateador_respuesta import redactar_diagnostico
from src.caracteristicas.constructor_matriz import extraer_vector_acustico_desde_onda
from src.caracteristicas.preprocesador_onda import cargar_y_estandarizar_onda
from src.ingesta.cliente_ytdlp import descargar_audio_url
from src.ingesta.validador_medios import es_archivo_valido
from src.preprocesamiento.calculo_target import invertir_target_log
from src.preprocesamiento.escalado_features import cargar_escalador, transformar_features

logger = configurar_registro("app.orquestador")


@dataclass
class ResultadoInferencia:
    exito: bool
    mensaje_error: str | None = None
    vector: dict = field(default_factory=dict)
    engagement_estimado: float | None = None  # fraccion: 0.085 equivale a 8.5 %
    explicacion_shap: dict | None = None
    deficiencias: list = field(default_factory=list)
    diagnostico: dict = field(default_factory=dict)
    umbrales: dict = field(default_factory=dict)
    modo_demo: bool = False
    tiempos: dict = field(default_factory=dict)


class MotorInferencia:
    """Carga los recursos pesados una vez y ejecuta la cadena completa por consulta."""

    def __init__(self, cliente_llm: ClienteOllama | None = None) -> None:
        self.cliente_llm = cliente_llm or ClienteOllama()
        self.escalador = self._cargar(settings.RUTA_ESCALADOR, cargar_escalador)
        self.modelo = self._cargar(settings.RUTA_MODELO_XGBOOST, joblib.load)
        self.umbrales, self.umbrales_oficiales = self._cargar_umbrales()
        self.explicar = self._importar_shap()

    @property
    def modo_demo(self) -> bool:
        return self.escalador is None or self.modelo is None

    # ---------------------- carga de recursos ---------------------- #
    @staticmethod
    def _cargar(ruta: Path, cargador):
        if not Path(ruta).is_file():
            logger.warning("Artefacto no encontrado (modo demo): %s", ruta.name)
            return None
        try:
            return cargador(ruta)
        except Exception as err:
            logger.error("No se pudo cargar %s: %s", ruta.name, err)
        return None

    @staticmethod
    def _cargar_umbrales() -> tuple[dict, bool]:
        ruta = settings.RUTA_UMBRALES_VIRALES
        if ruta.is_file():
            try:
                return json.loads(ruta.read_text(encoding="utf-8")), True
            except (json.JSONDecodeError, OSError) as err:
                logger.error("umbrales_virales.json ilegible: %s", err)
        return dict(settings.UMBRALES_REFERENCIA), False

    @staticmethod
    def _importar_shap():
        try:
            from src.modelos.explicabilidad_shap import explicar_prediccion_individual
            return explicar_prediccion_individual
        except ImportError as err:
            logger.warning("Explicabilidad SHAP no disponible aun: %s", err)
            return None

    # ------------------------- inferencia -------------------------- #
    def _predecir(self, vector: dict) -> tuple[float | None, dict | None]:
        if self.modo_demo:
            return None, None
        try:
            df = pd.DataFrame([vector], columns=settings.COLUMNAS_ACUSTICAS)
            cols = settings.COLUMNAS_ACUSTICAS
            escalado = transformar_features(df, self.escalador, cols)[cols]
            engagement = float(invertir_target_log(float(self.modelo.predict(escalado)[0])))
        except Exception as err:
            logger.error("Fallo en la prediccion: %s", err)
            return None, None
        try:
            explicacion = self.explicar(self.modelo, escalado) if self.explicar else None
        except Exception as err:
            logger.error("Fallo en SHAP: %s", err)
            explicacion = None
        return engagement, explicacion

    def analizar_archivo(self, ruta: Path) -> ResultadoInferencia:
        inicio = time.perf_counter()
        valido, motivo = es_archivo_valido(ruta)
        if not valido:
            return ResultadoInferencia(False, f"Audio no valido: {motivo}")
        try:
            onda, sr = cargar_y_estandarizar_onda(ruta)
            vector = extraer_vector_acustico_desde_onda(onda, sr=sr)
        except Exception as err:
            logger.error("Fallo en DSP: %s", err)
            return ResultadoInferencia(False, f"No se pudo procesar la senal: {err}")
        t_dsp = time.perf_counter()
        engagement, explicacion = self._predecir(vector)
        t_modelo = time.perf_counter()
        shap_vals = explicacion.get("contribuciones") if explicacion else None
        favorables = explicacion.get("atributos_favorables") if explicacion else None
        deficiencias = evaluar_desviaciones(vector, self.umbrales, shap_vals)
        diagnostico = redactar_diagnostico(deficiencias, self.cliente_llm, engagement, favorables)
        fin = time.perf_counter()

        tiempos = {"dsp": t_dsp - inicio, "modelo": t_modelo - t_dsp,
                   "agente": fin - t_modelo, "total": fin - inicio}
        return ResultadoInferencia(True, None, vector, engagement, explicacion, deficiencias,
                                   diagnostico, self.umbrales, self.modo_demo, tiempos)

    def analizar_bytes(self, datos: bytes, nombre: str) -> ResultadoInferencia:
        """Analiza un archivo subido desde la interfaz usando un temporal que se borra."""
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / f"consulta{Path(nombre).suffix or '.mp3'}"
            ruta.write_bytes(datos)
            return self.analizar_archivo(ruta)

    def analizar_url(self, url: str) -> tuple[dict | None, ResultadoInferencia]:
        """Descarga el audio del enlace (Persona 1), lo analiza y devuelve bytes + resultado."""
        with tempfile.TemporaryDirectory() as carpeta:
            descarga = descargar_audio_url(url, "consulta_web", Path(carpeta))
            if not descarga["exito"]:
                return None, ResultadoInferencia(
                    False, f"No se pudo obtener el audio del enlace. {descarga['motivo_error']}")
            ruta = Path(descarga["ruta_audio"])
            audio = {"nombre": url, "bytes": ruta.read_bytes(), "formato": ruta.suffix.lstrip(".")}
            return audio, self.analizar_archivo(ruta)
