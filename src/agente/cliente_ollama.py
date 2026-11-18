"""Conector HTTP hacia el modelo local de Ollama (Fase 7, Tarea 7.3).

Usa unicamente la libreria estandar (urllib) para que cada linea de la
peticion sea explicable. Nunca lanza excepciones hacia la interfaz: ante
cualquier fallo devuelve un diccionario con exito=False para que el
formateador active el motor de respaldo basado en reglas.
"""
import json
import socket
import time
import urllib.error
import urllib.request

from config import settings
from config.logging_config import configurar_registro

logger = configurar_registro("agente.ollama")


class ClienteOllama:
    """Encapsula la comunicacion con la API REST de Ollama (puerto 11434)."""

    def __init__(
        self,
        host: str = settings.OLLAMA_HOST,
        modelo: str = settings.OLLAMA_MODEL,
        temperatura: float = settings.OLLAMA_TEMPERATURA,
        timeout_sec: float = settings.OLLAMA_TIMEOUT_SEC,
    ) -> None:
        self.host = host.rstrip("/")
        self.modelo = modelo
        self.temperatura = temperatura
        self.timeout_sec = timeout_sec

    # ------------------------------------------------------------------ #
    # Utilidades HTTP
    # ------------------------------------------------------------------ #
    def _peticion(self, ruta: str, cuerpo: dict | None, timeout: float) -> dict:
        """Ejecuta GET (sin cuerpo) o POST (con cuerpo JSON) y decodifica la respuesta."""
        datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
        solicitud = urllib.request.Request(
            url=f"{self.host}{ruta}",
            data=datos,
            headers={"Content-Type": "application/json"},
            method="POST" if datos is not None else "GET",
        )
        with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
            return json.loads(respuesta.read().decode("utf-8"))

    # ------------------------------------------------------------------ #
    # Diagnostico del servicio
    # ------------------------------------------------------------------ #
    def esta_disponible(self) -> bool:
        """Comprueba en pocos segundos si el servidor de Ollama responde."""
        try:
            self._peticion("/api/tags", None, settings.OLLAMA_TIMEOUT_PING_SEC)
            return True
        except (OSError, ValueError):  # URLError, timeout y ConnectionError heredan de OSError
            return False

    def modelo_instalado(self) -> bool:
        """Verifica que el modelo configurado (p. ej. llama3.2:1b) ya fue descargado."""
        try:
            datos = self._peticion("/api/tags", None, settings.OLLAMA_TIMEOUT_PING_SEC)
        except (OSError, ValueError):
            return False
        nombres = {m.get("name", "") for m in datos.get("models", [])}
        return self.modelo in nombres

    # ------------------------------------------------------------------ #
    # Generacion de respuestas
    # ------------------------------------------------------------------ #
    def generar_respuesta(self, prompt_usuario: str, prompt_sistema: str = "") -> dict:
        """Envia el prompt al endpoint /api/chat y devuelve un diccionario de control.

        Retorna: {"exito", "texto", "modelo", "duracion_seg", "motivo_error"}
        """
        mensajes = []
        if prompt_sistema:
            mensajes.append({"role": "system", "content": prompt_sistema})
        mensajes.append({"role": "user", "content": prompt_usuario})

        cuerpo = {
            "model": self.modelo,
            "messages": mensajes,
            "stream": False,  # respuesta completa en un solo JSON
            "options": {
                "temperature": self.temperatura,  # 0.2 reduce la dispersion
                "num_predict": settings.OLLAMA_MAX_TOKENS,
            },
        }

        inicio = time.perf_counter()
        try:
            datos = self._peticion("/api/chat", cuerpo, self.timeout_sec)
            texto = datos.get("message", {}).get("content", "").strip()
            if not texto:
                return self._fallo("El modelo devolvio una respuesta vacia", inicio)
            duracion = round(time.perf_counter() - inicio, 2)
            logger.info("Respuesta de %s generada en %.2f s", self.modelo, duracion)
            return {
                "exito": True,
                "texto": texto,
                "modelo": self.modelo,
                "duracion_seg": duracion,
                "motivo_error": None,
            }
        except urllib.error.HTTPError as err:
            # 404 suele indicar que el modelo no esta descargado (ollama pull)
            return self._fallo(f"HTTP {err.code}: {err.reason}", inicio)
        except (socket.timeout, TimeoutError):
            return self._fallo(f"Tiempo de espera agotado ({self.timeout_sec}s)", inicio)
        except urllib.error.URLError as err:
            return self._fallo(f"Ollama no disponible en {self.host}: {err.reason}", inicio)
        except (json.JSONDecodeError, OSError) as err:
            return self._fallo(f"Respuesta invalida o error de sistema: {err}", inicio)

    def _fallo(self, motivo: str, inicio: float) -> dict:
        logger.warning("Fallo de Ollama: %s", motivo)
        return {
            "exito": False,
            "texto": "",
            "modelo": self.modelo,
            "duracion_seg": round(time.perf_counter() - inicio, 2),
            "motivo_error": motivo,
        }


def generar_respuesta(prompt_usuario: str, prompt_sistema: str = "") -> dict:
    """Atajo funcional para usar el cliente sin instanciar la clase."""
    return ClienteOllama().generar_respuesta(prompt_usuario, prompt_sistema)
