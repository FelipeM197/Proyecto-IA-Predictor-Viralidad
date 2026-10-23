"""
Script de Diagnostico y Auditoria de Entorno.
Verifica:
1. Version de Python (>= 3.10)
2. Presencia binaria de FFmpeg en el sistema operativo
3. Estado y modelos del servidor local de Ollama
"""

import sys
import shutil
import subprocess
import json
import urllib.request
import urllib.error
from pathlib import Path

def verificar_python() -> bool:
    print("\n--- 1. Verificacion de Python ---")
    v = sys.version_info
    print(f"Version detectada: {v.major}.{v.minor}.{v.micro}")
    if v.major >= 3 and v.minor >= 10:
        print("[OK] Python cumple con los requisitos (>= 3.10)")
        return True
    else:
        print("[FAIL] Se requiere Python 3.10 o superior")
        return False

def verificar_ffmpeg() -> bool:
    print("\n--- 2. Verificacion de FFmpeg ---")
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("[FAIL] FFmpeg no se encontro en el PATH del sistema.")
        print("       -> Para instalarlo en Windows ejecuta en PowerShell:")
        print("          winget install Gyan.FFmpeg")
        print("       -> Y reinicia la terminal para actualizar las variables de entorno.")
        return False
    
    try:
        res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        linea = res.stdout.split("\n")[0] if res.stdout else "Detectado"
        print(f"[OK] FFmpeg detectado: {linea}")
        return True
    except Exception as e:
        print(f"[FAIL] Error al ejecutar ffmpeg: {e}")
        return False

def verificar_ollama() -> bool:
    print("\n--- 3. Verificacion de Servidor Local Ollama ---")
    url = "http://localhost:11434/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Auditor-Entorno"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            modelos = [m.get("name") for m in data.get("models", [])]
            print(f"[OK] Servidor Ollama respondiendo en localhost:11434")
            print(f"     Modelos instalados detectados: {modelos if modelos else 'Ninguno (ejecuta: ollama pull llama3)'}")
            return True
    except urllib.error.URLError:
        print("[ALERTA] No se pudo conectar a http://localhost:11434.")
        print("         -> Asegurate de tener Ollama abierto o ejecutando en segundo plano.")
        print("         -> Descarga: https://ollama.com/")
        return False
    except Exception as e:
        print(f"[ALERTA] Error al consultar Ollama: {e}")
        return False

def main():
    print("=" * 60)
    print("AUDITORIA DE ENTORNO: PREDICTOR DE ENGAGEMENT POR AUDIO")
    print("=" * 60)
    
    py_ok = verificar_python()
    ffmpeg_ok = verificar_ffmpeg()
    ollama_ok = verificar_ollama()
    
    print("\n" + "=" * 60)
    print("RESUMEN DE DIAGNOSTICO:")
    print(f" - Python: {'LISTO' if py_ok else 'PENDIENTE'}")
    print(f" - FFmpeg: {'LISTO' if ffmpeg_ok else 'PENDIENTE DE INSTALACION'}")
    print(f" - Ollama: {'LISTO' if ollama_ok else 'APAGADO / NO INSTALADO'}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
