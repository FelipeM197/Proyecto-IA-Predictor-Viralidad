# Contexto y Pendientes: Integrante 3 (Agente LLM Local e Interfaz Gráfica)

> **Documento de trabajo individual para el Integrante 3 (`Persona 3`)**  
> Basado en la arquitectura formal de **`plan_maestro_detallado.md`** (Fases 1, 7, 8, 9 y 10).

---

## 1. Responsabilidad Formal en el Proyecto
El Integrante 3 es el responsable de la integración del Modelo de Lenguaje de Gran Escala local (LLM mediante Ollama con Llama 3), la ingeniería de prompts anclada en valores numéricos y métricas SHAP, la implementación del plan de contingencia fuera de línea (*offline fallback*), el desarrollo de la aplicación web interactiva en Streamlit y la orquestación completa de la inferencia desde la subida del archivo o URL hasta el reporte al usuario.

---

## 2. Lo que ya está completado y verificado (Commits `cd686ad` y `eb6429b`)

* **Diagnóstico de Entorno (`scripts/` — Fase 1):**
  - `check_environment.py`: Verificación automatizada de versión de Python, dependencias requeridas, binario de FFmpeg y disponibilidad del puerto 11434 de Ollama.
  - `guia_ffmpeg.md`: Manual técnico de instalación y configuración de variables de entorno para FFmpeg en Windows.
* **Agente de Diagnóstico y Recomendaciones (`src/agente/` — Fase 7):**
  - `detector_deficiencias.py`: Comparación determinista del vector acústico contra `umbrales_virales.json` y ordenamiento de criticidad cruzado con la magnitud e impacto de los valores SHAP.
  - `constructor_prompts.py`: Ensamble de prompts con instrucción de sistema estricta y anclaje numérico de variables, prohibiendo alucinaciones y forzando tres recomendaciones concretas de producción.
  - `cliente_ollama.py`: Cliente HTTP hacia `http://localhost:11434/api/generate` con temperatura de 0.2, timeout controlado de 15 segundos y manejo de excepciones de conexión.
  - `formateador_respuesta.py`: Sanitización del texto generado y motor heurístico de contingencia que genera sugerencias deterministas si Ollama no está en ejecución.
* **Interfaz Gráfica y Orquestación (`app/` — Fase 8):**
  - `main.py`: Punto de entrada de Streamlit con configuración de página, carga de estilos CSS y gestión de estado con `st.session_state`.
  - `estilos.css`: Hoja de estilos sin librerías externas para diseño profesional y legible.
  - `componentes/panel_entrada.py`: Selector de entrada dual (subida de MP3/WAV o enlace web de TikTok/YouTube).
  - `componentes/reproductor_onda.py`: Visualización interactiva de la forma de onda temporal con delimitación visual del silencio inicial.
  - `componentes/graficos_plotly.py`: Gráfico de radar con cinco ejes acústicos normalizados, velocímetro de tasa de interacción y gráfico de barras de impacto SHAP.
  - `componentes/panel_recomendaciones.py`: Despliegue de tarjetas de sugerencias con semáforo de criticidad y estado del motor (Ollama local o contingencia heurística).
  - `servicios/orquestador_inferencia.py`: Flujo integral de inferencia que coordina validación, extracción acústica, normalización, predicción y consulta al agente generativo.
* **Pruebas Automatizadas (Fase 10):**
  - `tests/test_agente_ollama.py`: Cobertura de detección de deficiencias, formato de prompts y disparo inmediato del plan de respaldo ante desconexión (aprobado con código de salida 0).

---

## 3. Lo que falta exactamente por hacer (Lista de Pendientes Críticos)

Aunque el código está subido y opera correctamente en modo sintético, quedan pendientes las siguientes actividades obligatorias para la entrega final:

1. **Cuaderno de Documentación de Interfaz (Fase 9, Línea 484 del Plan Maestro):**
   - El Plan Maestro exige en la Fase 9 la entrega de un cuaderno en `notebooks/` que documente el flujo completo de inferencia y las salidas de la aplicación.
   - **Acción requerida:** Crear el cuaderno `notebooks/05_simulacion_orquestacion_interfaz.ipynb` donde se ejecute de extremo a extremo el `OrquestadorInferencia` con un audio sintético o de prueba, graficando las salidas generadas (forma de onda, radar de Plotly y reporte textual del agente) para sustentar el pipeline de frontend ante el tribunal.
2. **Validación del Runtime Local de Ollama:**
   - La aplicación cuenta con un respaldo determinista (`modo_offline`), pero para la defensa en vivo ante el jurado se requiere el modelo generativo real.
   - **Acción requerida:** Comprobar que el servicio local de Ollama esté instalado y ejecutándose en segundo plano con el modelo base descargado:
     ```bash
     ollama run llama3:8b
     ```
     Verificar que `cliente_ollama.verificar_servicio()` retorne `True` y que el tiempo de respuesta no supere los 10 segundos en la máquina de presentación.
3. **Desactivar el Modo Demostración al recibir los Modelos Entrenados:**
   - En `app/servicios/orquestador_inferencia.py`, la propiedad `modo_demo` se activa automáticamente si no existen los artefactos serializados en `modelos_guardados/`.
   - **Acción requerida:** Una vez que la Persona 2 genere y entregue los archivos `escalador_acustico.joblib`, `modelo_xgboost_final.joblib` y `umbrales_virales.json`, la Persona 3 debe verificar que el orquestador cargue los modelos reales con el decorador `@st.cache_resource`, validando que las predicciones provengan del modelo entrenado y no de la función sintética de reserva.
4. **Preparación para la Defensa Oral en Vivo (Tarea 10.2 del Plan Maestro):**
   - **Pregunta y prueba que le pedirá el jurado en vivo:**
     *«Altere la instrucción del sistema en `src/agente/constructor_prompts.py` para obligar al modelo a redactar dos recomendaciones en lugar de tres y demuestre el cambio inmediatamente en la pantalla de Streamlit.»*
   - **Cómo responder y ejecutar:**
     1. Abrir `src/agente/constructor_prompts.py`.
     2. Modificar la plantilla del prompt: cambiar *«Redacta exactamente 3 recomendaciones técnicas»* por *«Redacta exactamente 2 recomendaciones técnicas numeradas»*.
     3. Presionar el botón de análisis en la aplicación web (`streamlit run app/main.py`) y mostrar que el contenedor de recomendaciones renderiza dos tarjetas en lugar de tres.
     4. Argumentar: *«Al acotar la salida a dos recomendaciones forzamos al modelo a priorizar exclusivamente las dos anomalías con mayor magnitud de impacto negativo según los valores SHAP, eliminando consejos secundarios y facilitando decisiones rápidas de edición para el creador.»*

---

## 4. Contratos de Entrega e Interconexión con el Equipo
* **Dependencia con Persona 1 (Ingesta y DSP):**
  - La aplicación web invoca `src.caracteristicas.constructor_matriz.extraer_vector_acustico_desde_onda()` y `src.ingesta.cliente_ytdlp.descargar_audio_youtube()`.
* **Dependencia con Persona 2 (Modelos y Explicabilidad):**
  - Requiere los artefactos `modelos_guardados/escalador_acustico.joblib`, `modelos_guardados/modelo_xgboost_final.joblib` y `modelos_guardados/umbrales_virales.json` para abandonar el modo de reserva y operar con inferencia real.
* **Entregable hacia la Sustentación General:**
  - Aplicación operativa mediante `streamlit run app/main.py`, cuaderno de sustentación `notebooks/05_simulacion_orquestacion_interfaz.ipynb` y dominio de la contingencia ante fallos de conexión a Internet.
