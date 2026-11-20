# Contexto y Pendientes: Integrante 2 (Estadística, Modelos y Explicabilidad SHAP)

> **Documento de trabajo individual para el Integrante 2 (`Persona 2`)**  
> Basado en la arquitectura formal de **`plan_maestro_detallado.md`** (Fases 1, 4, 5, 6, 9 y 10).

---

## 1. Responsabilidad Formal en el Proyecto
El Integrante 2 es el responsable de la formulación matemática de la variable objetivo (*Engagement Rate*), el tratamiento estadístico de valores anómalos, el particionamiento estratificado, el escalado sin fuga de información (*data leakage*), el entrenamiento de las cuatro familias de modelos predictivos (incluyendo la red neuronal MLP propia) y la explicabilidad matemática local mediante valores SHAP.

---

## 2. Lo que ya está completado y verificado (Fases 1 y 4)

* **Infraestructura Base (`config/` y `scripts/` — Fase 1):**
  - `config/settings.py`: Centralización de rutas con `pathlib.Path`, parámetros DSP a $22\,050\text{ Hz}$ y cero números mágicos.
  - `config/logging_config.py`: Registro dual en consola y archivo rotativo a 5 MB.
  - `scripts/init_dirs.py` y `config/README.md`: Inicialización de directorios físicos y justificación del Teorema de Nyquist-Shannon.
* **Preprocesamiento Estadístico (`src/preprocesamiento/` — Fase 4, Commit `dfc21c3`):**
  - `calculo_target.py`: Filtro de $100$ visualizaciones mínimas, formulación de la tasa relativa $\frac{\text{likes} + \text{comments} + \text{shares}}{\text{views}}$, corrección de asimetría con $\ln(1+x)$ e inversión a escala real con $e^y - 1$.
  - `limpieza_outliers.py`: Limpieza de indeterminaciones matemáticas (`inf` y `-inf`), acotamiento suave por rango intercuartil ($1.5 \cdot \text{IQR}$) hacia percentiles $P_1$ y $P_{99}$ e imputación por mediana.
  - `divisor_dataset.py`: Partición estratificada por cuantiles continuos en 70 % entrenamiento, 15 % validación y 15 % prueba.
  - `escalado_features.py`: Ajuste de `StandardScaler` exclusivamente sobre entrenamiento para impedir fuga de datos.
  - Suites de pruebas unitarias: `tests/test_calculo_target.py`, `tests/test_limpieza_divisor_escalado.py` y `tests/test_preprocesamiento.py` aprobadas con código de salida 0.

---

## 3. Lo que falta exactamente por hacer (Lista de Pendientes Críticos)

Esta es la sección central que desbloqueará el funcionamiento completo de la aplicación:

1. **Construir y Entrenar los Modelos Propios (`src/modelos/` — Fase 5):**
   * **`modelo_baseline.py` (Tarea 5.1):** Regresión lineal con regularización `RidgeCV` ($L_2$) y `ElasticNetCV` ($L_1 + L_2$).
   * **`modelo_red_neuronal.py` (Tarea 5.2):** Perceptrón Multicapa (`MLPRegressor`) con topología $(128 \to 64 \to 32 \to 1)$, activación ReLU, regularización L2 y parada temprana. Debe generar: `modelos_guardados/modelo_mlp_final.joblib`.
   * **`modelo_arboles.py` (Tarea 5.3):** Random Forest (200 árboles) y XGBoost Regressor.
   * **`optimizador_hiperparametros.py` (Tarea 5.4):** Búsqueda sistemática con validación cruzada de 5 pliegues para XGBoost.
   * **`evaluador_metricas.py` (Tarea 5.4):** Evaluación sobre el conjunto de prueba independiente calculando MAE, RMSE y $R^2$. Debe generar: `modelos_guardados/modelo_xgboost_final.joblib` y `modelos_guardados/reporte_comparativo_modelos.json`.
2. **Explicabilidad Matemática y Umbrales Virales (`src/modelos/explicabilidad_shap.py` — Fase 6):**
   * Implementar `TreeExplainer` para desglosar cada predicción en contribuciones aditivas exactas, aislando el top 3 de variables que más suman (`atributos_favorables`) y el top 3 que más restan (`atributos_perjudiciales`).
   * Aislar el 15 % superior de publicaciones virales para calcular los percentiles $P_{25}, P_{50}, P_{75}$ de cada variable física en su escala original. Debe generar: `modelos_guardados/umbrales_virales.json`.
3. **Cuadernos Académicos de Sustento (Fase 9):**
   * `notebooks/03_analisis_estadistico_target.ipynb`: Pruebas de normalidad de Shapiro-Wilk, asimetría antes/después de $\ln(1+x)$ y correlaciones de Spearman con las 28 variables acústicas.
   * `notebooks/04_benchmark_modelos_y_shap.ipynb`: Comparativa gráfica de métricas de los 4 modelos y gráficos globales de interpretabilidad SHAP.
4. **Prueba de Inferencia y Modificación en Vivo ante el Jurado (Fase 10):**
   * `tests/test_modelos_inferencia.py`: Pasar un vector sintético de 28 dimensiones al modelo cargado de disco y verificar que retorne un flotante positivo finito.
   * **Simulacro de modificación en vivo (Tarea 10.2 del Plan Maestro):**
     - El jurado pedirá: *«Cambie el filtro de visualizaciones de 100 a 500 vistas en vivo y explique qué ocurre matemáticamente con la varianza del error»*.
     - **Respuesta:** *«Al elevar el corte a 500 vistas eliminamos publicaciones de bajo alcance donde el ruido muestral domina sobre el patrón acústico. Como la varianza del cociente escala como $\frac{1}{\text{views}}$, el resultado es una reducción de la varianza del error residual (menor RMSE en validación) a cambio de una ligera reducción muestral en el conjunto de entrenamiento.»*

---

## 4. Contratos de Entrega hacia la Persona 3 (Streamlit y LLM)
* `modelos_guardados/escalador_acustico.joblib` (objeto `StandardScaler`).
* `modelos_guardados/modelo_xgboost_final.joblib` (modelo predictivo ganador).
* `modelos_guardados/umbrales_virales.json` (percentiles virales para alimentar el detector de deficiencias).
* Función `invertir_target_log()` para mostrar porcentajes reales en el velocímetro de la UI.
