import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor

from config import settings


class RedNeuronalAcustica:
    """Perceptron multicapa tabular para estimar engagement en escala logaritmica."""

    def __init__(
        self,
        capas_ocultas: tuple[int, ...] = settings.CAPAS_OCULTAS_MLP,
        alpha_l2: float = settings.ALPHA_L2_MLP,
        tasa_aprendizaje: float = settings.LR_INICIAL_MLP,
        tamano_lote: int = settings.BATCH_SIZE_MLP,
        max_iter: int = settings.MAX_ITER_MLP,
        fraccion_val: float = settings.PROPORCION_VAL,
        paciencia: int = settings.PACIENCIA_MLP,
        semilla: int = settings.SEMILLA_ALEATORIA,
    ) -> None:
        self.capas_ocultas = capas_ocultas
        self.alpha_l2 = alpha_l2
        self.tasa_aprendizaje = tasa_aprendizaje
        self.tamano_lote = tamano_lote
        self.max_iter = max_iter
        self.fraccion_val = fraccion_val
        self.paciencia = paciencia
        self.semilla = semilla
        self.modelo = self._crear_estimador(tamano_lote)

    def _crear_estimador(self, batch_efectivo: int) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=self.capas_ocultas,
            activation="relu",
            solver="adam",
            alpha=self.alpha_l2,
            learning_rate_init=self.tasa_aprendizaje,
            batch_size=batch_efectivo,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=self.fraccion_val,
            n_iter_no_change=self.paciencia,
            random_state=self.semilla,
        )

    def entrenar(self, x_train, y_train, x_val=None, y_val=None) -> dict[str, object]:
        n_muestras = len(x_train)
        if n_muestras < 2:
            raise ValueError("Se requieren al menos 2 muestras para entrenar el MLP.")

        # early_stopping descuenta fraccion_val internamente; acotar el lote evita advertencias de sklearn
        n_efectivo = max(1, n_muestras - math.ceil(n_muestras * self.fraccion_val))
        batch_efectivo = max(1, min(self.tamano_lote, n_efectivo))
        if batch_efectivo != self.modelo.batch_size:
            self.modelo = self._crear_estimador(batch_efectivo)

        self.modelo.fit(x_train, y_train)

        resumen: dict[str, object] = {
            "iteraciones": int(self.modelo.n_iter_),
            "perdida_final": float(self.modelo.loss_),
            "tamano_lote_efectivo": int(batch_efectivo),
        }
        if getattr(self.modelo, "best_validation_score_", None) is not None:
            resumen["mejor_r2_val_interna"] = float(self.modelo.best_validation_score_)

        if x_val is not None and y_val is not None:
            predicciones_val = self.predecir(x_val)
            resumen["metricas_validacion"] = {
                "MAE": float(mean_absolute_error(y_val, predicciones_val)),
                "RMSE": float(root_mean_squared_error(y_val, predicciones_val)),
                "R2": float(r2_score(y_val, predicciones_val)),
            }

        return resumen

    def predecir(self, x) -> np.ndarray:
        return np.asarray(self.modelo.predict(x), dtype=float)

    def guardar(self, ruta_salida: str | Path = settings.RUTA_MODELO_MLP) -> Path:
        ruta = Path(ruta_salida)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        # Se serializa el estimador nativo para permitir .predict directo tras joblib.load
        joblib.dump(self.modelo, ruta)
        return ruta

    @classmethod
    def cargar(
        cls, ruta_modelo: str | Path = settings.RUTA_MODELO_MLP
    ) -> "RedNeuronalAcustica":
        ruta = Path(ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontro el modelo MLP en: {ruta}")

        objeto_cargado = joblib.load(ruta)
        if isinstance(objeto_cargado, cls):
            return objeto_cargado

        instancia = cls()
        instancia.modelo = objeto_cargado
        return instancia
