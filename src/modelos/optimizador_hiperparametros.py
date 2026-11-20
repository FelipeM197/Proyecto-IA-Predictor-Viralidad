from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from config import settings
from src.modelos.modelo_arboles import crear_xgboost


def optimizar_xgboost_cv(
    x_train,
    y_train,
    param_grid: dict | None = None,
    pliegues_cv: int = settings.PLIEGUES_CV,
    semilla: int = settings.SEMILLA_ALEATORIA,
) -> tuple[XGBRegressor, dict[str, object]]:
    n_muestras = len(x_train)
    if n_muestras < 2:
        raise ValueError("Se requieren al menos 2 muestras para ejecutar validacion cruzada.")

    rejilla = settings.PARAM_GRID_XGB if param_grid is None else param_grid
    pliegues_efectivos = max(2, min(pliegues_cv, n_muestras))
    particionador = KFold(n_splits=pliegues_efectivos, shuffle=True, random_state=semilla)

    # n_jobs=1 evita conflictos de subprocesos entre joblib y Windows
    busqueda = GridSearchCV(
        estimator=crear_xgboost(semilla=semilla),
        param_grid=rejilla,
        scoring="neg_root_mean_squared_error",
        cv=particionador,
        n_jobs=1,
        refit=True,
    )
    busqueda.fit(x_train, y_train)

    mejores_parametros = {
        clave: valor.item() if hasattr(valor, "item") else valor
        for clave, valor in busqueda.best_params_.items()
    }
    resumen = {
        "mejores_parametros": mejores_parametros,
        "mejor_rmse_cv": float(-busqueda.best_score_),
        "pliegues_cv": int(pliegues_efectivos),
        "total_combinaciones": int(len(busqueda.cv_results_["params"])),
    }
    return busqueda.best_estimator_, resumen
