from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import settings


def crear_random_forest(
    n_estimadores: int = settings.N_ESTIMADORES_RF,
    profundidad_max: int | None = settings.MAX_DEPTH_RF,
    submuestra: float | None = settings.SUBSAMPLE_ARBOLES,
    semilla: int = settings.SEMILLA_ALEATORIA,
    n_trabajos: int = 1,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimadores,
        max_depth=profundidad_max,
        max_samples=submuestra,
        random_state=semilla,
        n_jobs=n_trabajos,
    )


def entrenar_random_forest(
    x_train,
    y_train,
    n_estimadores: int = settings.N_ESTIMADORES_RF,
    profundidad_max: int | None = settings.MAX_DEPTH_RF,
    submuestra: float | None = settings.SUBSAMPLE_ARBOLES,
    semilla: int = settings.SEMILLA_ALEATORIA,
    n_trabajos: int = 1,
) -> RandomForestRegressor:
    modelo = crear_random_forest(
        n_estimadores=n_estimadores,
        profundidad_max=profundidad_max,
        submuestra=submuestra,
        semilla=semilla,
        n_trabajos=n_trabajos,
    )
    modelo.fit(x_train, y_train)
    return modelo


def crear_xgboost(
    profundidad_max: int = settings.PARAM_GRID_XGB["max_depth"][1],
    tasa_aprendizaje: float = settings.PARAM_GRID_XGB["learning_rate"][1],
    n_estimadores: int = settings.PARAM_GRID_XGB["n_estimators"][-1],
    submuestra: float = settings.SUBSAMPLE_ARBOLES,
    muestreo_columnas: float = settings.SUBSAMPLE_ARBOLES,
    semilla: int = settings.SEMILLA_ALEATORIA,
    n_trabajos: int = 1,
) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        max_depth=profundidad_max,
        learning_rate=tasa_aprendizaje,
        n_estimators=n_estimadores,
        subsample=submuestra,
        colsample_bytree=muestreo_columnas,
        random_state=semilla,
        n_jobs=n_trabajos,
    )


def entrenar_xgboost(
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    profundidad_max: int = settings.PARAM_GRID_XGB["max_depth"][1],
    tasa_aprendizaje: float = settings.PARAM_GRID_XGB["learning_rate"][1],
    n_estimadores: int = settings.PARAM_GRID_XGB["n_estimators"][-1],
    submuestra: float = settings.SUBSAMPLE_ARBOLES,
    muestreo_columnas: float = settings.SUBSAMPLE_ARBOLES,
    semilla: int = settings.SEMILLA_ALEATORIA,
    n_trabajos: int = 1,
) -> XGBRegressor:
    modelo = crear_xgboost(
        profundidad_max=profundidad_max,
        tasa_aprendizaje=tasa_aprendizaje,
        n_estimadores=n_estimadores,
        submuestra=submuestra,
        muestreo_columnas=muestreo_columnas,
        semilla=semilla,
        n_trabajos=n_trabajos,
    )
    if x_val is not None and y_val is not None:
        modelo.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        return modelo

    modelo.fit(x_train, y_train)
    return modelo
