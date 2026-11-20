from sklearn.linear_model import ElasticNetCV, RidgeCV

from config import settings


def entrenar_ridge(
    x_train,
    y_train,
    alphas=None,
    pliegues_cv: int | None = None,
) -> RidgeCV:
    n_muestras = len(x_train)
    if n_muestras < 2:
        raise ValueError("Se requieren al menos 2 muestras para ajustar RidgeCV.")

    lista_alphas = settings.ALPHAS_RIDGE if alphas is None else alphas
    pliegues = settings.PLIEGUES_CV if pliegues_cv is None else pliegues_cv
    # Acota pliegues si la particion de prueba unitaria es menor que PLIEGUES_CV
    cv_efectivo = max(2, min(pliegues, n_muestras))

    modelo = RidgeCV(alphas=lista_alphas, cv=cv_efectivo)
    modelo.fit(x_train, y_train)
    return modelo


def entrenar_elasticnet(
    x_train,
    y_train,
    alphas=None,
    l1_ratios=None,
    pliegues_cv: int | None = None,
    semilla: int | None = None,
) -> ElasticNetCV:
    n_muestras = len(x_train)
    if n_muestras < 2:
        raise ValueError("Se requieren al menos 2 muestras para ajustar ElasticNetCV.")

    lista_alphas = settings.ALPHAS_ELASTICNET if alphas is None else alphas
    ratios_l1 = settings.L1_RATIOS_ELASTICNET if l1_ratios is None else l1_ratios
    pliegues = settings.PLIEGUES_CV if pliegues_cv is None else pliegues_cv
    semilla_efectiva = settings.SEMILLA_ALEATORIA if semilla is None else semilla
    cv_efectivo = max(2, min(pliegues, n_muestras))

    # Mayor tope de iteraciones para converger ante colinealidad espectral y MFCC
    modelo = ElasticNetCV(
        alphas=lista_alphas,
        l1_ratio=ratios_l1,
        cv=cv_efectivo,
        max_iter=5000,
        random_state=semilla_efectiva,
    )
    modelo.fit(x_train, y_train)
    return modelo


def entrenar_baselines(x_train, y_train) -> dict[str, object]:
    return {
        "ridge": entrenar_ridge(x_train, y_train),
        "elasticnet": entrenar_elasticnet(x_train, y_train),
    }
