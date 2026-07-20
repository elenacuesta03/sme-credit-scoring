"""Funciones reutilizables para el modelo de originacion (Fase 2).

Prepara la matriz de features respetando las restricciones descubiertas en
la Fase 1 (rango temporal fiable, exclusion de leakage/identificadores,
tratamiento de variables de alta cardinalidad) y ofrece utilidades comunes
de entrenamiento y evaluacion para el scorecard (WoE + regresion logistica)
y el challenger (XGBoost).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.data_cleaning import ID_COLS, LEAKAGE_COLS

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Rango temporal y split (decisiones de Fase 2, ver notebooks/02_modelo.ipynb)
# ---------------------------------------------------------------------------

#: Rango de approval_year considerado fiable segun la Fase 1: fuera de este
#: rango, 1966-1988 tiene sesgo de muestra pequena y 2012-2014 sufre censura
#: por la derecha (prestamos que aun no han tenido tiempo de hacer default).
RELIABLE_YEAR_MIN = 1989
RELIABLE_YEAR_MAX = 2011

#: Split temporal train/test: train 1989-2007, test 2008-2011 (incluye la
#: cola de la crisis de 2008 para evaluar robustez fuera de distribucion).
TRAIN_YEAR_MAX = 2007

# ---------------------------------------------------------------------------
# Definicion de features
# ---------------------------------------------------------------------------

#: Variables numericas disponibles en el momento de la originacion.
NUMERIC_FEATURES = [
    "Term",
    "NoEmp",
    "CreateJob",
    "RetainedJob",
    "GrAppv",
    "SBA_Appv",
    "sba_guarantee_pct",
    "is_franchise",
    "flag_term_zero",
    "flag_noemp_zero",
]

#: Variables categoricas disponibles en el momento de la originacion.
#: `NAICS`, `FranchiseCode` y `NewExist` crudos se excluyen porque la
#: Fase 1 ya genero sus versiones limpias (`sector`, `is_franchise`,
#: `new_exist_label`); usar ambas duplicaria la misma senal. `City` y
#: `Zip` se excluyen por alta cardinalidad (32.5k y 33.6k valores unicos)
#: sin una tecnica de encoding robusta a overfitting; `State`/`BankState`
#: ya capturan la senal geografica relevante. `Bank` (5.8k valores
#: unicos) se conserva pero agrupada en las top-N entidades + "Other" via
#: :func:`group_rare_categories`.
#:
#: `urban_rural_label` se excluye deliberadamente pese a ser una columna
#: limpia de la Fase 1: el notebook 02_modelo.ipynb demuestra que su IV
#: inusualmente alto (0.56) no refleja riesgo geografico real, sino que
#: `UrbanRural == 0` ("Unknown") es casi un proxy perfecto de "prestamo
#: anterior a 1999" (la SBA no capturaba este campo de forma sistematica
#: antes de esa fecha), variable que coincide con el periodo de menor
#: morosidad del dataset. Mantenerla dejaria que el modelo aprenda un
#: atajo de calendario en vez de senal de riesgo generalizable.
CATEGORICAL_FEATURES = [
    "State",
    "BankState",
    "Bank_grouped",
    "sector",
    "new_exist_label",
    "RevLineCr",
    "LowDoc",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def group_rare_categories(
    series: pd.Series, top_n: int = 30, other_label: str = "Other"
) -> pd.Series:
    """Agrupa una variable categorica de alta cardinalidad.

    Mantiene las `top_n` categorias mas frecuentes tal cual y colapsa el
    resto (incluidos los nulos, tratados antes como su propia categoria
    "Missing") en `other_label`. Pensada para `Bank` (5.8k bancos unicos),
    donde un one-hot o target-encoding directo sobreajustaria.
    """
    filled = series.fillna("Missing")
    top_categories = filled.value_counts().nlargest(top_n).index
    return filled.where(filled.isin(top_categories), other_label)


def build_model_frame(df: pd.DataFrame, top_n_banks: int = 30) -> pd.DataFrame:
    """Construye el DataFrame de features + target listo para modelar.

    Aplica la agrupacion de `Bank` de alta cardinalidad y rellena nulos en
    las categoricas geograficas, y selecciona unicamente
    :data:`ALL_FEATURES` (que ya excluye, por construccion, todo lo que
    esta en `LEAKAGE_COLS`, `ID_COLS`, las columnas de alta cardinalidad
    sin tratar y las columnas de vintage/crudas superadas por sus
    versiones derivadas de la Fase 1).
    """
    out = df.copy()
    out["Bank_grouped"] = group_rare_categories(out["Bank"], top_n=top_n_banks)
    out["State"] = out["State"].fillna("Missing")
    out["BankState"] = out["BankState"].fillna("Missing")

    keep_cols = ALL_FEATURES + ["target", "approval_year"]
    return out[keep_cols].copy()


def assert_no_leakage(columns) -> None:
    """Verifica que ninguna columna de `LEAKAGE_COLS`/`ID_COLS` este en `columns`."""
    columns = set(columns)
    leaked = columns & set(LEAKAGE_COLS)
    ids = columns & set(ID_COLS)
    assert not leaked, f"Leakage detectado en las features: {leaked}"
    assert not ids, f"Identificadores detectados en las features: {ids}"


def temporal_train_test_split(
    df: pd.DataFrame,
    year_min: int = RELIABLE_YEAR_MIN,
    year_max: int = RELIABLE_YEAR_MAX,
    train_year_max: int = TRAIN_YEAR_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Particiona en train/test por `approval_year` (split temporal, no aleatorio).

    Restringe primero al rango fiable `[year_min, year_max]` y despues
    corta en `train_year_max`: train = `[year_min, train_year_max]`,
    test = `(train_year_max, year_max]`.
    """
    in_range = df[(df["approval_year"] >= year_min) & (df["approval_year"] <= year_max)]
    train = in_range[in_range["approval_year"] <= train_year_max].drop(columns="approval_year")
    test = in_range[in_range["approval_year"] > train_year_max].drop(columns="approval_year")
    return train, test


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------


def ks_statistic(y_true, y_proba) -> float:
    """Estadistico de Kolmogorov-Smirnov (separacion maxima entre TPR y FPR)."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_proba) -> dict:
    """AUC, Gini (2*AUC - 1) y KS para un conjunto de predicciones."""
    auc = roc_auc_score(y_true, y_proba)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_proba),
    }


def score_at_threshold(
    y_true,
    y_proba,
    threshold: float,
    cost_fn: Optional[float] = None,
    cost_fp: Optional[float] = None,
) -> dict:
    """Matriz de confusion y metricas de negocio para un punto de corte dado.

    Convencion: `y_proba` es la probabilidad de impago (clase positiva =
    CHGOFF). Un prestamo se **rechaza** si `y_proba >= threshold`.

    - ``tp``: rechazado y realmente malo (acierto).
    - ``fp``: rechazado pero realmente bueno (coste de oportunidad: margen
      no percibido de un buen prestamo).
    - ``fn``: aprobado pero realmente malo (coste de credito: perdida por
      impago).
    - ``tn``: aprobado y realmente bueno (acierto).

    Si se proporcionan `cost_fn` y `cost_fp` (en unidades relativas, p.ej.
    `cost_fp=1` y `cost_fn=8` para un ratio de coste 8:1), se anade el
    coste esperado total y por prestamo.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)

    result = {
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "approval_rate": (tn + fn) / n,
        "bad_rate_in_approved": fn / (tn + fn) if (tn + fn) else np.nan,
        "recall_bad": tp / (tp + fn) if (tp + fn) else np.nan,
        "precision_bad": tp / (tp + fp) if (tp + fp) else np.nan,
        "false_reject_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "accuracy": (tp + tn) / n,
    }

    if cost_fn is not None and cost_fp is not None:
        result["expected_cost"] = fn * cost_fn + fp * cost_fp
        result["expected_cost_per_loan"] = result["expected_cost"] / n

    return result


# ---------------------------------------------------------------------------
# Modelo 1: scorecard (WoE + regresion logistica + puntos)
# ---------------------------------------------------------------------------


def build_scorecard(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    pdo: float = 20,
    odds: float = 50,
    scorecard_points: float = 600,
    random_state: int = RANDOM_STATE,
):
    """Entrena el binning optimo (WoE/IV) y el scorecard de puntos.

    Devuelve `(scorecard, binning_process, iv_table)`. `iv_table` es el
    resumen de `BinningProcess` con el Information Value de cada
    variable, para poder filtrar por IV antes o despues de ajustar (ver
    notebook: variables con IV < 0.02 se consideran no predictivas, IV >
    0.5 se investigan por posible leakage residual).
    """
    from optbinning import BinningProcess
    from optbinning.scorecard import Scorecard
    from sklearn.linear_model import LogisticRegression

    variable_names = numeric_cols + categorical_cols
    binning_process = BinningProcess(
        variable_names=variable_names,
        categorical_variables=categorical_cols,
    )

    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=LogisticRegression(random_state=random_state),
        scaling_method="pdo_odds",
        scaling_method_params={
            "pdo": pdo,
            "odds": odds,
            "scorecard_points": scorecard_points,
        },
    )
    scorecard.fit(X_train[variable_names], y_train)

    iv_table = scorecard.binning_process_.summary().set_index("name")
    return scorecard, scorecard.binning_process_, iv_table


# ---------------------------------------------------------------------------
# Modelo 2: challenger XGBoost
# ---------------------------------------------------------------------------


def get_categories(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list]:
    """Extrae las categorias observadas en `df` (tipicamente train) para
    cada columna categorica, de forma que puedan aplicarse de forma
    identica a otros conjuntos (test) via :func:`to_xgb_frame`."""
    return {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols}


def to_xgb_frame(
    df: pd.DataFrame,
    categorical_cols: list[str],
    categories: Optional[dict[str, list]] = None,
) -> pd.DataFrame:
    """Convierte las columnas categoricas a dtype `category` para que
    XGBoost (`enable_categorical=True`) las trate de forma nativa, sin
    necesidad de one-hot encoding manual.

    Si se pasa `categories` (ver :func:`get_categories`), se fuerza ese
    mismo catalogo de categorias -- tipicamente el observado en train --
    en vez de inferirlo de `df`. Esto evita que XGBoost falle al predecir
    sobre test si aparece una categoria (p.ej. un `BankState` puntual) no
    vista en entrenamiento: esos valores se convierten en `NaN`, que
    XGBoost trata de forma nativa como dato faltante. Es ademas el
    comportamiento correcto para produccion: un modelo entrenado hasta
    hoy nunca ha podido ver categorias que solo existiran en el futuro.
    """
    out = df.copy()
    for col in categorical_cols:
        if categories is not None:
            out[col] = pd.Categorical(out[col], categories=categories[col])
        else:
            out[col] = out[col].astype("category")
    return out


def train_xgb_challenger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: Optional[dict] = None,
    n_iter: int = 15,
    n_splits: int = 3,
    random_state: int = RANDOM_STATE,
):
    """Ajusta un XGBoost con busqueda de hiperparametros ligera.

    Usa `TimeSeriesSplit` (los datos deben venir ordenados por
    `approval_year` antes de llamar a esta funcion) como esquema de
    validacion cruzada interna, en vez de k-fold aleatorio, para no
    filtrar informacion futura dentro del propio train.
    """
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    if param_distributions is None:
        param_distributions = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.02, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 5, 10],
            "reg_lambda": [1, 5, 10],
        }

    base_model = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,  # la paralelizacion la aporta RandomizedSearchCV (n_jobs=-1) sobre los folds/candidatos
    )

    cv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search
