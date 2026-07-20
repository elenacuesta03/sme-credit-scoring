"""Funciones reutilizables para el seguimiento (monitorizacion) de los
modelos de originacion entrenados en la Fase 2 (`src/modeling.py`).

Este modulo NO reentrena nada: asume que ya existen probabilidades/score
calculados por un modelo ya ajustado, y se limita a cuantificar su
estabilidad poblacional (PSI), su estabilidad discriminante (Gini por
cohorte) y su calibracion a lo largo del tiempo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#: Umbrales estandar de la industria para el Population Stability Index.
PSI_STABLE_MAX = 0.10
PSI_MODERATE_MAX = 0.25


def psi_alert_level(psi_value: float) -> str:
    """Clasifica un valor de PSI en el semaforo estandar de la industria:
    "Verde" (< 0.10, estable), "Ambar" (0.10-0.25, alerta moderada) o
    "Rojo" (> 0.25, alerta alta / cambio significativo de poblacion)."""
    if pd.isna(psi_value):
        return "Sin datos"
    if psi_value < PSI_STABLE_MAX:
        return "Verde"
    if psi_value < PSI_MODERATE_MAX:
        return "Ambar"
    return "Rojo"


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre una poblacion de referencia
    (`expected`, tipicamente el score o la variable en la poblacion de
    desarrollo/train) y una poblacion nueva (`actual`, p.ej. una cohorte
    de seguimiento posterior).

    PSI = sum( (pct_actual - pct_expected) * ln(pct_actual / pct_expected) )

    Si `expected` es numerica, se discretiza en `bins` tramos por
    cuantiles calculados sobre `expected`, y esos mismos cortes se
    aplican a `actual` (asi el PSI mide realmente el desplazamiento de
    `actual` respecto a la referencia, no una discretizacion distinta
    para cada una). Si `expected` es categorica/texto, cada valor unico
    se trata como su propio bin.

    Interpretacion estandar (ver :func:`psi_alert_level`): PSI < 0.10
    estable, 0.10-0.25 alerta moderada, > 0.25 alerta alta.
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if pd.api.types.is_numeric_dtype(expected):
        quantiles = np.linspace(0, 1, bins + 1)
        cut_points = np.unique(expected.quantile(quantiles).to_numpy())
        # Los extremos se abren a +-inf para que valores de `actual` fuera
        # del rango observado en `expected` (p.ej. una cohorte nueva con
        # scores mas altos o mas bajos que cualquiera visto en train) caigan
        # en el bin extremo en vez de quedar fuera de todos los bins.
        cut_points[0] = -np.inf
        cut_points[-1] = np.inf
        expected_binned = pd.cut(expected, bins=cut_points, duplicates="drop")
        actual_binned = pd.cut(actual, bins=cut_points, duplicates="drop")
    else:
        expected_binned = expected.astype(str)
        actual_binned = actual.astype(str)

    expected_pct = expected_binned.value_counts(normalize=True, dropna=False)
    actual_pct = actual_binned.value_counts(normalize=True, dropna=False)

    all_bins = expected_pct.index.union(actual_pct.index)
    # Un bin vacio en una de las dos poblaciones produce log(0) o division
    # por 0; se suaviza con un epsilon pequeno, practica habitual al
    # calcular PSI.
    eps = 1e-4
    expected_pct = expected_pct.reindex(all_bins, fill_value=0) + eps
    actual_pct = actual_pct.reindex(all_bins, fill_value=0) + eps

    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(psi_components.sum())


def psi_by_cohort(
    df: pd.DataFrame,
    reference: pd.Series,
    col: str,
    cohort_col: str = "approval_year",
    bins: int = 10,
) -> pd.DataFrame:
    """Aplica :func:`calculate_psi` cohorte a cohorte para la columna `col`
    de `df`, comparando siempre contra la misma poblacion de referencia
    fija `reference` (no contra la cohorte anterior: eso permitiria que un
    desplazamiento lento y acumulativo pasara desapercibido cohorte a
    cohorte aunque el desplazamiento total frente al modelo de desarrollo
    fuera grande).
    """
    rows = []
    for cohort, g in df.groupby(cohort_col):
        psi = calculate_psi(reference, g[col], bins=bins)
        rows.append({cohort_col: cohort, "n": len(g), "psi": psi, "alerta": psi_alert_level(psi)})
    return pd.DataFrame(rows).set_index(cohort_col)


def gini_by_cohort(
    df: pd.DataFrame,
    y_proba_col: str,
    cohort_col: str = "approval_year",
    target_col: str = "target",
) -> pd.DataFrame:
    """AUC, Gini (2*AUC-1) y tasa de impago observada por cohorte.

    Recibe el nombre de una columna de probabilidad ya calculada
    (`y_proba_col`) en vez del objeto modelo: el scorecard y el
    challenger XGBoost requieren preprocesados de features distintos
    (WoE vs. dtype categorico nativo), asi que en este proyecto se
    puntua el DataFrame una vez por modelo (ver notebook) y esta funcion
    se limita a agregar esos scores ya calculados por cohorte -- un
    patron habitual en pipelines de seguimiento (puntuar una vez,
    analizar muchas veces).

    Cohortes con una unica clase en `target_col` (Gini indefinido, tipico
    en cohortes muy pequenas o completamente sanas) devuelven `NaN` en
    `auc`/`gini` en vez de lanzar un error.
    """
    rows = []
    for cohort, g in df.groupby(cohort_col):
        y = g[target_col]
        n = len(g)
        bad_rate = y.mean()
        if y.nunique() < 2:
            rows.append({cohort_col: cohort, "n": n, "bad_rate": bad_rate, "auc": np.nan, "gini": np.nan})
            continue
        auc = roc_auc_score(y, g[y_proba_col])
        rows.append({cohort_col: cohort, "n": n, "bad_rate": bad_rate, "auc": auc, "gini": 2 * auc - 1})
    return pd.DataFrame(rows).set_index(cohort_col)


def calibration_by_cohort(
    df: pd.DataFrame,
    y_proba_col: str,
    cohort_col: str = "approval_year",
    target_col: str = "target",
) -> pd.DataFrame:
    """Tasa de impago media predicha vs. observada por cohorte.

    `predicted_rate` es la probabilidad media que el modelo asigna a la
    cohorte; `observed_rate` es la tasa de impago real. `gap` > 0
    significa que el modelo esta sobreestimando el riesgo de la cohorte
    (prudente); `gap` < 0 que lo esta infraestimando (mas preocupante
    desde el punto de vista de gestion de riesgo).
    """
    g = df.groupby(cohort_col).agg(
        n=(target_col, "size"),
        observed_rate=(target_col, "mean"),
        predicted_rate=(y_proba_col, "mean"),
    )
    g["gap"] = g["predicted_rate"] - g["observed_rate"]
    return g


def traffic_light(psi_value: float, gini_drop_pp: float) -> str:
    """Semaforo combinado de seguimiento a partir del PSI del score y de
    la caida de Gini en puntos porcentuales respecto al Gini de
    referencia (train). Regla: el nivel mas alto de alerta entre PSI y
    caida de Gini determina el color final (criterio conservador: basta
    con que una de las dos senales este mal para no dar luz verde).

    - Verde: PSI < 0.10 y caida de Gini < 5 p.p.
    - Ambar: PSI en 0.10-0.25, o caida de Gini entre 5 y 15 p.p.
    - Rojo: PSI > 0.25, o caida de Gini > 15 p.p.
    """
    psi_level = psi_alert_level(psi_value)

    if pd.isna(gini_drop_pp):
        gini_level = "Sin datos"
    elif gini_drop_pp < 5:
        gini_level = "Verde"
    elif gini_drop_pp < 15:
        gini_level = "Ambar"
    else:
        gini_level = "Rojo"

    order = {"Verde": 0, "Ambar": 1, "Rojo": 2, "Sin datos": 0}
    worst = max([psi_level, gini_level], key=lambda lvl: order[lvl])
    return worst
