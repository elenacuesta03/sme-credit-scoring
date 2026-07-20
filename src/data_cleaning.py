"""Carga y limpieza del dataset SBA "Should This Loan Be Approved or Denied?".

Este modulo centraliza TODA la limpieza del dataset crudo para que sea
reutilizable en las fases de modelado (Fase 2) y seguimiento temporal
(Fase 3). La funcion principal es :func:`load_and_clean`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Columnas que son identificadores y no aportan senal predictiva.
ID_COLS: list[str] = ["LoanNr_ChkDgt", "Name"]

#: Columnas que se conocen DESPUES de la concesion del prestamo y que por
#: tanto no pueden usarse como features en un modelo de originacion (fuga de
#: informacion / leakage). Se conservan en el DataFrame limpio porque seran
#: necesarias para el analisis temporal de la Fase 3, pero deben excluirse
#: explicitamente de la matriz de features en la Fase 2.
LEAKAGE_COLS: list[str] = [
    "ChgOffDate",
    "ChgOffPrinGr",
    "BalanceGross",
    "MIS_Status",
    "DisbursementDate",
    "DisbursementGross",
]

#: Columnas monetarias almacenadas como texto (p.ej. "$60,000.00 ").
_MONEY_COLS: list[str] = [
    "DisbursementGross",
    "BalanceGross",
    "ChgOffPrinGr",
    "GrAppv",
    "SBA_Appv",
]

#: Columnas de fecha en formato "28-Feb-99" con ano a 2 digitos.
_DATE_COLS: list[str] = ["ApprovalDate", "DisbursementDate", "ChgOffDate"]

#: Ultimo ano cubierto por el dataset (2014). Se usa para pivotar los anos a
#: 2 digitos: valores 00-14 se interpretan como 2000-2014 y valores 15-99
#: como 1915-1999 (el propio dataset no contiene fechas en el rango
#: intermedio 1915-1961, por lo que el pivote es inequivoco).
_TWO_DIGIT_YEAR_PIVOT = 14

#: Tabla oficial NAICS de 2 digitos (sector economico).
NAICS_SECTOR_MAP: dict[int, str] = {
    11: "Agriculture, Forestry, Fishing and Hunting",
    21: "Mining, Quarrying, and Oil and Gas Extraction",
    22: "Utilities",
    23: "Construction",
    31: "Manufacturing",
    32: "Manufacturing",
    33: "Manufacturing",
    42: "Wholesale Trade",
    44: "Retail Trade",
    45: "Retail Trade",
    48: "Transportation and Warehousing",
    49: "Transportation and Warehousing",
    51: "Information",
    52: "Finance and Insurance",
    53: "Real Estate and Rental and Leasing",
    54: "Professional, Scientific, and Technical Services",
    55: "Management of Companies and Enterprises",
    56: "Administrative and Support and Waste Management Services",
    61: "Educational Services",
    62: "Health Care and Social Assistance",
    71: "Arts, Entertainment, and Recreation",
    72: "Accommodation and Food Services",
    81: "Other Services (except Public Administration)",
    92: "Public Administration",
}


# ---------------------------------------------------------------------------
# Helpers de parsing
# ---------------------------------------------------------------------------


def _parse_money(series: pd.Series) -> pd.Series:
    """Convierte una columna monetaria en texto (p.ej. "$60,000.00 ") a float."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace({"nan": np.nan, "": np.nan})
    )
    return cleaned.astype(float)


def _parse_approval_fy(series: pd.Series) -> pd.Series:
    """Limpia ApprovalFY (tipos mixtos, p.ej. "1976A") y lo convierte a Int64."""
    digits = series.astype(str).str.extract(r"(\d+)")[0]
    return digits.astype("Int64")


def _parse_sba_date(series: pd.Series) -> pd.Series:
    """Parsea fechas en formato "28-Feb-99" con pivote correcto de ano.

    ``pandas``/``strptime`` interpretan por defecto los anos de 2 digitos
    con el pivote 68/69 (00-68 -> 2000-2068), lo cual es incorrecto para
    este dataset: anos como "91" deben mapear a 1991 y no a 2091. Aqui se
    fuerza el pivote en :data:`_TWO_DIGIT_YEAR_PIVOT` (14), coherente con
    que el dataset cubre 1962-2014.
    """
    text = series.astype(str).str.strip()
    text = text.replace({"nan": np.nan, "": np.nan, "NaT": np.nan})

    extracted = text.str.extract(r"^(\d{1,2})-(\w{3})-(\d{2})$")
    day, month, year2 = extracted[0], extracted[1], extracted[2]

    year2_num = year2.astype(float)
    century = np.where(year2_num <= _TWO_DIGIT_YEAR_PIVOT, 2000, 1900)
    full_year = (century + year2_num).astype("Int64").astype(str)

    rebuilt = day + "-" + month + "-" + full_year
    parsed = pd.to_datetime(rebuilt, format="%d-%b-%Y", errors="coerce")
    return parsed


def _map_binary_flag(series: pd.Series) -> pd.Series:
    """Normaliza columnas Y/N (RevLineCr, LowDoc): todo lo que no sea 'Y'/'N' -> 'Unknown'."""
    cleaned = series.astype(str).str.strip().str.upper()
    return cleaned.where(cleaned.isin(["Y", "N"]), other="Unknown")


def _map_new_exist(series: pd.Series) -> pd.Series:
    mapping = {1: "Existing", 2: "New"}
    return series.map(mapping).fillna("Unknown")


def _map_urban_rural(series: pd.Series) -> pd.Series:
    mapping = {0: "Unknown", 1: "Urban", 2: "Rural"}
    return series.map(mapping).fillna("Unknown")


def _map_sector(naics: pd.Series) -> pd.Series:
    two_digit = (naics // 10000).astype(int)
    return two_digit.map(NAICS_SECTOR_MAP).fillna("Unknown")


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------


def load_and_clean(path: str) -> pd.DataFrame:
    """Carga el CSV crudo de SBA y aplica toda la limpieza y el feature engineering ligero.

    Pasos aplicados (ver notebook 01_eda.ipynb para la auditoria detallada
    con antes/después de cada uno):

    1. Elimina filas con ``MIS_Status`` nulo (no se puede determinar el target).
    2. Crea la variable objetivo binaria ``target`` (1 = CHGOFF, 0 = PIF).
    3. Parsea las columnas monetarias (texto -> float).
    4. Limpia ``ApprovalFY`` (sufijos no numericos) y lo convierte a entero.
    5. Normaliza ``RevLineCr`` y ``LowDoc`` a {"Y", "N", "Unknown"}.
    6. Parsea las columnas de fecha con el pivote de ano correcto y valida
       que ninguna fecha resultante sea posterior a 2014.
    7. Feature engineering ligero: ``sector`` (NAICS a 2 digitos),
       ``new_exist_label``, ``urban_rural_label``, ``is_franchise``,
       ``sba_guarantee_pct``, ``approval_year``.
    8. Anade flags de trazabilidad para valores sospechosos (``Term`` == 0,
       ``NoEmp`` == 0) sin eliminar filas, ya que pueden representar casos
       legitimos (prestamos muy cortos, autonomos sin empleados) que se
       investigan en el EDA en lugar de descartarse a ciegas.

    Todas las columnas de :data:`LEAKAGE_COLS` se conservan (limpias) en el
    DataFrame resultante para permitir el analisis temporal de la Fase 3;
    deben excluirse explicitamente de la matriz de features al modelar.

    Parameters
    ----------
    path:
        Ruta al CSV crudo (``data/raw/SBAnational.csv``).

    Returns
    -------
    pd.DataFrame
        Dataset limpio, con una fila por prestamo.
    """
    df = pd.read_csv(path, low_memory=False)

    # 1-2. Target -------------------------------------------------------
    df = df[df["MIS_Status"].notna()].copy()
    df["target"] = (df["MIS_Status"].str.strip() == "CHGOFF").astype(int)

    # 3. Columnas monetarias ---------------------------------------------
    for col in _MONEY_COLS:
        df[col] = _parse_money(df[col])

    # 4. ApprovalFY --------------------------------------------------------
    df["ApprovalFY"] = _parse_approval_fy(df["ApprovalFY"])

    # 5. RevLineCr / LowDoc -------------------------------------------------
    df["RevLineCr"] = _map_binary_flag(df["RevLineCr"])
    df["LowDoc"] = _map_binary_flag(df["LowDoc"])

    # 6. Fechas --------------------------------------------------------------
    for col in _DATE_COLS:
        df[col] = _parse_sba_date(df[col])

    max_valid_date = pd.Timestamp("2014-12-31")
    for col in _DATE_COLS:
        n_future = (df[col] > max_valid_date).sum()
        if n_future:
            raise ValueError(
                f"{col} contiene {n_future} fechas posteriores a 2014; "
                "revisar el pivote de ano de _parse_sba_date."
            )

    # 7. Feature engineering ligero -------------------------------------
    df["sector"] = _map_sector(df["NAICS"])
    df["new_exist_label"] = _map_new_exist(df["NewExist"])
    df["urban_rural_label"] = _map_urban_rural(df["UrbanRural"])
    df["is_franchise"] = (df["FranchiseCode"] > 1).astype(int)
    df["sba_guarantee_pct"] = df["SBA_Appv"] / df["GrAppv"]
    df["approval_year"] = df["ApprovalDate"].dt.year

    # 8. Flags de trazabilidad para valores sospechosos ---------------------
    df["flag_term_zero"] = (df["Term"] == 0).astype(int)
    df["flag_noemp_zero"] = (df["NoEmp"] == 0).astype(int)

    df = df.reset_index(drop=True)
    return df
