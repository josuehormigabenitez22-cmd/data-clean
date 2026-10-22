"""Flag outliers in numeric columns."""

from collections.abc import Sequence
from typing import Literal

import pandas as pd

Method = Literal["iqr", "zscore"]
_DEFAULT_THRESHOLDS = {"iqr": 1.5, "zscore": 3.0}


def _is_number(series: pd.Series) -> bool:
    """Numeric columns, excluding booleans (True/False are not measurements)"""
    is_bool = pd.api.types.is_bool_dtype(series)
    return pd.api.types.is_numeric_dtype(series) and not is_bool


def _outlier_mask(series: pd.Series, method: str, threshold: float) -> pd.Series:
    if method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        mask = (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
    else:
        # Si la desviación es 0 todos los valores son iguales: z = 0/0 = NaN
        z = (series - series.mean()) / series.std()
        mask = z.abs() > threshold
    # Con tipos nullable (Int64, Float64) la comparación con un nulo da <NA>:
    # un valor que falta no es un outlier
    return mask.fillna(False).astype(bool)


def detect_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: Method = "iqr",
    threshold: float | None = None,
    suffix: str = "_outlier",
) -> pd.DataFrame:
    """Flag outliers with one boolean column per analyzed column.

    Rows are never removed: for each column ``col`` a new column
    ``col + suffix`` is added, ``True`` where the value is an outlier.
    Missing values are never flagged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. It is not modified.
    columns : Sequence[str], optional
        Numeric columns to analyze. By default all numeric columns
        (booleans excluded).
    method : {"iqr", "zscore"}, default "iqr"
        - ``"iqr"``: outside ``[Q1 - threshold * IQR, Q3 + threshold * IQR]``.
        - ``"zscore"``: ``|value - mean| / std > threshold``. Extreme values
          inflate the mean and std, so with few rows it may flag nothing
          (with 7 rows no z-score can exceed 2.27).
    threshold : float, optional
        Defaults to 1.5 for ``"iqr"`` and 3 for ``"zscore"``.
    suffix : str, default "_outlier"
        Appended to each column name to build its flag column.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the flag columns added at the end.

    Raises
    ------
    KeyError
        If a column in ``columns`` is not in ``df``.
    TypeError
        If a column in ``columns`` is not numeric.
    ValueError
        If ``method`` is unknown, ``threshold`` is not positive, ``suffix``
        is empty or a flag column already exists in ``df``.

    Examples
    --------
    >>> df = pd.DataFrame({"price": [25, 30, 28, 5000, 22, 31, 27]})
    >>> detect_outliers(df)["price_outlier"].tolist()
    [False, False, False, True, False, False, False]
    >>> detect_outliers(df, method="zscore")["price_outlier"].tolist()
    [False, False, False, False, False, False, False]
    >>> detect_outliers(df, method="zscore", threshold=2)["price_outlier"].tolist()
    [False, False, False, True, False, False, False]
    """
    if method not in _DEFAULT_THRESHOLDS:
        raise ValueError(f"method must be 'iqr' or 'zscore', got {method!r}")
    if threshold is None:
        threshold = _DEFAULT_THRESHOLDS[method]
    elif threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold!r}")
    if not suffix:
        raise ValueError("suffix cannot be empty: it would overwrite the columns")

    if columns is None:
        columns = [col for col in df.columns if _is_number(df[col])]
    else:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        not_numeric = [col for col in columns if not _is_number(df[col])]
        if not_numeric:
            raise TypeError(f"Columns are not numeric: {not_numeric}")

    # No pisamos columnas existentes: podrían ser datos o marcas de otra llamada
    existing = [f"{col}{suffix}" for col in columns if f"{col}{suffix}" in df.columns]
    if existing:
        raise ValueError(f"Flag columns already exist: {existing}")

    df = df.copy()  # Nunca modifiques el DataFrame que te pasan

    for col in columns:
        df[f"{col}{suffix}"] = _outlier_mask(df[col], method, threshold)

    return df
