"""Handle missing values column by column."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

_STRATEGIES = ("drop", "mean", "median", "mode")


def _is_missing(value: Any) -> bool:
    """Return True for None, NaN, NaT or pd.NA (but not for lists)."""
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _supports_average(series: pd.Series) -> bool:
    """Mean and median only make sense for numbers and dates."""
    is_number = pd.api.types.is_numeric_dtype(series)
    is_bool = pd.api.types.is_bool_dtype(series)
    return (is_number and not is_bool) or pd.api.types.is_datetime64_any_dtype(series)


def _statistic(series: pd.Series, strategy: str) -> Any:
    # Sin ningún dato no hay estadístico (y NumPy avisaría de "empty slice")
    if series.isna().all():
        return None
    if strategy == "mean":
        return series.mean()
    if strategy == "median":
        return series.median()
    # mode() devuelve todos los empates ordenados: nos quedamos con el primero
    return series.mode().iloc[0]


def _fill(series: pd.Series, value: Any) -> pd.Series:
    if _is_missing(value):
        # Columna sin ningún dato: no hay nada con lo que rellenar
        return series
    if isinstance(series.dtype, pd.CategoricalDtype):
        # Una categoría solo admite valores que ya están entre sus categorías
        if value not in series.cat.categories:
            series = series.cat.add_categories([value])
    elif pd.api.types.is_integer_dtype(series) and not float(value).is_integer():
        # Una mediana de 4.5 no cabe en una columna de enteros
        series = series.astype("Float64")
    return series.fillna(value)


def handle_missing(
    df: pd.DataFrame,
    strategies: Mapping[str, Any],
    *,
    fill_values: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Drop or fill missing values with a different strategy per column.

    Rows are dropped first, so ``"mean"``, ``"median"`` and ``"mode"`` are
    computed only on the rows that are kept. The original index is kept.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. It is not modified.
    strategies : Mapping[str, Any]
        What to do with the missing values of each column:

        - ``"drop"``: drop the rows where this column is missing.
        - ``"mean"`` / ``"median"``: fill with the column mean or median
          (numeric and datetime columns only).
        - ``"mode"``: fill with the most frequent value (the first one
          if there is a tie).
        - any other value: fill with that value, e.g. ``"Unknown"`` or ``0``.

        Columns not listed are left untouched.
    fill_values : Mapping[str, Any], optional
        Fixed fill values that are always taken literally. Use it when the
        value itself is a strategy name, e.g. ``{"level": "median"}``.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the missing values handled.

    Raises
    ------
    KeyError
        If a column in ``strategies`` or ``fill_values`` is not in ``df``.
    TypeError
        If ``"mean"`` or ``"median"`` is used on a non-numeric column.
    ValueError
        If a column appears in both ``strategies`` and ``fill_values``,
        or a fill value is itself missing (``None``, ``NaN``...).

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "client": ["Ana", "Carlos", None, "Pedro"],
    ...     "purchases": [5.0, 12.0, 4.0, None],
    ...     "city": ["Madrid", None, "Madrid", "Sevilla"],
    ... })
    >>> out = handle_missing(
    ...     df, {"client": "drop", "purchases": "median", "city": "Unknown"}
    ... )
    >>> out["purchases"].tolist()
    [5.0, 12.0, 8.5]
    >>> out["city"].tolist()
    ['Madrid', 'Unknown', 'Sevilla']
    >>> out.index.tolist()
    [0, 1, 3]
    """
    fill_values = dict(fill_values or {})

    missing = [col for col in [*strategies, *fill_values] if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    both = [col for col in strategies if col in fill_values]
    if both:
        raise ValueError(f"Columns in both strategies and fill_values: {both}")

    # Separamos los nombres de estrategia de los valores fijos de relleno
    named = {
        col: value
        for col, value in strategies.items()
        if isinstance(value, str) and value in _STRATEGIES
    }
    fixed = {col: v for col, v in strategies.items() if col not in named}
    fixed |= fill_values

    empty_fill = [col for col, value in fixed.items() if _is_missing(value)]
    if empty_fill:
        raise ValueError(f"Fill values cannot be missing: {empty_fill}")
    no_average = [
        col
        for col, strategy in named.items()
        if strategy in ("mean", "median") and not _supports_average(df[col])
    ]
    if no_average:
        raise TypeError(f"mean/median need numeric or datetime columns: {no_average}")

    df = df.copy()  # Nunca modifiques el DataFrame que te pasan

    # Primero se borran filas, para que los estadísticos usen solo las que quedan
    drop = [col for col, strategy in named.items() if strategy == "drop"]
    if drop:
        df = df.dropna(subset=drop)

    for col, strategy in named.items():
        if strategy != "drop":
            df[col] = _fill(df[col], _statistic(df[col], strategy))
    for col, value in fixed.items():
        df[col] = _fill(df[col], value)

    return df
