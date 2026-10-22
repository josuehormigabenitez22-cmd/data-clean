"""Convert columns to their proper data types."""

from collections.abc import Collection, Mapping
from typing import Any, Literal

import pandas as pd

from jhb_dataclean._utils import strip_accents

TargetType = Literal["datetime", "numeric", "boolean", "string"]
_TARGET_TYPES = ("datetime", "numeric", "boolean", "string")


def _normalize_token(value: object) -> str:
    """Lowercase, strip and remove accents so ' Sí ' matches 'si'."""
    return strip_accents(str(value)).strip().lower()


def _to_datetime(series: pd.Series, *, dayfirst: bool) -> pd.Series:
    # Excel y Parquet ya traen las fechas tipadas: no hay nada que convertir
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # pandas 3 aplica dayfirst incluso a "2024-04-10" (lo lee como 4 de octubre),
    # así que primero leemos las fechas ISO (año-mes-día) sin ambigüedad...
    iso = pd.to_datetime(series, format="ISO8601", errors="coerce")
    # ...y solo el resto ("15/03/2024") se interpreta con dayfirst
    rest = pd.to_datetime(
        series.where(iso.isna()), format="mixed", dayfirst=dayfirst, errors="coerce"
    )
    return iso.fillna(rest)


def _to_numeric(series: pd.Series) -> pd.Series:
    is_bool = pd.api.types.is_bool_dtype(series)
    if pd.api.types.is_numeric_dtype(series) and not is_bool:
        return series
    # Los espacios alrededor ("  2300 ") harían que el número se pierda como NaN
    values = series.astype(object).map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    return pd.to_numeric(values, errors="coerce")


def _to_boolean(
    series: pd.Series,
    *,
    true_values: Collection[str],
    false_values: Collection[str],
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    lookup = {_normalize_token(v): True for v in true_values}
    lookup |= {_normalize_token(v): False for v in false_values}

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        # Excel puede traer 1/0 o 1.0/0.0 en lugar de texto
        if isinstance(value, int | float) and value in (0, 1):
            return bool(value)
        # Lo que no se reconoce queda como nulo, igual que errors="coerce"
        return lookup.get(_normalize_token(value), pd.NA)

    # "boolean" (con B minúscula) es el booleano de pandas que admite nulos
    return series.astype(object).map(convert).astype("boolean")


def _to_string(series: pd.Series, *, width: int | None) -> pd.Series:
    def convert(value: Any) -> Any:
        if pd.isna(value):
            return value
        # Un entero con nulos llega como float desde Excel: 8001.0 -> "8001"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    # infer_objects deja el tipo de texto propio de cada versión de pandas
    text = series.astype(object).map(convert).infer_objects()
    if width is not None:
        # zfill recupera los ceros a la izquierda que Excel o el CSV perdieron
        text = text.str.zfill(width)
    return text


def convert_types(
    df: pd.DataFrame,
    types: Mapping[str, TargetType],
    *,
    dayfirst: bool = True,
    true_values: Collection[str] = ("si", "yes", "true", "1"),
    false_values: Collection[str] = ("no", "false", "0"),
    zfill: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Convert columns to datetime, numeric, boolean or string types.

    Values that cannot be converted become missing (``NaT``, ``NaN`` or
    ``<NA>``) instead of raising. Columns that already have the target
    type are left unchanged, so data coming from Excel or Parquet works
    the same as data coming from CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. It is not modified.
    types : Mapping[str, {"datetime", "numeric", "boolean", "string"}]
        Target type for each column to convert. Other columns are untouched.
    dayfirst : bool, default True
        Interpret ambiguous dates such as ``"03/04/2024"`` as day/month.
    true_values, false_values : Collection[str]
        Text values mapped to ``True`` / ``False`` in ``"boolean"`` columns.
        Matching ignores case, surrounding spaces and accents.
    zfill : Mapping[str, int], optional
        Left-pad ``"string"`` columns with zeros up to the given width,
        e.g. ``{"zip_code": 5}`` turns ``8001`` into ``"08001"``.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the requested columns converted.

    Raises
    ------
    KeyError
        If a column in ``types`` is not in ``df``.
    ValueError
        If a target type is unknown, a value is in both ``true_values``
        and ``false_values``, or ``zfill`` refers to a column that is not
        converted to ``"string"`` or uses a width lower than 1.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "date": ["2024-01-15", "15/03/2024"],
    ...     "amount": ["1500.50", "N/A"],
    ...     "active": ["si", "NO"],
    ...     "zip_code": [28001, 8001],
    ... })
    >>> out = convert_types(
    ...     df,
    ...     {"date": "datetime", "amount": "numeric",
    ...      "active": "boolean", "zip_code": "string"},
    ...     zfill={"zip_code": 5},
    ... )
    >>> out["date"].dt.strftime("%Y-%m-%d").tolist()
    ['2024-01-15', '2024-03-15']
    >>> out["amount"].tolist()
    [1500.5, nan]
    >>> out["active"].tolist()
    [True, False]
    >>> out["zip_code"].tolist()
    ['28001', '08001']
    """
    missing = [col for col in types if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    invalid = {col: kind for col, kind in types.items() if kind not in _TARGET_TYPES}
    if invalid:
        raise ValueError(f"Unknown target types {invalid}; use one of {_TARGET_TYPES}")

    overlap = {_normalize_token(v) for v in true_values} & {
        _normalize_token(v) for v in false_values
    }
    if overlap:
        raise ValueError(f"Values in both true_values and false_values: {overlap}")

    zfill = dict(zfill or {})
    not_string = [col for col in zfill if types.get(col) != "string"]
    if not_string:
        raise ValueError(f"zfill columns must be converted to 'string': {not_string}")
    bad_width = {col: w for col, w in zfill.items() if not isinstance(w, int) or w < 1}
    if bad_width:
        raise ValueError(f"zfill widths must be integers >= 1: {bad_width}")

    df = df.copy()  # Nunca modifiques el DataFrame que te pasan

    for col, kind in types.items():
        if kind == "datetime":
            df[col] = _to_datetime(df[col], dayfirst=dayfirst)
        elif kind == "numeric":
            df[col] = _to_numeric(df[col])
        elif kind == "boolean":
            df[col] = _to_boolean(
                df[col], true_values=true_values, false_values=false_values
            )
        else:
            df[col] = _to_string(df[col], width=zfill.get(col))

    return df
