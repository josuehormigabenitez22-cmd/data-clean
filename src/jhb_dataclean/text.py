"""Normalize free-text columns"""

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

from jhb_dataclean._utils import strip_accents

Case = Literal["lower", "upper", "title"]


def _is_text(series: pd.Series) -> bool:
    """Return True if series holds strings"""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return False
    if pd.api.types.is_string_dtype(series):
        return True
    # pandas 2 guarda el texto como "object": hay que mirar los valores no nulos
    values = series.dropna()
    if series.dtype != object or values.empty:
        return False
    return values.map(lambda value: isinstance(value, str)).all()


def normalize_text(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    case: Case | None = "lower",
    remove_accents: bool = False,
    collapse_spaces: bool = True,
    empty_as_na: bool = True,
    replacements: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize text columns: whitespace, case, accents and known typos."""

    if case not in (None, "lower", "upper", "title"):
        raise ValueError(
            f"case must be 'lower', 'upper', 'title' or None, got {case!r}"
        )

    df = df.copy()  # Nunca modifiques el DataFrame que te pasan

    if columns is None:
        columns = [col for col in df.columns if _is_text(df[col])]
    else:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        not_text = [col for col in columns if not _is_text(df[col])]
        if not_text:
            raise TypeError(f"Columns do not contain text: {not_text}")

    for col in columns:
        text = df[col].str.strip()
        if collapse_spaces:
            text = text.str.replace(r"\s+", " ", regex=True)
        if case is not None:
            # Equivale a text.str.lower(), text.str.upper() o text.str.title()
            text = getattr(text.str, case)()
        if remove_accents:
            text = text.map(strip_accents, na_action="ignore")
        if empty_as_na:
            text = text.mask(text == "")
        if replacements:
            text = text.replace(dict(replacements))
        df[col] = text

    return df
