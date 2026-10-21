"""Standardize DataFrame column names."""

import re

import pandas as pd

from jhb_dataclean._utils import strip_accents


def _to_snake_case(name: str) -> str:
    """Convert a single column name to snake_case ASCII."""
    name = strip_accents(name).strip()
    # Separa camelCase: "precioUnitario" -> "precio_Unitario"
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    # Cualquier grupo de caracteres no alfanuméricos se convierte en "_"
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    return name.strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case.

    Strips surrounding whitespace, removes accents, splits camelCase,
    lowercases and replaces any run of non-alphanumeric characters
    with a single underscore.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. It is not modified.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with standardized column names.

    Raises
    ------
    ValueError
        If two or more columns end up with the same name.

    Examples
    --------
    >>> df = pd.DataFrame(columns=[" Fecha Pedido ", "Precio (€)", "Categoría"])
    >>> standardize_columns(df).columns.tolist()
    ['fecha_pedido', 'precio', 'categoria']
    """
    df = df.copy()  # Nunca modifiques el DataFrame que te pasan
    new_columns = [_to_snake_case(str(col)) for col in df.columns]

    duplicated = pd.Index(new_columns)[pd.Index(new_columns).duplicated()]
    if not duplicated.empty:
        raise ValueError(
            f"Standardizing produced duplicated column names: {sorted(set(duplicated))}"
        )

    df.columns = new_columns
    return df
