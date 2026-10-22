"""Remove duplicated rows."""

from collections.abc import Sequence
from typing import Literal

import pandas as pd

Keep = Literal["first", "last", False]


def remove_duplicates(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    keep: Keep = "first",
    order_by: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Remove duplicated rows, optionally choosing which copy to keep.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. It is not modified.
    columns : Sequence[str], optional
        Columns that identify a duplicate, e.g. an order id. By default
        all columns are used, so only exact duplicates are removed.
    keep : {"first", "last", False}, default "first"
        Which copy to keep: the first one, the last one, or none of them.
    order_by : str or Sequence[str], optional
        Sort by these columns (ascending) before choosing the copy, so
        ``keep="last"`` keeps the most recent row. Rows with a missing
        ``order_by`` value are never preferred. The result keeps the
        original row order and index.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` without the duplicated rows.

    Raises
    ------
    KeyError
        If a column in ``columns`` or ``order_by`` is not in ``df``.
    ValueError
        If ``columns`` is empty or ``keep`` is not ``"first"``, ``"last"``
        or ``False``.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "order_id": [1, 2, 1],
    ...     "amount": [100, 250, 120],
    ...     "date": ["2024-01-20", "2024-01-16", "2024-01-15"],
    ... })
    >>> remove_duplicates(df, ["order_id"])["amount"].tolist()
    [100, 250]
    >>> remove_duplicates(df, ["order_id"], keep="last")["amount"].tolist()
    [250, 120]
    >>> latest = remove_duplicates(df, ["order_id"], keep="last", order_by="date")
    >>> latest["amount"].tolist()
    [100, 250]
    """
    # "is not False" y no "!= False": 0 == False en Python y colaría como válido
    if keep is not False and keep not in ("first", "last"):
        raise ValueError(f"keep must be 'first', 'last' or False, got {keep!r}")
    if columns is not None and len(columns) == 0:
        raise ValueError("columns cannot be empty; use None to compare all columns")
    if isinstance(order_by, str):
        order_by = [order_by]

    missing = [
        col for col in [*(columns or []), *(order_by or [])] if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    df = df.copy()  # Nunca modifiques el DataFrame que te pasan

    # Trabajamos con posiciones (0, 1, 2...) y no con el índice, porque el índice
    # puede tener etiquetas repetidas y entonces no identificaría cada fila
    positions = pd.RangeIndex(len(df))
    if order_by:
        # Los nulos van al extremo que nunca se elige: al final si nos quedamos
        # con el primero y al principio si nos quedamos con el último
        na_position = "first" if keep == "last" else "last"
        positions = (
            df[order_by]
            .reset_index(drop=True)
            .sort_values(order_by, kind="stable", na_position=na_position)
            .index
        )

    is_duplicate = df.iloc[positions].duplicated(subset=columns, keep=keep)
    # sorted() devuelve las filas que se quedan a su orden original
    kept = sorted(positions[~is_duplicate.to_numpy()])
    return df.iloc[kept]
