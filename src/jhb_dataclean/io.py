"""Read and write DataFrames choosing the format from the file extension."""

import importlib.util
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

# Extensión -> (nombre del formato en pandas, motor por defecto, extra que lo instala)
_FORMATS = {
    ".csv": ("csv", None, None),
    ".xlsx": ("excel", "openpyxl", "excel"),
    ".parquet": ("parquet", "pyarrow", "parquet"),
}


def _resolve_format(path: Path, kwargs: dict[str, Any]) -> str:
    """Return the pandas format name for ``path`` and set its default engine."""
    suffix = path.suffix.lower()
    if suffix not in _FORMATS:
        raise ValueError(
            f"Unsupported file extension {suffix!r}; use one of {list(_FORMATS)}"
        )
    fmt, engine, extra = _FORMATS[suffix]

    if engine is not None and "engine" not in kwargs:
        # Comprobamos el motor antes de llamar a pandas para dar un mensaje que
        # diga qué instalar, en vez del ImportError genérico de pandas
        if importlib.util.find_spec(engine) is None:
            raise ImportError(
                f'{suffix} files need {engine}: pip install "jhb-dataclean[{extra}]"'
            )
        kwargs["engine"] = engine
    return fmt


def read_data(path: str | PathLike[str], **kwargs: Any) -> pd.DataFrame:
    """Read a CSV, Excel (.xlsx) or Parquet file into a DataFrame.

    Parameters
    ----------
    path : str or PathLike
        File to read. The format is chosen from its extension.
    **kwargs
        Passed to ``pd.read_csv``, ``pd.read_excel`` or ``pd.read_parquet``,
        e.g. ``sep=";"`` or ``sheet_name="Sales"``.

    Returns
    -------
    pd.DataFrame
        The file contents. Each format keeps types differently: Excel and
        Parquet may return dates and numbers already typed, and Excel drops
        leading zeros, so pass the result through ``convert_types``.

    Raises
    ------
    ValueError
        If the extension is not ``.csv``, ``.xlsx`` or ``.parquet``.
    ImportError
        If the optional dependency for the format is not installed.
    FileNotFoundError
        If ``path`` does not exist.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "sales.csv"
    ...     _ = path.write_text("zip_code,amount\\n08001,10.5\\n", encoding="utf-8")
    ...     read_data(path, dtype={"zip_code": str}).to_dict("list")
    {'zip_code': ['08001'], 'amount': [10.5]}
    """
    path = Path(path)
    fmt = _resolve_format(path, kwargs)
    if not path.exists():
        # pandas también lo detecta, pero con un mensaje distinto según el formato
        raise FileNotFoundError(f"File not found: {path}")
    return getattr(pd, f"read_{fmt}")(path, **kwargs)


def write_data(df: pd.DataFrame, path: str | PathLike[str], **kwargs: Any) -> None:
    """Write a DataFrame to a CSV, Excel (.xlsx) or Parquet file.

    The index is not written unless ``index=True`` is passed, so that
    ``read_data`` returns the same columns that were written.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write. It is not modified.
    path : str or PathLike
        Destination file. The format is chosen from its extension and an
        existing file is overwritten.
    **kwargs
        Passed to ``df.to_csv``, ``df.to_excel`` or ``df.to_parquet``.

    Raises
    ------
    ValueError
        If the extension is not ``.csv``, ``.xlsx`` or ``.parquet``.
    ImportError
        If the optional dependency for the format is not installed.

    Examples
    --------
    >>> import tempfile
    >>> df = pd.DataFrame({"city": ["Madrid", "Málaga"], "sales": [3, 5]})
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     write_data(df, Path(folder) / "sales.csv")
    ...     read_data(Path(folder) / "sales.csv").equals(df)
    True
    """
    path = Path(path)
    fmt = _resolve_format(path, kwargs)
    # Sin índice por defecto: si no, al leer aparece una columna "Unnamed: 0"
    kwargs.setdefault("index", False)
    getattr(df, f"to_{fmt}")(path, **kwargs)
