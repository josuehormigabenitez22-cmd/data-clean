"""jhb_dataclean: clean data functions with pandas"""

from importlib.metadata import PackageNotFoundError, version

from jhb_dataclean.columns import standardize_columns
from jhb_dataclean.conversion import convert_types
from jhb_dataclean.duplicates import remove_duplicates
from jhb_dataclean.io import read_data, write_data
from jhb_dataclean.missing import handle_missing
from jhb_dataclean.outliers import detect_outliers
from jhb_dataclean.text import normalize_text

try:
    # La versión solo se escribe en pyproject.toml: la leemos del paquete instalado
    __version__ = version("jhb-dataclean")
except PackageNotFoundError:
    # El código se está usando sin instalar (por ejemplo con PYTHONPATH=src)
    __version__ = "0.0.0+unknown"

# API pública, en el orden típico de una limpieza: leer, limpiar y guardar.
# Es también lo que importa "from jhb_dataclean import *"
__all__ = [
    "read_data",
    "standardize_columns",
    "normalize_text",
    "convert_types",
    "handle_missing",
    "remove_duplicates",
    "detect_outliers",
    "write_data",
    "__version__",
]
