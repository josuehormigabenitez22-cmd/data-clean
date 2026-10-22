"""Tests for the public API exported by jhb_dataclean."""

import importlib
import importlib.metadata

import pytest

import jhb_dataclean


def test_public_api() -> None:
    assert jhb_dataclean.__all__ == [
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


@pytest.mark.parametrize("name", jhb_dataclean.__all__)
def test_every_name_is_exported(name: str) -> None:
    assert hasattr(jhb_dataclean, name)


def test_version_is_a_string() -> None:
    assert isinstance(jhb_dataclean.__version__, str)
    assert jhb_dataclean.__version__


def test_version_fallback_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_installed(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    # Volvemos a ejecutar __init__.py como si el paquete no estuviera instalado
    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    try:
        importlib.reload(jhb_dataclean)
        assert jhb_dataclean.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(jhb_dataclean)
