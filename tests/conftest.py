"""Shared fixtures for the jhb_dataclean test suite."""

import pandas as pd
import pytest


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Small DataFrame with the typical problems of a raw export."""
    return pd.DataFrame(
        {
            " Fecha Pedido ": ["2024-01-15", "15/03/2024", "2024-04-10"],
            "precioUnitario": ["1500.50", "N/A", " 2300 "],
            "Ciudad": [" Madrid ", "MADRID", "   "],
            "Categoría": pd.Series(["a", "b", "a"], dtype="category"),
        }
    )


@pytest.fixture(params=[object, "string"], ids=["object", "string"])
def text_dtype(request: pytest.FixtureRequest) -> object:
    """Text dtypes to test: pandas 2 stores text as object by default."""
    return request.param
