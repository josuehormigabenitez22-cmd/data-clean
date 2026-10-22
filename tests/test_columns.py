"""Tests for jhb_dataclean.columns."""

import pandas as pd
import pytest

from jhb_dataclean import standardize_columns


def test_standardizes_messy_names(messy_df: pd.DataFrame) -> None:
    result = standardize_columns(messy_df)
    assert result.columns.tolist() == [
        "fecha_pedido",
        "precio_unitario",
        "ciudad",
        "categoria",
    ]


def test_does_not_modify_input(messy_df: pd.DataFrame) -> None:
    original = messy_df.copy()
    standardize_columns(messy_df)
    pd.testing.assert_frame_equal(messy_df, original)


def test_keeps_values_untouched(messy_df: pd.DataFrame) -> None:
    result = standardize_columns(messy_df)
    # Solo cambian los nombres: los datos deben ser exactamente los mismos
    assert result.to_numpy().tolist() == messy_df.to_numpy().tolist()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("precioUnitario", "precio_unitario"),
        ("Precio (€)", "precio"),
        ("  Total__Neto  ", "total_neto"),
        ("ÁREA", "area"),
        ("id2Cliente", "id2_cliente"),
        (0, "0"),
    ],
)
def test_single_name(name: object, expected: str) -> None:
    df = pd.DataFrame(columns=[name])
    assert standardize_columns(df).columns.tolist() == [expected]


def test_duplicated_result_raises() -> None:
    df = pd.DataFrame(columns=["Precio", "precio "])
    with pytest.raises(ValueError, match="duplicated column names"):
        standardize_columns(df)


def test_empty_dataframe() -> None:
    assert standardize_columns(pd.DataFrame()).columns.tolist() == []
