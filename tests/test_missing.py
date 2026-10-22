"""Tests for jhb_dataclean.missing."""

import numpy as np
import pandas as pd
import pytest

from jhb_dataclean import handle_missing


@pytest.fixture
def customers() -> pd.DataFrame:
    """Same data as examples/nan-values.py."""
    return pd.DataFrame(
        {
            "cliente": ["Ana", "Carlos", None, "Pedro", "Ana"],
            "compras": [5, 12, None, 3, 5],
            "ciudad": ["Madrid", "Barcelona", "Madrid", None, "Madrid"],
        }
    )


def test_example_script(customers: pd.DataFrame) -> None:
    result = handle_missing(customers, {"ciudad": "Desconocida", "compras": "median"})
    assert result["ciudad"].tolist() == [
        "Madrid",
        "Barcelona",
        "Madrid",
        "Desconocida",
        "Madrid",
    ]
    assert result["compras"].tolist() == [5.0, 12.0, 5.0, 3.0, 5.0]


def test_does_not_modify_input(customers: pd.DataFrame) -> None:
    original = customers.copy()
    handle_missing(customers, {"cliente": "drop", "ciudad": "mode"})
    pd.testing.assert_frame_equal(customers, original)


def test_untouched_columns(customers: pd.DataFrame) -> None:
    result = handle_missing(customers, {"ciudad": "Desconocida"})
    pd.testing.assert_series_equal(result["compras"], customers["compras"])
    assert result["cliente"].isna().sum() == 1


def test_drop_keeps_original_index(customers: pd.DataFrame) -> None:
    result = handle_missing(customers, {"cliente": "drop"})
    assert result.index.tolist() == [0, 1, 3, 4]


def test_drop_happens_before_statistics() -> None:
    df = pd.DataFrame({"k": ["a", None, "b"], "v": [1.0, 100.0, None]})
    result = handle_missing(df, {"k": "drop", "v": "mean"})
    # La media se calcula sin la fila borrada (100), así que es 1.0 y no 50.5
    assert result["v"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("mean", 3.0), ("median", 1.5), ("mode", 1.0)],
)
def test_statistics(strategy: str, expected: float) -> None:
    df = pd.DataFrame({"v": [1.0, 1.0, 2.0, 8.0, None]})
    assert handle_missing(df, {"v": strategy})["v"].iloc[-1] == expected


def test_mode_tie_takes_first_sorted_value() -> None:
    df = pd.DataFrame({"c": ["b", "a", None]})
    assert handle_missing(df, {"c": "mode"})["c"].iloc[-1] == "a"


def test_fill_values_are_literal() -> None:
    df = pd.DataFrame({"level": ["low", None]})
    result = handle_missing(df, {}, fill_values={"level": "median"})
    assert result["level"].tolist() == ["low", "median"]


def test_nullable_integer_becomes_float_when_needed() -> None:
    df = pd.DataFrame({"v": pd.array([1, 2, None], dtype="Int64")})
    result = handle_missing(df, {"v": "mean"})
    assert result["v"].dtype == "Float64"
    assert result["v"].tolist() == [1.0, 2.0, 1.5]


def test_nullable_integer_kept_when_possible() -> None:
    df = pd.DataFrame({"v": pd.array([1, 3, None], dtype="Int64")})
    result = handle_missing(df, {"v": "median"})
    assert result["v"].dtype == "Int64"
    assert result["v"].tolist() == [1, 3, 2]


def test_category_fill_adds_category() -> None:
    df = pd.DataFrame({"c": pd.Series(["a", None], dtype="category")})
    result = handle_missing(df, {"c": "otro"})
    assert result["c"].tolist() == ["a", "otro"]
    assert isinstance(result["c"].dtype, pd.CategoricalDtype)


def test_datetime_median() -> None:
    df = pd.DataFrame({"f": pd.to_datetime(["2024-01-01", None, "2024-01-05"])})
    result = handle_missing(df, {"f": "median"})
    assert result["f"].iloc[1] == pd.Timestamp("2024-01-03")


def test_all_missing_column_is_left_as_is() -> None:
    df = pd.DataFrame({"v": [np.nan, np.nan]})
    assert handle_missing(df, {"v": "median"})["v"].isna().all()


def test_empty_dataframe() -> None:
    df = pd.DataFrame({"v": pd.Series([], dtype=float)})
    assert handle_missing(df, {"v": "median"}).shape == (0, 1)


@pytest.mark.parametrize(
    ("strategies", "fill_values"),
    [({"nope": "drop"}, None), ({}, {"nope": 1})],
)
def test_missing_column_raises(
    customers: pd.DataFrame,
    strategies: dict[str, object],
    fill_values: dict[str, object] | None,
) -> None:
    with pytest.raises(KeyError, match="Columns not found"):
        handle_missing(customers, strategies, fill_values=fill_values)


@pytest.mark.parametrize("column", ["ciudad", "flag"])
def test_average_on_non_numeric_raises(column: str) -> None:
    df = pd.DataFrame({"ciudad": ["a", None], "flag": [True, False]})
    with pytest.raises(TypeError, match="numeric or datetime"):
        handle_missing(df, {column: "mean"})


@pytest.mark.parametrize(
    ("strategies", "fill_values", "message"),
    [
        ({"compras": "mean"}, {"compras": 1}, "both strategies and fill_values"),
        ({"compras": None}, None, "cannot be missing"),
        ({"ciudad": np.nan}, None, "cannot be missing"),
    ],
)
def test_invalid_arguments_raise(
    customers: pd.DataFrame,
    strategies: dict[str, object],
    fill_values: dict[str, object] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        handle_missing(customers, strategies, fill_values=fill_values)
