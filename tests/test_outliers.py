"""Tests for jhb_dataclean.outliers."""

import pandas as pd
import pytest

from jhb_dataclean import detect_outliers


@pytest.fixture
def products() -> pd.DataFrame:
    """Same data as examples/outliers.py."""
    return pd.DataFrame(
        {
            "producto": ["A", "B", "C", "D", "E", "F", "G"],
            "precio": [25, 30, 28, 5000, 22, 31, 27],
        }
    )


def test_example_script(products: pd.DataFrame) -> None:
    result = detect_outliers(products)
    assert result.columns.tolist() == ["producto", "precio", "precio_outlier"]
    assert result["precio_outlier"].dtype == bool
    assert result["precio_outlier"].tolist() == [False] * 3 + [True] + [False] * 3


def test_does_not_modify_input(products: pd.DataFrame) -> None:
    original = products.copy()
    detect_outliers(products)
    pd.testing.assert_frame_equal(products, original)


def test_rows_and_values_are_kept(products: pd.DataFrame) -> None:
    result = detect_outliers(products)
    pd.testing.assert_frame_equal(result[products.columns], products)


def test_matches_manual_iqr(products: pd.DataFrame) -> None:
    # La misma cuenta que examples/outliers.py hace a mano
    q1, q3 = products["precio"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    expected = (products["precio"] < low) | (products["precio"] > high)
    result = detect_outliers(products)["precio_outlier"]
    assert result.tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("threshold", "flagged"),
    [(None, 0), (2.0, 1)],
)
def test_zscore(products: pd.DataFrame, threshold: float | None, flagged: int) -> None:
    # Con 7 filas ningún z-score puede pasar de 2.27, así que el 3 por defecto
    # no marca nada
    result = detect_outliers(products, method="zscore", threshold=threshold)
    assert result["precio_outlier"].sum() == flagged


def test_numeric_columns_by_default() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.5, 2.5, 3.5],
            "flag": [True, False, True],
            "s": list("xyz"),
        }
    )
    result = detect_outliers(df)
    assert result.columns.tolist()[4:] == ["a_outlier", "b_outlier"]


def test_explicit_columns_and_suffix(products: pd.DataFrame) -> None:
    result = detect_outliers(products, ["precio"], suffix="_raro")
    assert "precio_raro" in result.columns


def test_missing_values_are_never_outliers() -> None:
    df = pd.DataFrame({"v": [1.0, 2.0, None, 2.0, 100.0]})
    assert detect_outliers(df)["v_outlier"].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_nullable_integers(method: str) -> None:
    df = pd.DataFrame({"v": pd.array([1, 2, None, 2, 1], dtype="Int64")})
    result = detect_outliers(df, method=method)  # type: ignore[arg-type]
    assert result["v_outlier"].dtype == bool
    assert not result["v_outlier"].any()


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_constant_column(method: str) -> None:
    # Con desviación 0 el z-score es 0/0 = NaN: no debe marcar nada
    df = pd.DataFrame({"v": [5, 5, 5, 5]})
    assert not detect_outliers(df, method=method)["v_outlier"].any()  # type: ignore[arg-type]


def test_empty_dataframe() -> None:
    df = pd.DataFrame({"v": pd.Series([], dtype=float)})
    assert detect_outliers(df).columns.tolist() == ["v", "v_outlier"]


def test_missing_column_raises(products: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Columns not found"):
        detect_outliers(products, ["nope"])


@pytest.mark.parametrize("column", ["producto", "flag"])
def test_non_numeric_column_raises(products: pd.DataFrame, column: str) -> None:
    products["flag"] = True
    with pytest.raises(TypeError, match="not numeric"):
        detect_outliers(products, [column])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "mad"}, "method must be"),
        ({"threshold": 0}, "threshold must be positive"),
        ({"threshold": -1.5}, "threshold must be positive"),
        ({"suffix": ""}, "suffix cannot be empty"),
        ({"suffix": "o"}, "Flag columns already exist"),
    ],
)
def test_invalid_arguments_raise(
    products: pd.DataFrame, kwargs: dict[str, object], message: str
) -> None:
    # Con suffix="o", "precio" + "o" choca con la columna "precioo"
    products["precioo"] = 0
    with pytest.raises(ValueError, match=message):
        detect_outliers(products, ["precio"], **kwargs)  # type: ignore[arg-type]
