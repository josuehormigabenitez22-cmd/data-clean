"""Tests for jhb_dataclean.duplicates."""

import pandas as pd
import pytest

from jhb_dataclean import remove_duplicates


@pytest.fixture
def orders() -> pd.DataFrame:
    """Same data as examples/duplicated-values.py."""
    return pd.DataFrame(
        {
            "pedido_id": [1, 2, 3, 2, 4, 3],
            "cliente": ["Ana", "Carlos", "Lucía", "Carlos", "Pedro", "Lucía"],
            "importe": [100, 250, 75, 250, 180, 75],
            "fecha": [
                "2024-01-15",
                "2024-01-16",
                "2024-01-16",
                "2024-01-16",
                "2024-01-17",
                "2024-01-16",
            ],
        }
    )


def test_exact_duplicates_by_default(orders: pd.DataFrame) -> None:
    result = remove_duplicates(orders)
    assert result.index.tolist() == [0, 1, 2, 4]


def test_does_not_modify_input(orders: pd.DataFrame) -> None:
    original = orders.copy()
    remove_duplicates(orders, ["pedido_id"], keep="last", order_by="fecha")
    pd.testing.assert_frame_equal(orders, original)


@pytest.mark.parametrize(
    ("keep", "expected"),
    [("first", [0, 1, 2, 4]), ("last", [0, 3, 4, 5]), (False, [0, 4])],
)
def test_keep(orders: pd.DataFrame, keep: str | bool, expected: list[int]) -> None:
    result = remove_duplicates(orders, ["pedido_id"], keep=keep)  # type: ignore[arg-type]
    assert result.index.tolist() == expected


def test_result_matches_pandas(orders: pd.DataFrame) -> None:
    # Sin order_by debe hacer lo mismo que drop_duplicates
    pd.testing.assert_frame_equal(
        remove_duplicates(orders, ["cliente"], keep="last"),
        orders.drop_duplicates(subset=["cliente"], keep="last"),
    )


@pytest.fixture
def versions() -> pd.DataFrame:
    """The same order saved three times, not in date order."""
    return pd.DataFrame(
        {
            "order_id": [1, 2, 1, 1],
            "amount": [100, 250, 120, 90],
            "date": ["2024-01-20", "2024-01-16", "2024-01-15", "2024-01-10"],
        }
    )


@pytest.mark.parametrize(("keep", "amount"), [("first", 90), ("last", 100)])
def test_order_by_chooses_copy(versions: pd.DataFrame, keep: str, amount: int) -> None:
    result = remove_duplicates(versions, ["order_id"], keep=keep, order_by="date")  # type: ignore[arg-type]
    assert result.loc[result["order_id"] == 1, "amount"].tolist() == [amount]


def test_order_by_keeps_original_order_and_index(versions: pd.DataFrame) -> None:
    result = remove_duplicates(versions, ["order_id"], keep="last", order_by="date")
    assert result.index.tolist() == [0, 1]


@pytest.mark.parametrize("keep", ["first", "last"])
def test_missing_order_by_value_is_never_preferred(keep: str) -> None:
    df = pd.DataFrame({"id": [1, 1, 1], "date": ["2024-01-01", None, "2024-01-09"]})
    result = remove_duplicates(df, ["id"], keep=keep, order_by="date")  # type: ignore[arg-type]
    assert result["date"].notna().all()


def test_order_by_several_columns() -> None:
    df = pd.DataFrame({"id": [1, 1, 1], "day": [2, 2, 1], "hour": [9, 18, 23]})
    result = remove_duplicates(df, ["id"], keep="last", order_by=["day", "hour"])
    assert result.index.tolist() == [1]


def test_repeated_index_labels() -> None:
    # Con etiquetas repetidas el índice no identifica cada fila: se usan posiciones
    df = pd.DataFrame({"id": [1, 2, 1], "v": ["a", "b", "c"]}, index=[0, 0, 0])
    result = remove_duplicates(df, ["id"], keep="last", order_by="v")
    assert result["v"].tolist() == ["b", "c"]


def test_empty_dataframe() -> None:
    df = pd.DataFrame({"id": pd.Series([], dtype=int)})
    assert remove_duplicates(df, ["id"], order_by="id").empty


def test_missing_column_raises(orders: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match=r"Columns not found: \['nope', 'nada'\]"):
        remove_duplicates(orders, ["nope"], order_by="nada")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"columns": []}, "columns cannot be empty"),
        ({"keep": "none"}, "keep must be"),
        # 0 == False en Python, pero no debe aceptarse como keep=False
        ({"keep": 0}, "keep must be"),
    ],
)
def test_invalid_arguments_raise(
    orders: pd.DataFrame, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        remove_duplicates(orders, **kwargs)  # type: ignore[arg-type]
