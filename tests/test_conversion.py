"""Tests for jhb_dataclean.conversion."""

import numpy as np
import pandas as pd
import pytest

from jhb_dataclean import convert_types

ALL_TYPES = {
    "fecha": "datetime",
    "importe": "numeric",
    "activo": "boolean",
    "codigo_postal": "string",
}


@pytest.fixture
def raw() -> pd.DataFrame:
    """Same data as examples/incorrect-types.py."""
    return pd.DataFrame(
        {
            "fecha": ["2024-01-15", "2024-02-20", "15/03/2024", "2024-04-10"],
            "importe": ["1500.50", "2300", "N/A", "890.25"],
            "activo": ["si", "no", "si", "SI"],
            "codigo_postal": [28001, 8001, 41001, 28020],
        }
    )


def test_example_script(raw: pd.DataFrame) -> None:
    result = convert_types(raw, ALL_TYPES, zfill={"codigo_postal": 5})

    assert pd.api.types.is_datetime64_any_dtype(result["fecha"])
    assert result["fecha"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-15",
        "2024-02-20",
        "2024-03-15",
        "2024-04-10",
    ]
    assert result["importe"].tolist()[:2] == [1500.5, 2300.0]
    assert np.isnan(result["importe"].iloc[2])
    assert result["activo"].dtype == "boolean"
    assert result["activo"].tolist() == [True, False, True, True]
    assert result["codigo_postal"].tolist() == ["28001", "08001", "41001", "28020"]


def test_does_not_modify_input(raw: pd.DataFrame) -> None:
    original = raw.copy()
    convert_types(raw, ALL_TYPES, zfill={"codigo_postal": 5})
    pd.testing.assert_frame_equal(raw, original)


def test_untouched_columns(raw: pd.DataFrame) -> None:
    result = convert_types(raw, {"importe": "numeric"})
    pd.testing.assert_frame_equal(
        result.drop(columns="importe"), raw.drop(columns="importe")
    )


def test_iso_dates_ignore_dayfirst() -> None:
    # pandas 3 leería "2024-04-10" como 4 de octubre con dayfirst=True
    df = pd.DataFrame({"f": ["2024-04-10", "03/04/2024", "2024-04-10 13:45"]})
    result = convert_types(df, {"f": "datetime"})
    assert result["f"].astype(str).tolist() == [
        "2024-04-10 00:00:00",
        "2024-04-03 00:00:00",
        "2024-04-10 13:45:00",
    ]


def test_dayfirst_false() -> None:
    df = pd.DataFrame({"f": ["03/04/2024"]})
    result = convert_types(df, {"f": "datetime"}, dayfirst=False)
    assert result["f"].dt.strftime("%Y-%m-%d").tolist() == ["2024-03-04"]


def test_already_typed_columns_are_kept() -> None:
    # Así llegan los datos desde Excel o Parquet
    df = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2024-01-15", None]),
            "importe": [1.5, 2.0],
            "activo": [True, False],
        }
    )
    result = convert_types(
        df, {"fecha": "datetime", "importe": "numeric", "activo": "boolean"}
    )
    pd.testing.assert_series_equal(result["fecha"], df["fecha"])
    pd.testing.assert_series_equal(result["importe"], df["importe"])
    assert result["activo"].tolist() == [True, False]


def test_excel_like_values() -> None:
    df = pd.DataFrame(
        {
            "importe": [" 2300 ", None],
            "activo": [1.0, np.nan],
            "codigo_postal": [8001.0, np.nan],
        }
    )
    result = convert_types(
        df,
        {"importe": "numeric", "activo": "boolean", "codigo_postal": "string"},
        zfill={"codigo_postal": 5},
    )
    assert result["importe"].iloc[0] == 2300.0
    assert result["activo"].tolist()[0] is True
    assert result["activo"].isna().tolist() == [False, True]
    assert result["codigo_postal"].iloc[0] == "08001"
    assert result["codigo_postal"].isna().tolist() == [False, True]


@pytest.mark.parametrize("value", [" Sí ", "SI", "yes", "TRUE", "1", True, 1])
def test_true_values(value: object) -> None:
    df = pd.DataFrame({"b": pd.Series([value], dtype=object)})
    assert convert_types(df, {"b": "boolean"})["b"].tolist() == [True]


def test_unknown_boolean_becomes_missing() -> None:
    df = pd.DataFrame({"b": ["si", "quizás", None]})
    result = convert_types(df, {"b": "boolean"})
    assert result["b"].isna().tolist() == [False, True, True]


def test_custom_boolean_values() -> None:
    df = pd.DataFrame({"b": ["vale", "nanai"]})
    result = convert_types(
        df, {"b": "boolean"}, true_values=["vale"], false_values=["nanai"]
    )
    assert result["b"].tolist() == [True, False]


def test_category_column() -> None:
    df = pd.DataFrame({"b": pd.Series(["si", None], dtype="category")})
    assert convert_types(df, {"b": "boolean"})["b"].isna().tolist() == [False, True]


def test_all_missing_column() -> None:
    df = pd.DataFrame({"x": [None, None]})
    assert convert_types(df, {"x": "numeric"})["x"].isna().all()


def test_missing_column_raises(raw: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Columns not found"):
        convert_types(raw, {"nope": "numeric"})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"types": {"importe": "int"}}, "Unknown target types"),
        ({"types": {"importe": "numeric"}, "zfill": {"importe": 5}}, "zfill columns"),
        (
            {"types": {"codigo_postal": "string"}, "zfill": {"codigo_postal": 0}},
            "zfill widths",
        ),
        (
            {
                "types": {"activo": "boolean"},
                "true_values": ["si"],
                "false_values": ["SÍ"],
            },
            "both true_values and false_values",
        ),
    ],
)
def test_invalid_arguments_raise(
    raw: pd.DataFrame, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        convert_types(raw, **kwargs)  # type: ignore[arg-type]
