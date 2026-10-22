"""Tests for jhb_dataclean.text."""

import pandas as pd
import pytest

from jhb_dataclean import normalize_text


@pytest.fixture
def cities(text_dtype: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": pd.Series(
                [" Madrid ", "BARCELONA", "Barcleona", "   ", None], dtype=text_dtype
            ),
            "sales": [1, 2, 3, 4, 5],
        }
    )


def test_default_cleanup(cities: pd.DataFrame) -> None:
    result = normalize_text(cities, replacements={"barcleona": "barcelona"})
    values = result["city"].tolist()
    assert values[:3] == ["madrid", "barcelona", "barcelona"]
    # Tanto el texto vacío como el nulo original quedan como nulos
    assert result["city"].isna().tolist() == [False, False, False, True, True]


def test_does_not_modify_input(cities: pd.DataFrame) -> None:
    original = cities.copy()
    normalize_text(cities)
    pd.testing.assert_frame_equal(cities, original)


def test_only_text_columns_by_default(cities: pd.DataFrame) -> None:
    result = normalize_text(cities)
    pd.testing.assert_series_equal(result["sales"], cities["sales"])


def test_category_columns_are_not_text() -> None:
    df = pd.DataFrame({"c": pd.Series([" A "], dtype="category")})
    pd.testing.assert_frame_equal(normalize_text(df), df)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("lower", "málaga del sur"),
        ("upper", "MÁLAGA DEL SUR"),
        ("title", "Málaga Del Sur"),
        (None, "málaga DEL sur"),
    ],
)
def test_case(case: str | None, expected: str) -> None:
    df = pd.DataFrame({"c": ["  málaga   DEL sur "]})
    assert normalize_text(df, case=case)["c"].tolist() == [expected]


def test_remove_accents() -> None:
    df = pd.DataFrame({"c": ["Málaga", "Cádiz", None]})
    result = normalize_text(df, remove_accents=True)
    assert result["c"].tolist()[:2] == ["malaga", "cadiz"]
    assert result["c"].isna().tolist() == [False, False, True]


def test_keep_spaces_and_empty_strings() -> None:
    df = pd.DataFrame({"c": ["a   b", "   "]})
    result = normalize_text(df, collapse_spaces=False, empty_as_na=False)
    assert result["c"].tolist() == ["a   b", ""]


def test_explicit_columns() -> None:
    df = pd.DataFrame({"a": [" X "], "b": [" Y "]})
    result = normalize_text(df, ["a"])
    assert result.to_dict("list") == {"a": ["x"], "b": [" Y "]}


def test_empty_dataframe() -> None:
    assert normalize_text(pd.DataFrame()).shape == (0, 0)


def test_missing_column_raises(cities: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Columns not found"):
        normalize_text(cities, ["nope"])


def test_non_text_column_raises(cities: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="do not contain text"):
        normalize_text(cities, ["sales"])


def test_invalid_case_raises(cities: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="case must be"):
        normalize_text(cities, case="capitalize")  # type: ignore[arg-type]
