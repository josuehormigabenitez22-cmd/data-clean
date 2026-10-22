"""Tests for jhb_dataclean.io."""

from pathlib import Path

import pandas as pd
import pytest

from jhb_dataclean import convert_types, read_data, write_data

pytest.importorskip("openpyxl")
pytest.importorskip("pyarrow")

EXTENSIONS = [".csv", ".xlsx", ".parquet"]


@pytest.fixture
def sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["Madrid", "Málaga", "Sevilla"],
            "sales": [3, 5, 8],
            "price": [1.5, 2.0, 2.5],
        }
    )


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_round_trip(tmp_path: Path, sales: pd.DataFrame, extension: str) -> None:
    path = tmp_path / f"sales{extension}"
    write_data(sales, path)
    pd.testing.assert_frame_equal(read_data(path), sales)


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_missing_values_round_trip(tmp_path: Path, extension: str) -> None:
    # Miramos solo isna(): en pandas 2 el nulo de texto vuelve como NaN desde
    # CSV y Excel pero como None desde Parquet. La columna "sales" evita una fila
    # entera vacía al final, que Excel no conserva al leer
    df = pd.DataFrame({"city": ["Madrid", None], "sales": [3, 5], "price": [1.5, None]})
    path = tmp_path / f"sales{extension}"
    write_data(df, path)
    assert read_data(path).isna().to_dict("list") == {
        "city": [False, True],
        "sales": [False, False],
        "price": [False, True],
    }


@pytest.mark.parametrize("extension", EXTENSIONS)
def test_write_does_not_modify_input(
    tmp_path: Path, sales: pd.DataFrame, extension: str
) -> None:
    original = sales.copy()
    write_data(sales, tmp_path / f"sales{extension}")
    pd.testing.assert_frame_equal(sales, original)


def test_accepts_str_and_uppercase_extension(
    tmp_path: Path, sales: pd.DataFrame
) -> None:
    path = str(tmp_path / "SALES.CSV")
    write_data(sales, path)
    assert read_data(path).shape == sales.shape


def test_index_is_not_written_by_default(tmp_path: Path, sales: pd.DataFrame) -> None:
    path = tmp_path / "sales.csv"
    write_data(sales, path)
    assert "Unnamed: 0" not in read_data(path).columns


def test_index_can_be_written(tmp_path: Path, sales: pd.DataFrame) -> None:
    path = tmp_path / "sales.csv"
    write_data(sales, path, index=True)
    assert read_data(path).columns[0] == "Unnamed: 0"


def test_kwargs_are_passed_to_pandas(tmp_path: Path, sales: pd.DataFrame) -> None:
    path = tmp_path / "sales.csv"
    write_data(sales, path, sep=";")
    assert path.read_text(encoding="utf-8").startswith("city;sales;price")
    pd.testing.assert_frame_equal(read_data(path, sep=";"), sales)


def test_excel_sheet_name(tmp_path: Path, sales: pd.DataFrame) -> None:
    path = tmp_path / "sales.xlsx"
    write_data(sales, path, sheet_name="Ventas")
    pd.testing.assert_frame_equal(read_data(path, sheet_name="Ventas"), sales)


def test_same_data_from_every_format(tmp_path: Path) -> None:
    # El mismo dato sucio guardado en los tres formatos: cada formato lo lee con
    # tipos distintos (el CSV ya convierte "N/A" en nulo), pero tras
    # convert_types el resultado debe ser idéntico
    raw = pd.DataFrame(
        {
            "fecha": ["2024-01-15", "15/03/2024", None],
            "importe": ["1500.50", "N/A", " 2300 "],
            "activo": ["si", "no", None],
            "codigo_postal": [28001, 8001, 41001],
        }
    )
    types = {
        "fecha": "datetime",
        "importe": "numeric",
        "activo": "boolean",
        "codigo_postal": "string",
    }
    results = []
    for extension in EXTENSIONS:
        path = tmp_path / f"raw{extension}"
        write_data(raw, path)
        data = read_data(path)
        results.append(convert_types(data, types, zfill={"codigo_postal": 5}))  # type: ignore[arg-type]

    for result in results[1:]:
        pd.testing.assert_frame_equal(result, results[0])
    assert results[0]["codigo_postal"].tolist() == ["28001", "08001", "41001"]


def test_unsupported_extension_raises(tmp_path: Path, sales: pd.DataFrame) -> None:
    path = tmp_path / "sales.json"
    with pytest.raises(ValueError, match="Unsupported file extension '.json'"):
        write_data(sales, path)
    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_data(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File not found"):
        read_data(tmp_path / "nope.csv")


@pytest.mark.parametrize(
    ("extension", "extra"), [(".xlsx", "excel"), (".parquet", "parquet")]
)
def test_missing_engine_raises(
    tmp_path: Path,
    sales: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
    extra: str,
) -> None:
    # Simulamos que el motor no está instalado: find_spec devuelve None
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError, match=rf"jhb-dataclean\[{extra}\]"):
        write_data(sales, tmp_path / f"sales{extension}")
