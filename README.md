# jhb-dataclean

Small, predictable functions to clean pandas DataFrames: column names, free
text, data types, missing values, duplicates and outliers, plus reading and
writing CSV, Excel and Parquet files.

Every function takes a DataFrame and returns a **new** one: your input is never
modified, so the steps chain naturally.

## Installation

```bash
pip install jhb-dataclean
```

CSV support comes with pandas. Excel and Parquet need an optional extra:

```bash
pip install "jhb-dataclean[excel]"    # .xlsx files (openpyxl)
pip install "jhb-dataclean[parquet]"  # .parquet files (pyarrow)
pip install "jhb-dataclean[all]"      # both
```

Requires Python 3.11+ and pandas 2.2+ (pandas 3 is supported).

## Quick start

```pycon
>>> import pandas as pd
>>> import jhb_dataclean as dc
>>> raw = pd.DataFrame({
...     " Order ID ": [1, 2, 2, 3, 4],
...     "City": [" Madrid ", "MÁLAGA", "MÁLAGA", "Barcleona", None],
...     "orderDate": ["2024-01-15", "16/01/2024", "16/01/2024", "2024-01-17", "2024-01-18"],
...     "Amount": ["100.5", "250", "250", "N/A", "99000"],
...     "Paid": ["yes", "no", "no", "SI", "1"],
... })
>>> df = dc.standardize_columns(raw)
>>> df = dc.normalize_text(df, remove_accents=True, replacements={"barcleona": "barcelona"})
>>> df = dc.convert_types(df, {"order_date": "datetime", "amount": "numeric", "paid": "boolean"})
>>> df = dc.remove_duplicates(df, ["order_id"])
>>> df = dc.handle_missing(df, {"city": "unknown", "amount": "median"})
>>> df = dc.detect_outliers(df, ["amount"])
>>> df.columns.tolist()
['order_id', 'city', 'order_date', 'amount', 'paid', 'amount_outlier']
>>> df["city"].tolist()
['madrid', 'malaga', 'barcelona', 'unknown']
>>> df["amount"].tolist()
[100.5, 250.0, 250.0, 99000.0]
>>> df["amount_outlier"].tolist()
[False, False, False, True]

```

## Functions

| Function | What it does |
| --- | --- |
| `read_data` | Read a `.csv`, `.xlsx` or `.parquet` file, choosing the format from the extension |
| `standardize_columns` | Turn column names into `snake_case` without accents or symbols |
| `normalize_text` | Trim spaces, fix case, remove accents and replace known typos |
| `convert_types` | Convert columns to datetime, numeric, boolean or string |
| `handle_missing` | Drop rows or fill missing values with mean, median, mode or a value |
| `remove_duplicates` | Remove duplicated rows, choosing which copy to keep |
| `detect_outliers` | Flag outliers with the IQR or z-score method |
| `write_data` | Write a DataFrame to `.csv`, `.xlsx` or `.parquet` |

Every function has a full docstring: run `help(dc.convert_types)` to see all
its options.

### `standardize_columns`

```pycon
>>> df = pd.DataFrame(columns=[" Fecha Pedido ", "precioUnitario", "Categoría", "Precio (€)"])
>>> dc.standardize_columns(df).columns.tolist()
['fecha_pedido', 'precio_unitario', 'categoria', 'precio']

```

It raises `ValueError` if two columns would end up with the same name.

### `normalize_text`

By default it works on every text column: it trims and collapses spaces,
lowercases, and turns empty strings into missing values.

```pycon
>>> df = pd.DataFrame({"city": [" Málaga ", "MALAGA", "Barcleona", "   "]})
>>> clean = dc.normalize_text(df, remove_accents=True, replacements={"barcleona": "barcelona"})
>>> clean["city"].tolist()[:3]
['malaga', 'malaga', 'barcelona']
>>> clean["city"].isna().tolist()
[False, False, False, True]

```

Use `case="upper"`, `case="title"` or `case=None` to change the case rule.

### `convert_types`

Pass a mapping of column to target type: `"datetime"`, `"numeric"`,
`"boolean"` or `"string"`. Values that cannot be converted become missing.

```pycon
>>> df = pd.DataFrame({
...     "date": ["2024-01-15", "15/03/2024"],
...     "amount": ["1500.50", "N/A"],
...     "active": ["sí", "no"],
...     "zip_code": [28001, 8001],
... })
>>> types = {"date": "datetime", "amount": "numeric", "active": "boolean", "zip_code": "string"}
>>> result = dc.convert_types(df, types, zfill={"zip_code": 5})
>>> result["date"].dt.strftime("%Y-%m-%d").tolist()
['2024-01-15', '2024-03-15']
>>> result["active"].tolist()
[True, False]
>>> result["zip_code"].tolist()
['28001', '08001']

```

- Dates are read day first (`15/03/2024` is 15 March); ISO dates like
  `2024-04-10` are always read correctly. Use `dayfirst=False` for
  month-first data.
- Booleans accept `si`/`yes`/`true`/`1` and `no`/`false`/`0`, ignoring case
  and accents. Change them with `true_values` and `false_values`.
- `zfill` restores leading zeros in codes such as zip codes.

### `handle_missing`

Pass a mapping of column to strategy: `"drop"`, `"mean"`, `"median"`, `"mode"`
or any other value to fill with. Rows are dropped **before** the statistics
are computed.

```pycon
>>> df = pd.DataFrame({
...     "customer": ["Ana", "Carlos", None, "Pedro"],
...     "purchases": [5, 12, 4, None],
...     "city": ["Madrid", None, "Madrid", "Sevilla"],
... })
>>> result = dc.handle_missing(df, {"customer": "drop", "purchases": "median", "city": "unknown"})
>>> result["purchases"].tolist()
[5.0, 12.0, 8.5]
>>> result["city"].tolist()
['Madrid', 'unknown', 'Sevilla']

```

To fill with a value that is also a strategy name (for example the text
`"median"`), use `fill_values={"level": "median"}`.

### `remove_duplicates`

By default only exact duplicates are removed. Pass the key columns to compare
only those, and `order_by` to choose which copy to keep:

```pycon
>>> df = pd.DataFrame({
...     "order_id": [1, 2, 1],
...     "amount": [100, 250, 120],
...     "date": ["2024-01-20", "2024-01-16", "2024-01-15"],
... })
>>> dc.remove_duplicates(df, ["order_id"], keep="last", order_by="date")["amount"].tolist()
[100, 250]

```

The result keeps the original row order and index.

### `detect_outliers`

Rows are never removed: a boolean column `<column>_outlier` is added so you can
decide what to do.

```pycon
>>> df = pd.DataFrame({"price": [25, 30, 28, 5000, 22, 31, 27]})
>>> dc.detect_outliers(df)["price_outlier"].tolist()
[False, False, False, True, False, False, False]
>>> int(dc.detect_outliers(df, method="zscore", threshold=2)["price_outlier"].sum())
1

```

The IQR method (default, `threshold=1.5`) works well with few rows. The
z-score method (`threshold=3`) needs more data: with 7 rows no value can
reach a z-score of 3.

### `read_data` and `write_data`

```python
df = dc.read_data("sales.csv", sep=";")
df = dc.read_data("sales.xlsx", sheet_name="2024")
dc.write_data(df, "sales_clean.parquet")
```

Extra arguments go straight to the pandas function (`pd.read_csv`,
`df.to_excel`...). The index is not written unless you pass `index=True`.

## Things to know about file formats

Each format keeps types differently, so the same data may come back with
different types. Pass the result through `convert_types` to get the same
DataFrame from any format.

- **CSV** stores everything as text: `"N/A"` and empty cells are read as
  missing, and codes like `08001` lose their leading zero unless you pass
  `dtype={"zip_code": str}`.
- **Excel** may return numbers and dates already typed, drops leading zeros,
  and does not keep a final row that is completely empty.
- **Parquet** keeps the original types.

## Development

The project uses [pixi](https://pixi.sh) to manage the environment:

```bash
pixi install
pixi run test   # pytest
pixi run lint   # ruff
```

## License

[Apache-2.0](LICENSE)
