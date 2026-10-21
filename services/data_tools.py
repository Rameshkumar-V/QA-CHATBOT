"""Load CSV and Excel uploads into clean pandas DataFrames."""
import csv
import io
import json
import re
from typing import Any

import pandas as pd
from services.excel_headers import column_names, preview_sheet, read_sheet

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 200_000
MAX_TOOL_CHARS = 3_000
MAX_TOTAL_BYTES = 60 * 1024 * 1024
SUPPORTED_TYPES = ['csv', 'tsv', 'xlsx', 'xlsm', 'xls']
MISSING_MARKERS = frozenset({'', '-', '--', 'na', 'n/a', 'nan', 'none', 'null', 'nat',
                             'not provided', 'not submitted', 'not available'})
ISO_DATE = re.compile(r'^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$')
SLASH_DATE = re.compile(r'^(\d{1,2})[/.-](\d{1,2})[/.-]\d{4}(?: \d{1,2}:\d{2}(?::\d{2})?)?$')
FORMATTED_NUMBER = re.compile(r'^[-+]?\s*[₹$€£]?\s*[\d,]+(?:\.\d+)?$')


def missing_mask(series: pd.Series) -> pd.Series:
    """Blank/null cells and explicit absence markers; zero is not assumed missing."""
    normalized = series.astype('string').str.strip().str.casefold()
    return (series.isna() | normalized.isin(MISSING_MARKERS)).fillna(True).astype(bool)


def compact(value: Any, max_chars=MAX_TOOL_CHARS) -> str:
    """Return valid JSON, explicitly marking budget omissions instead of slicing JSON."""
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, str) and len(item) > 300:
            return item[:300] + ' [truncated]'
        if pd.api.types.is_scalar(item) and pd.isna(item):
            return None
        return item

    payload = clean(value)
    def encode():
        return json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False)
    result = encode()
    if len(result) <= max_chars:
        return result
    if not isinstance(payload, dict):
        payload = {'items': payload}
    payload['truncated'] = True
    # Shrink the largest container, retaining metadata and whole JSON records.
    while len(encode()) > max_chars:
        candidates = [(len(json.dumps(v, default=str)), k) for k, v in payload.items()
                      if isinstance(v, (list, dict)) and v]
        if not candidates:
            return json.dumps({'truncated': True, 'error': 'Result exceeds output budget; request fewer columns or rows.'})
        _, key = max(candidates)
        container = payload[key]
        keep = len(container) // 2
        payload[key] = container[:keep] if isinstance(container, list) else dict(list(container.items())[:keep])
        payload['omitted_details'] = 'Output budget reached; request a narrower result.'
    return encode()


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame column names are strings and mixed-type columns are Arrow-compatible."""
    df.columns = column_names(df.columns)
    for col in (c for c in df.columns if df[c].dtype == 'object'):
        if df[col].map(lambda x: isinstance(x, bytes)).any():
            df[col] = df[col].map(lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else x)
    try:
        from streamlit.dataframe_util import fix_arrow_incompatible_column_types
        return fix_arrow_incompatible_column_types(df)
    except Exception:
        df_copy = None
        for col in df.columns:
            if df[col].dtype == 'object':
                types = {type(x) for x in df[col].dropna().iloc[:100]}
                if len(types) > 1:
                    if df_copy is None:
                        df_copy = df.copy()
                    df_copy[col] = df[col].astype('string')
        return df_copy if df_copy is not None else df


def _text_values(series):
    if not (series.dtype == object or isinstance(series.dtype, pd.StringDtype)):
        return None
    values = series[~missing_mask(series)]
    if values.empty or not values.map(lambda v: isinstance(v, str)).all():
        return None
    return values.astype(str).str.strip()


def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that are unambiguously dates or formatted numbers (₹1,200)."""
    for column in df.columns:
        values = _text_values(df[column])
        if values is None:
            continue
        if values.str.match(ISO_DATE).all():
            df[column] = pd.to_datetime(df[column].where(~missing_mask(df[column])), errors='coerce', format='ISO8601')
            continue
        parts = values.str.extract(SLASH_DATE)
        if parts.notna().all().all():
            first, second = parts[0].astype(int), parts[1].astype(int)
            # Only convert when day/month order is provable from the data.
            if (first > 12).any() and (second <= 12).all():
                dayfirst = True
            elif (second > 12).any() and (first <= 12).all():
                dayfirst = False
            else:
                continue
            df[column] = pd.to_datetime(df[column].where(~missing_mask(df[column])), errors='coerce',
                                        dayfirst=dayfirst, format='mixed')
            continue
        formatted = values.str.contains(r'[,₹$€£]')
        cleaned = values.str.replace(r'[\s,₹$€£]', '', regex=True)
        leading_zero = cleaned.str.match(r'^[-+]?0\d')
        if formatted.any() and values.str.match(FORMATTED_NUMBER).all() and not leading_zero.any():
            df[column] = pd.to_numeric(df[column].where(~missing_mask(df[column]))
                                       .astype('string').str.replace(r'[\s,₹$€£]', '', regex=True), errors='coerce')
    return df


def _read_csv(raw, suffix):
    sample = raw[:65536].decode('utf-8', errors='ignore')
    delimiter = '\t'
    if suffix != 'tsv':
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
        except csv.Error:
            delimiter = ','
    for encoding in ('utf-8-sig', 'latin-1'):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=delimiter, encoding=encoding, nrows=MAX_ROWS + 1)
        except UnicodeDecodeError:
            continue


def read_upload(file, excel_headers=None) -> dict[str, pd.DataFrame]:
    filename = file.name
    raw = file.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError(f"{filename}: files must be 20 MB or smaller")
    suffix = filename.rsplit('.', 1)[-1].lower()
    tables = {}
    if suffix in ('csv', 'tsv'):
        tables[filename] = _read_csv(raw, suffix)
    elif suffix in ('xlsx', 'xlsm', 'xls'):
        engine = 'xlrd' if suffix == 'xls' else 'openpyxl'
        with pd.ExcelFile(io.BytesIO(raw), engine=engine) as book:
            if len(book.sheet_names) > 30:
                raise ValueError('Excel file has too many sheets (maximum 30)')
            for sheet in book.sheet_names:
                options = (excel_headers or {}).get(sheet, {})
                tables[f'{filename} :: {sheet}'] = read_sheet(book, sheet, MAX_ROWS, **options)
    else:
        raise ValueError(f'{filename}: upload an Excel (.xlsx, .xls) or CSV file')
    for name, df in tables.items():
        if len(df) > MAX_ROWS:
            raise ValueError(f'{name}: maximum {MAX_ROWS:,} rows per table')
        attrs = dict(df.attrs)
        df = infer_types(sanitize_dataframe(df))
        df.attrs.update(attrs)
        tables[name] = df
    return tables


def excel_previews(file):
    raw = file.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError('Files must be 20 MB or smaller.')
    engine = 'xlrd' if file.name.lower().endswith('.xls') else 'openpyxl'
    with pd.ExcelFile(io.BytesIO(raw), engine=engine) as book:
        if len(book.sheet_names) > 30:
            raise ValueError('Excel file has too many sheets (maximum 30).')
        return {sheet: preview_sheet(book, sheet) for sheet in book.sheet_names}
