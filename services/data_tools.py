"""Read-only, bounded file analysis tools used by the LangChain agent."""
import io
import json
import re
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from services.search import DocumentSearch

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 200_000
MAX_TEXT_CHARS = 1_000_000
MAX_TOOL_CHARS = 3_000
MAX_TOTAL_BYTES = 60 * 1024 * 1024
MISSING_MARKERS = frozenset({'', '-', '--', 'na', 'n/a', 'nan', 'none', 'null',
                             'not provided', 'not submitted', 'not available'})


def missing_mask(series: pd.Series) -> pd.Series:
    """Blank/null cells and explicit absence markers; zero is not assumed missing."""
    normalized = series.astype('string').str.strip().str.casefold()
    return series.isna() | normalized.isin(MISSING_MARKERS)


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
    df.columns = [str(c) for c in df.columns]
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


def read_upload(file) -> dict[str, pd.DataFrame]:
    filename = file.name
    raw = file.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError(f"{filename}: files must be 20 MB or smaller")
    suffix = filename.rsplit('.', 1)[-1].lower()
    tables = {}
    if suffix == 'csv':
        try:
            df = pd.read_csv(io.BytesIO(raw), nrows=MAX_ROWS + 1)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding='latin-1', nrows=MAX_ROWS + 1)
        tables[filename] = df
    elif suffix in ('xlsx', 'xls'):
        engine = 'openpyxl' if suffix == 'xlsx' else 'xlrd'
        with pd.ExcelFile(io.BytesIO(raw), engine=engine) as book:
            if len(book.sheet_names) > 30:
                raise ValueError('Excel file has too many sheets (maximum 30)')
            for sheet in book.sheet_names:
                tables[f'{filename} :: {sheet}'] = pd.read_excel(book, sheet_name=sheet, nrows=MAX_ROWS + 1)
    elif suffix == 'json':
        data = json.loads(raw.decode('utf-8-sig'))
        if isinstance(data, list):
            tables[filename] = pd.json_normalize(data)
        elif isinstance(data, dict):
            try:
                tables[filename] = pd.json_normalize(data)
            except Exception:
                tables[filename] = pd.DataFrame([{'json': compact(data)}])
        else:
            tables[filename] = pd.DataFrame({'value': [data]})
    elif suffix in ('txt', 'md'):
        text = raw.decode('utf-8-sig', errors='replace')
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError('Text exceeds 1,000,000 characters; split the document into smaller files.')
        tables[filename] = pd.DataFrame({'text': text.splitlines()})
    elif suffix == 'pdf':
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError('Password-protected PDFs are not supported')
        lines = []
        used = 0
        if len(reader.pages) > 100:
            raise ValueError('PDF exceeds 100 pages; split it into smaller files.')
        for i, page in enumerate(reader.pages, 1):
            chunk = page.extract_text() or ''
            if chunk:
                lines.append({'page': i, 'text': chunk})
                used += len(chunk)
            if used > MAX_TEXT_CHARS:
                raise ValueError('PDF text exceeds 1,000,000 characters; split it into smaller files.')
        if not lines:
            raise ValueError('No readable text found in PDF. Use a text-based PDF or run OCR first.')
        tables[filename] = pd.DataFrame(lines, columns=['page', 'text'])
    else:
        raise ValueError('Unsupported format')
    for name, df in tables.items():
        if len(df) > MAX_ROWS:
            raise ValueError(f'{name}: maximum {MAX_ROWS:,} rows per table')
        tables[name] = sanitize_dataframe(df)
    return tables


def make_tools(tables: dict[str, pd.DataFrame], search_index=None):
    search_index = search_index or DocumentSearch(tables)
    def get_table(table: str) -> pd.DataFrame:
        if table not in tables:
            raise ValueError(f'Unknown table {table!r}. Call list_tables first.')
        return tables[table]

    @tool
    def search_documents(table: str, query: str, offset: int = 0) -> str:
        """Search all rows via a full-text index. Use a few keywords. Returns 3 ranked passages with row/column/page sources; offset pages through matches."""
        try:
            return compact(search_index.search(table, query, offset=offset))
        except Exception as exc:
            return compact({'error': str(exc)})

    @tool
    def list_tables() -> str:
        """List available uploaded files/Excel sheets and their row/column counts."""
        return compact({'total_tables': len(tables), 'tables': [{'table': k, 'rows': len(v), 'columns': len(v.columns)} for k, v in tables.items()]})

    @tool
    def describe_table(table: str, column: str = '', start: int = 0) -> str:
        """List column names, types and missing counts (12/page). column adds stats and 2 samples for one column. start pages schema."""
        try:
            df = get_table(table)
            total_columns = len(df.columns)
            start = max(0, int(start))
            df = df[[column]] if column else df.iloc[:, start:start + 12]
            result = {'table': table, 'rows': len(df),
                      'total_columns': total_columns, 'start': start,
                      'next_start': start + 12 if not column and start + 12 < total_columns else None,
                      'columns': [{'name': str(c), 'dtype': str(df[c].dtype),
                                   'nulls': int(df[c].isna().sum()),
                                   'missing': int(missing_mask(df[c]).sum())} for c in df.columns]}
            if column:
                result['sample'] = df.head(2).to_dict('records')
                numeric = df.select_dtypes(include='number')
                if len(numeric.columns):
                    result['numeric_statistics'] = numeric.describe().round(3).to_dict()
            return compact(result)
        except Exception as e:
            return f'Error: {e}'

    @tool
    def preview_rows(table: str, start: int = 0, count: int = 10) -> str:
        """View a bounded slice of table rows. start is a zero-based row index; count at most 20."""
        try:
            df = get_table(table)
            start = max(0, min(int(start), len(df)))
            count = max(1, min(int(count), 20))
            return compact({'table': table, 'start': start, 'rows': df.iloc[start:start+count].fillna('').to_dict('records')})
        except Exception as e:
            return f'Error: {e}'

    @tool
    def value_counts(table: str, column: str, top_n: int = 15) -> str:
        """Count occurrences of categories/values in a column. Useful for categorical distribution."""
        try:
            df = get_table(table)
            if column not in df:
                return f'Unknown column: {column}. Available: {list(df.columns)}'
            counts = df[column].astype('string').fillna('<missing>').value_counts().head(max(1, min(top_n, 30)))
            return compact({'table': table, 'column': column, 'counts': [{'value': k, 'count': int(v)} for k, v in counts.items()]})
        except Exception as e:
            return f'Error: {e}'

    @tool
    def aggregate(table: str, column: str, operation: str = 'mean', group_by: str = '') -> str:
        """Calculate sum, mean, min, max, median or count on a column, optionally grouped by a categorical column."""
        try:
            df = get_table(table)
            if column not in df:
                return f'Unknown column: {column}. Available: {list(df.columns)}'
            operation = operation.lower()
            if operation not in ('sum', 'mean', 'min', 'max', 'median', 'count'):
                return 'operation must be sum, mean, min, max, median or count'
            selected = df[list(dict.fromkeys([column] + ([group_by] if group_by else [])))].copy()
            if operation != 'count':
                selected[column] = pd.to_numeric(selected[column], errors='coerce')
            if group_by:
                if group_by not in df:
                    return f'Unknown group_by column: {group_by}'
                result = selected.groupby(group_by, dropna=False)[column].agg(operation).sort_values(ascending=False)
                return compact({'operation': operation, 'column': column, 'group_by': group_by,
                                'rows_scanned': len(df), 'total_groups': len(result), 'returned_groups': min(20, len(result)),
                                'groups': [{'group': str(k), 'result': None if pd.isna(v) else float(v)} for k, v in result.head(20).items()]})
            result = getattr(selected[column], operation)()
            return compact({'operation': operation, 'column': column, 'rows_scanned': len(df), 'result': None if pd.isna(result) else float(result)})
        except Exception as e:
            return f'Error: {e}'

    @tool
    def filter_rows(table: str, column: str, operator: str, value: str = '', limit: int = 15,
                    columns: list[str] | None = None) -> str:
        """Query all rows: is_missing/is_not_missing (no value needed), ==, !=, >, >=, <, <=, contains. columns selects returned fields. Returns exact match count and up to 20 matching rows."""
        try:
            df = get_table(table)
            if column not in df:
                return f'Unknown column: {column}. Available: {list(df.columns)}'
            if columns is not None and any(c not in df for c in columns):
                return compact({'error': 'Unknown return column. Use exact names from describe_table.'})
            if operator in ('is_missing', 'is_not_missing'):
                mask = missing_mask(df[column])
                if operator == 'is_not_missing':
                    mask = ~mask
            elif operator == 'contains':
                mask = df[column].astype('string').str.contains(re.escape(value), case=False, na=False, regex=True)
            elif operator in ('>', '>=', '<', '<='):
                nums = pd.to_numeric(df[column], errors='coerce')
                needle = float(value)
                mask = {'>': nums > needle, '>=': nums >= needle, '<': nums < needle, '<=': nums <= needle}[operator]
            elif operator in ('==', '!='):
                mask = df[column].astype('string').fillna('<missing>').str.lower().eq(value.lower())
                if operator == '!=':
                    mask = ~mask
            else:
                return compact({'error': 'Use is_missing, is_not_missing, ==, !=, >, >=, <, <= or contains.'})
            found = df.loc[mask]
            selected = found[columns] if columns is not None else found
            result = {'table': table, 'column': column, 'rows_scanned': len(df), 'matches': len(found),
                      'rows': selected.head(max(1, min(limit, 20))).to_dict('records')}
            if operator in ('is_missing', 'is_not_missing'):
                result['missing_definition'] = 'Null, blank/whitespace, -, --, NA, N/A, NaN, None, null, not provided/submitted/available. Zero is not treated as missing.'
            return compact(result)
        except Exception as e:
            return f'Error: {e}'

    return [list_tables, describe_table, preview_rows, value_counts, aggregate, filter_rows, search_documents]
