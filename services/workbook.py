"""Loaded sheets addressed by short, SQL-safe IDs, plus a compact overview for the agent."""
import re

import pandas as pd

from services.data_tools import missing_mask
from services.table_store import TableStore, to_number

SQL_KEYWORDS = frozenset({'all', 'and', 'as', 'by', 'case', 'check', 'default', 'end', 'from', 'group',
                          'in', 'index', 'is', 'join', 'key', 'limit', 'not', 'null', 'on', 'or', 'order',
                          'primary', 'references', 'select', 'set', 'table', 'to', 'union', 'values', 'where'})


def sheet_id(name, taken):
    """'Sales Report.xlsx :: Q1' -> 'sales_report_q1'; 'orders.csv' -> 'orders'."""
    file, _, sheet = name.partition(' :: ')
    base = file.rsplit('.', 1)[0] + (' ' + sheet if sheet else '')
    slug = re.sub(r'[^0-9a-z]+', '_', base.casefold()).strip('_')[:40] or 'sheet'
    if slug[0].isdigit():
        slug = 't_' + slug
    if slug in SQL_KEYWORDS:
        slug += '_data'
    candidate, suffix = slug, 2
    while candidate in taken:
        candidate = f'{slug}_{suffix}'
        suffix += 1
    return candidate


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return 'boolean'
    if pd.api.types.is_numeric_dtype(series):
        return 'number'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'date'
    sample = series[~missing_mask(series)].head(1000)
    if len(sample) and sample.map(to_number).notna().mean() >= 0.9:
        return 'number'
    return 'text'


class Workbook:
    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.frames, self.labels = {}, {}
        for name, frame in tables.items():
            key = sheet_id(name, self.frames)
            self.frames[key] = frame
            self.labels[key] = name.replace(' :: ', ' › ')
        self.store = TableStore(self.frames)

    def close(self):
        self.store.close()

    def overview(self, ids=None, max_chars=4000, max_columns=60):
        lines = ['Sheets (use the id in tool calls and as the SQL table name):']
        for key in ids or self.frames:
            frame = self.frames[key]
            lines.append(f'- {key}: "{self.labels[key]}", {len(frame):,} rows × {len(frame.columns)} columns')
            if sum(map(len, lines)) > max_chars:
                continue
            shown = [f'{c} ({column_kind(frame[c])})' for c in frame.columns[:max_columns]]
            more = len(frame.columns) - max_columns
            lines.append('  columns: ' + ', '.join(shown) + (f', … {more} more (describe_sheet)' if more > 0 else ''))
            if frame.attrs.get('generic_headers'):
                lines.append('  note: unlabelled columns ' + ', '.join(frame.attrs['generic_headers'][:10])
                             + ' (the header row may be wrong; the user can change it in Excel header settings)')
        return '\n'.join(lines)
