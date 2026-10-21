"""DataFrame tools for the QA agent.

Every tool scans all rows on the server and returns a small result to the model. The model never
receives a whole sheet: row listings are capped, and the full result table is kept for the user.
"""
from dataclasses import dataclass
import datetime as dt
import difflib
import json
import operator
import re
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.data_tools import compact, missing_mask
from services.table_store import MAX_RESULT_ROWS, QueryExecutionError, to_number
from services.workbook import column_kind

MAX_MODEL_ROWS = 25
MAX_MODEL_GROUPS = 30
MAX_OUTPUT_CHARS = 4000
CELL_CHARS = 120
OP_ALIASES = {'==': '=', 'eq': '=', 'equals': '=', '<>': '!=', 'ne': '!=', 'not_equals': '!=', 'gt': '>',
              'gte': '>=', 'lt': '<', 'lte': '<=', 'like': 'contains', 'is_null': 'is_missing',
              'is_not_missing': 'not_missing', 'not_null': 'not_missing', 'is_not_null': 'not_missing',
              'not in': 'not_in', 'startswith': 'starts_with', 'endswith': 'ends_with'}
COMPARE = {'>': operator.gt, '>=': operator.ge, '<': operator.lt, '<=': operator.le}
GRAINS = {'day': 'D', 'week': 'W', 'month': 'M', 'quarter': 'Q', 'year': 'Y'}


class ToolError(ValueError):
    pass


class Filter(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    column: str
    op: Literal['=', '!=', '>', '>=', '<', '<=', 'contains', 'not_contains', 'starts_with', 'ends_with',
                'in', 'not_in', 'between', 'is_missing', 'not_missing']
    value: str = Field(default='', description='Text, number, or date as YYYY-MM-DD')
    values: list[str] = Field(default_factory=list, description='For in/not_in, or [low, high] for between')

    @field_validator('op', mode='before')
    @classmethod
    def alias(cls, value):
        value = str(value).strip().lower()
        return OP_ALIASES.get(value, value)


class Metric(BaseModel):
    op: Literal['count', 'count_distinct', 'sum', 'mean', 'median', 'min', 'max', 'std']
    column: str = Field(default='', description='Omit for a row count')

    @field_validator('op', mode='before')
    @classmethod
    def alias(cls, value):
        value = str(value).strip().lower()
        return {'avg': 'mean', 'average': 'mean', 'total': 'sum', 'nunique': 'count_distinct',
                'distinct': 'count_distinct'}.get(value, value)

    @property
    def label(self):
        return 'count' if self.op == 'count' and not self.column else f'{self.op}({self.column})'


class DescribeArgs(BaseModel):
    sheet: str = Field(description='Sheet id')
    columns: list[str] = Field(default_factory=list, description='Only these columns (default: all)')


class ReadArgs(BaseModel):
    sheet: str = Field(description='Sheet id')
    start: int = Field(default=1, description='First row number (1-based)')
    count: int = Field(default=5, description=f'Rows to return, at most {MAX_MODEL_ROWS}')
    columns: list[str] = Field(default_factory=list, description='Columns to return (default: all)')


class FindArgs(BaseModel):
    sheet: str = Field(description='Sheet id')
    filters: list[Filter] = Field(default_factory=list)
    match: Literal['all', 'any'] = Field(default='all', description='Combine filters with AND (all) or OR (any)')
    columns: list[str] = Field(default_factory=list, description='Columns to return; pick the useful ones')
    sort_by: str = Field(default='', description='Column to sort by')
    descending: bool = False
    limit: int = Field(default=10, description=f'Rows shown to you, at most {MAX_MODEL_ROWS}')


class AggregateArgs(BaseModel):
    sheet: str = Field(description='Sheet id')
    metrics: list[Metric] = Field(default_factory=list, description='Default: row count')
    group_by: list[str] = Field(default_factory=list, description='Up to 3 columns')
    time_grain: Literal['none', 'day', 'week', 'month', 'quarter', 'year'] = Field(
        default='none', description='Bucket date columns in group_by')
    filters: list[Filter] = Field(default_factory=list)
    match: Literal['all', 'any'] = 'all'
    sort_by: str = Field(default='', description='A metric label like sum(Amount), or a group column')
    descending: bool = True
    limit: int = Field(default=20, description=f'Groups shown to you, at most {MAX_MODEL_GROUPS}')


class SqlArgs(BaseModel):
    sql: str = Field(description='One read-only SQLite SELECT statement')


@dataclass
class Step:
    """One executed operation, shown to the user under the answer."""
    title: str
    summary: str
    frame: pd.DataFrame | None = None
    sql: str = ''
    chart: str = ''
    error: bool = False


def cell(value, limit=CELL_CHARS):
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.date().isoformat() if value == value.normalize() else value.isoformat(sep=' ')
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value.is_integer():
            return int(value)
        return round(value, 4) if abs(value) >= 1 else float(f'{value:.4g}')
    text = re.sub(r'\s+', ' ', str(value)).strip()
    return text[:limit] + '…' if len(text) > limit else text


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.map(to_number), errors='coerce').astype(float)


def as_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.where(~missing_mask(series)), errors='coerce', format='mixed')


def text(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.casefold()


def parse_number(value):
    return to_number(value) if str(value).strip() else None


def parse_date(value):
    if not re.search(r'\d{4}', str(value)):
        return None
    parsed = pd.to_datetime(str(value).strip(), errors='coerce')
    return None if pd.isna(parsed) else parsed


def dumps(payload):
    return json.dumps(payload, ensure_ascii=False, default=str, separators=(',', ':'))


def encode(payload, max_chars=MAX_OUTPUT_CHARS):
    """Drop listed rows (never counts) until the output fits the model budget."""
    output = dumps(payload)
    while len(output) > max_chars and payload.get('rows'):
        payload['rows'] = payload['rows'][:len(payload['rows']) // 2]
        payload['shown'] = len(payload['rows'])
        output = dumps(payload)
    return output if len(output) <= max_chars else compact(payload, max_chars)


def records(frame, limit):
    return [[cell(value) for value in row] for row in frame.head(limit).itertuples(index=False, name=None)]


class Toolkit:
    def __init__(self, workbook, sheets=None):
        self.workbook = workbook
        self.ids = list(sheets or workbook.frames)
        self.steps: list[Step] = []

    # Lookup helpers ---------------------------------------------------------------------------
    def sheet(self, name):
        key = str(name).strip()
        normalized = key.casefold().replace(' :: ', ' › ')
        for sid in self.ids:
            if key == sid or normalized in (sid.casefold(), self.workbook.labels[sid].casefold()):
                return sid, self.workbook.frames[sid]
        matches = [sid for sid in self.ids if self.workbook.labels[sid].casefold().endswith('› ' + normalized)]
        if len(matches) == 1 or len(self.ids) == 1:
            sid = (matches or self.ids)[0]
            return sid, self.workbook.frames[sid]
        raise ToolError(f'Unknown sheet {key!r}. Available sheet ids: {", ".join(self.ids)}')

    def column(self, frame, name, sid):
        name = str(name)
        if name in frame.columns:
            return name
        def norm(value):
            return re.sub(r'[^0-9a-z]+', '', str(value).casefold())
        exact = [c for c in frame.columns if norm(c) == norm(name)]
        if len(exact) == 1:
            return exact[0]
        lookup = {norm(c): c for c in frame.columns}
        close = [lookup[n] for n in difflib.get_close_matches(norm(name), list(lookup), n=4, cutoff=0.5)]
        hint = f' Did you mean: {", ".join(map(repr, close))}?' if close else ''
        raise ToolError(f'Column {name!r} is not in {sid}.{hint} Use describe_sheet to list columns.')

    # Filtering ----------------------------------------------------------------------------------
    def condition(self, frame, flt, sid):
        series = frame[self.column(frame, flt.column, sid)]
        op = flt.op
        if op == 'is_missing':
            return missing_mask(series)
        if op == 'not_missing':
            return ~missing_mask(series)
        if op in ('contains', 'not_contains', 'starts_with', 'ends_with'):
            needle = flt.value.strip().casefold()
            values = text(series)
            found = {'starts_with': values.str.startswith(needle), 'ends_with': values.str.endswith(needle)
                     }.get(op, values.str.contains(needle, regex=False))
            found = found.fillna(False).astype(bool)
            return ~found if op == 'not_contains' else found
        if op in ('in', 'not_in', '=', '!='):
            values = flt.values if op in ('in', 'not_in') and flt.values else [flt.value]
            found = self.equals(series, values)
            return ~found if op in ('!=', 'not_in') else found
        if op == 'between':
            if len(flt.values) != 2:
                raise ToolError('between needs values=[low, high].')
            return self.compare(series, '>=', flt.values[0]) & self.compare(series, '<=', flt.values[1])
        return self.compare(series, op, flt.value)

    def equals(self, series, values):
        found = text(series).isin([str(v).strip().casefold() for v in values]).fillna(False).astype(bool)
        numbers = [n for n in map(parse_number, values) if n is not None]
        if numbers and column_kind(series) == 'number':
            found |= numeric(series).isin(numbers)
        dates = [d for d in map(parse_date, values) if d is not None]
        if dates and column_kind(series) == 'date':
            parsed = as_dates(series)
            if all(d == d.normalize() for d in dates):
                parsed = parsed.dt.normalize()
            found |= parsed.isin(dates).fillna(False)
        return found

    def compare(self, series, op, value):
        kind = column_kind(series)
        date = parse_date(value)
        number = parse_number(value)
        if date is not None and (kind == 'date' or (kind == 'text' and as_dates(series).notna().mean() > 0.8)):
            left, right = as_dates(series), date
            if date == date.normalize() and op in ('<=', '>'):
                # A date-only bound covers that whole day: "<= 2024-02-29" includes 29 Feb 18:00.
                op, right = {'<=': '<', '>': '>='}[op], date + pd.Timedelta(days=1)
        elif number is not None:
            left, right = numeric(series), number
        else:
            left, right = text(series), str(value).strip().casefold()
        return pd.Series(COMPARE[op](left, right), index=series.index).fillna(False).astype(bool)

    def filtered(self, frame, filters, match, sid):
        filters = [Filter.model_validate(f) if isinstance(f, dict) else f for f in filters]
        if not filters:
            return frame
        masks = [self.condition(frame, flt, sid) for flt in filters]
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined & mask if match == 'all' else combined | mask
        return frame[combined]

    @staticmethod
    def where(filters, match):
        if not filters:
            return ''
        def one(flt):
            if flt.op in ('is_missing', 'not_missing'):
                return f'{flt.column} {flt.op.replace("_", " ")}'
            value = ', '.join(flt.values) if flt.values else flt.value
            return f'{flt.column} {flt.op.replace("_", " ")} {value}'
        return ' where ' + (' and ' if match == 'all' else ' or ').join(map(one, filters))

    # Tools ------------------------------------------------------------------------------------------
    def describe_sheet(self, sheet, columns=()):
        sid, frame = self.sheet(sheet)
        selected = [self.column(frame, c, sid) for c in columns] or list(frame.columns[:25])
        described = []
        for name in selected:
            series = frame[name]
            missing = missing_mask(series)
            present = series[~missing]
            info = {'name': name, 'type': column_kind(series), 'missing': int(missing.sum())}
            if info['type'] == 'number' and len(present):
                values = numeric(present).dropna()
                if len(values):
                    info.update(min=cell(values.min()), max=cell(values.max()), mean=cell(values.mean()))
                if len(values) < len(present):
                    info['non_numeric_values'] = int(len(present) - len(values))
            elif info['type'] == 'date' and len(present):
                values = as_dates(present).dropna()
                if len(values):
                    info.update(min=cell(values.min()), max=cell(values.max()))
            elif len(present):
                labels = present.astype('string').str.strip()
                counts = labels.value_counts()
                info['distinct'] = int(len(counts))
                if len(counts) == len(present) and len(present) > 5:
                    info['examples'] = [cell(v, 40) for v in counts.index[:3]]
                else:
                    info['top_values'] = {cell(k, 40): int(v) for k, v in counts.head(8).items()}
            described.append(info)
        payload = {'sheet': sid, 'rows': len(frame), 'columns': described,
                   'sample_rows': {'columns': selected, 'rows': records(frame[selected], 3)}}
        rest = [c for c in frame.columns if c not in selected]
        if rest and not columns:
            payload['more_columns'] = rest[:80]
        self.steps.append(Step(f'Inspected the columns of {self.workbook.labels[sid]}',
                               f'{len(selected)} of {len(frame.columns)} columns, {len(frame):,} rows'))
        return compact(payload, MAX_OUTPUT_CHARS)

    def read_rows(self, sheet, start=1, count=5, columns=()):
        sid, frame = self.sheet(sheet)
        start = max(1, int(start))
        count = max(1, min(int(count), MAX_MODEL_ROWS))
        selected = [self.column(frame, c, sid) for c in columns] or list(frame.columns)
        window = frame[selected].iloc[start - 1:start - 1 + count]
        payload = {'sheet': sid, 'total_rows': len(frame), 'first_row': start,
                   'columns': selected, 'rows': records(window, count), 'shown': len(window)}
        self.steps.append(Step(f'Looked at rows {start}–{start + len(window) - 1} of {self.workbook.labels[sid]}',
                               f'{len(window)} of {len(frame):,} rows'))
        return encode(payload)

    def find_rows(self, sheet, filters=(), match='all', columns=(), sort_by='', descending=False, limit=10):
        sid, frame = self.sheet(sheet)
        filters = [Filter.model_validate(f) if isinstance(f, dict) else f for f in filters]
        found = self.filtered(frame, filters, match, sid)
        if sort_by:
            key = self.column(frame, sort_by, sid)
            kind = column_kind(frame[key])
            sorter = numeric if kind == 'number' else as_dates if kind == 'date' else text
            found = found.sort_values(key, ascending=not descending, na_position='last', key=sorter, kind='stable')
        selected = [self.column(frame, c, sid) for c in columns] or list(frame.columns)
        view = found[selected]
        limit = max(1, min(int(limit), MAX_MODEL_ROWS))
        payload = {'sheet': sid, 'rows_scanned': len(frame), 'matched': len(found),
                   'columns': selected, 'rows': records(view, limit), 'shown': min(limit, len(found))}
        if len(found) > payload['shown']:
            payload['note'] = (f'Only {payload["shown"]} of {len(found):,} matching rows are shown to you. '
                               'The user sees all of them in a table under your answer.')
        self.steps.append(Step(f'Searched {self.workbook.labels[sid]}{self.where(filters, match)}',
                               f'{len(found):,} of {len(frame):,} rows matched',
                               view.head(MAX_RESULT_ROWS).reset_index(drop=True)))
        return encode(payload)

    def aggregate(self, sheet, metrics=(), group_by=(), time_grain='none', filters=(), match='all',
                  sort_by='', descending=True, limit=20):
        sid, frame = self.sheet(sheet)
        if len(group_by) > 3:
            raise ToolError('Group by at most 3 columns.')
        metrics = [Metric.model_validate(m) if isinstance(m, dict) else m for m in metrics] or [Metric(op='count')]
        filters = [Filter.model_validate(f) if isinstance(f, dict) else f for f in filters]
        found = self.filtered(frame, list(filters), match, sid)
        work, keys, notes, timed = pd.DataFrame(index=found.index), [], [], False
        for name in group_by:
            key = self.column(frame, name, sid)
            series = found[key]
            if time_grain != 'none' and column_kind(series) in ('date', 'text'):
                dates = as_dates(series)
                if dates.notna().any():
                    periods = dates.dt.to_period(GRAINS[time_grain])
                    series = (periods.dt.start_time.dt.strftime('%Y-%m-%d') if time_grain in ('week', 'day')
                              else periods.astype('string'))
                    timed = True
            work[key] = series
            keys.append(key)
        named = {}
        for index, metric in enumerate(metrics):
            source = f'__m{index}'
            if metric.op == 'count' and not metric.column:
                work[source] = 1
                named[metric.label] = (source, 'sum')
                continue
            column = self.column(frame, metric.column, sid)
            series = found[column]
            if metric.op == 'count':
                work[source] = (~missing_mask(series)).astype(int)
                named[metric.label] = (source, 'sum')
            elif metric.op == 'count_distinct':
                work[source] = series.where(~missing_mask(series))
                named[metric.label] = (source, 'nunique')
            elif metric.op in ('min', 'max') and column_kind(series) == 'date':
                work[source] = as_dates(series)
                named[metric.label] = (source, metric.op)
            else:
                values = numeric(series)
                present = int((~missing_mask(series)).sum())
                if present and values.notna().sum() == 0:
                    raise ToolError(f'{column!r} has no numeric values; use count or count_distinct.')
                if values.notna().sum() < present:
                    notes.append(f'{present - int(values.notna().sum())} non-numeric values in {column} were ignored.')
                work[source] = values
                named[metric.label] = (source, metric.op)
        if keys:
            result = work.groupby(keys, dropna=False, sort=False).agg(**named).reset_index()
            for key in keys:
                result[key] = result[key].astype('object').where(result[key].notna(), '(missing)')
        else:
            result = pd.DataFrame([{label: getattr(work[source], how)() if len(work) else None
                                    for label, (source, how) in named.items()}])
        labels = list(named)
        if sort_by:
            target = next((label for label, metric in zip(labels, metrics)
                           if sort_by.casefold() in (label.casefold(), metric.column.casefold())), None)
            target = target or self.column(result, sort_by, sid)
            result = result.sort_values(target, ascending=not descending, na_position='last', kind='stable')
        elif timed:
            result = result.sort_values(keys[0], kind='stable')
        elif keys:
            result = result.sort_values(labels[0], ascending=False, na_position='last', kind='stable')
        result = result.reset_index(drop=True)
        limit = max(1, min(int(limit), MAX_MODEL_GROUPS))
        payload = {'sheet': sid, 'rows_scanned': len(frame), 'rows_matched': len(found)}
        if keys:
            payload.update(groups=len(result), columns=list(result.columns), rows=records(result, limit),
                           shown=min(limit, len(result)))
            if len(result) > limit:
                payload['note'] = f'Only {limit} of {len(result):,} groups are shown to you; the user sees all of them.'
        else:
            payload['result'] = {label: cell(value) for label, value in result.iloc[0].items()}
        if notes:
            payload['warnings'] = notes
        by = f' by {", ".join(keys)}' + (f' ({time_grain})' if timed else '') if keys else ''
        chart = ('line' if timed else 'bar') if keys and 1 < len(result) <= 60 else ''
        self.steps.append(Step(f'Calculated {", ".join(labels)}{by} in {self.workbook.labels[sid]}'
                               f'{self.where(filters, match)}',
                               f'{len(found):,} of {len(frame):,} rows used' + (f', {len(result):,} groups' if keys else ''),
                               result.head(MAX_RESULT_ROWS), chart=chart))
        return encode(payload)

    def sql_query(self, sql):
        result = self.workbook.store.query(sql, set(self.ids))
        frame = result.frame
        payload = {'rows_returned': len(frame), 'columns': list(frame.columns),
                   'rows': records(frame, MAX_MODEL_ROWS), 'shown': min(MAX_MODEL_ROWS, len(frame))}
        if len(frame) > MAX_MODEL_ROWS:
            payload['note'] = f'Only {MAX_MODEL_ROWS} of {len(frame):,} rows are shown to you; the user sees all of them.'
        if result.truncated:
            payload['warning'] = f'Result was cut at {len(frame):,} rows; aggregate in SQL instead of listing.'
        self.steps.append(Step('Ran a SQL query over ' + ', '.join(self.workbook.labels[s] for s in result.sources),
                               f'{len(frame):,} result rows' + (' (cut at the display limit)' if result.truncated else ''),
                               frame, sql=sql))
        return encode(payload)

    # LangChain wrappers ----------------------------------------------------------------------------
    def guarded(self, function, title):
        def run(**kwargs):
            try:
                return function(**kwargs)
            except (ToolError, QueryExecutionError) as exc:
                message = str(exc)
            except Exception as exc:  # a failed operation is reported to the model, not raised
                message = f'{type(exc).__name__}: {str(exc)[:200]}'
            self.steps.append(Step(f'{title} failed', message, error=True))
            return dumps({'error': message})
        return run

    def tools(self):
        specs = [
            ('describe_sheet', self.describe_sheet, DescribeArgs,
             'Column types, missing counts, min/max/mean, most common values and 3 sample rows. '
             'Use it to learn exact category spellings and formats before filtering.'),
            ('read_rows', self.read_rows, ReadArgs,
             f'Peek at up to {MAX_MODEL_ROWS} rows by position to understand the layout. Never page through a sheet.'),
            ('find_rows', self.find_rows, FindArgs,
             'Filter every row of a sheet and list matches. Returns the exact match count and a few rows; the user '
             'gets the full list. Text matching ignores case. Use sort_by + descending + limit for top/bottom N.'),
            ('aggregate', self.aggregate, AggregateArgs,
             'count / count_distinct / sum / mean / median / min / max / std over every row, optionally filtered and '
             'grouped (time_grain buckets dates). Use for totals, averages, breakdowns, rankings and distributions.'),
            ('sql_query', self.sql_query, SqlArgs,
             'Read-only SQLite SELECT over all sheets (table = sheet id; double-quote column names). Use only for '
             'joins, calculated columns, date maths, HAVING or window functions. Dates are text YYYY-MM-DD. '
             'to_number(x) parses numeric text; is_missing(x)=1 for blank/NA cells.'),
        ]
        return [StructuredTool.from_function(func=self.guarded(function, name.replace('_', ' ')), name=name,
                                             description=description, args_schema=schema)
                for name, function, schema, description in specs]
