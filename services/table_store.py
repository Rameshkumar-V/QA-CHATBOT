"""A session-local SQLite store that can only read uploaded tables."""
from dataclasses import dataclass
import json
import sqlite3
import threading
import time

import pandas as pd

from services.data_tools import MISSING_MARKERS

MAX_RESULT_ROWS = 10_000
MAX_RESULT_BYTES = 10 * 1024 * 1024


class QueryExecutionError(ValueError):
    pass


@dataclass
class TableResult:
    frame: pd.DataFrame
    sources: list[str]
    truncated: bool


class TableStore:
    def __init__(self, tables):
        self.tables = tables
        self.connection = None
        self.lock = threading.RLock()
        self.catalog = {f'T{i}': {'name': name, 'rows': len(frame), 'columns': [
            {'id': f'C{j}', 'name': str(column), 'type': str(frame.iloc[:, j].dtype)}
            for j, column in enumerate(frame.columns)]} for i, (name, frame) in enumerate(tables.items())}
        for alias, frame in zip(self.catalog, tables.values()):
            if 'header_row' in frame.attrs:
                self.catalog[alias]['header'] = {key: frame.attrs[key] for key in
                    ('header_row', 'header_rows', 'header_mode', 'generic_headers') if key in frame.attrs}

    def close(self):
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def _build(self):
        if self.connection is not None:
            return
        connection = sqlite3.connect(':memory:', check_same_thread=False, cached_statements=0)
        try:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2 * 1024 * 1024)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 20_000)
            for alias, (_, frame) in zip(self.catalog, self.tables.items()):
                local = frame.copy()
                local.columns = [c['id'] for c in self.catalog[alias]['columns']]
                for column in local.columns:
                    if local[column].dtype == 'object':
                        local[column] = local[column].map(lambda value: json.dumps(value, default=str)
                            if isinstance(value, (dict, list, tuple)) else value)
                local.insert(0, '__row_number', range(1, len(local) + 1))
                local.to_sql(alias, connection, index=False)
            def is_missing(value):
                return int(value is None or str(value).strip().casefold() in MISSING_MARKERS)
            def to_number(value):
                try:
                    return float(value) if not is_missing(value) else None
                except (TypeError, ValueError):
                    return None
            connection.create_function('is_missing', 1, is_missing, deterministic=True)
            connection.create_function('to_number', 1, to_number, deterministic=True)
            connection.execute('PRAGMA query_only = ON')
            connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 50)
            connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
            self.connection = connection
        except Exception:
            connection.close()
            raise

    def query(self, sql, allowed_tables=None, timeout=3):
        allowed = set(self.catalog) if allowed_tables is None else set(allowed_tables)
        if not allowed.issubset(self.catalog):
            raise QueryExecutionError('The selected dataset scope is invalid.')
        used = set()
        def authorize(action, arg1, arg2, database, trigger):
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 in allowed:
                used.add(arg1)
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and str(arg2).casefold() not in ('load_extension', 'readfile', 'writefile'):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        with self.lock:
            self._build()
            deadline = time.monotonic() + timeout
            self.connection.set_authorizer(authorize)
            self.connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            cursor = None
            try:
                cursor = self.connection.execute(sql)
                if not used:
                    raise QueryExecutionError('A query must read at least one uploaded table.')
                records, size, truncated = [], 0, False
                columns = [col[0] for col in cursor.description]
                if len(set(columns)) != len(columns):
                    raise QueryExecutionError('Use unique aliases for result columns.')
                for row in cursor:
                    size += sum(len(str(value).encode('utf-8')) for value in row)
                    if len(records) >= MAX_RESULT_ROWS or size > MAX_RESULT_BYTES:
                        truncated = True
                        break
                    records.append(row)
                return TableResult(pd.DataFrame.from_records(records, columns=columns),
                                   [self.catalog[alias]['name'] for alias in self.catalog if alias in used], truncated)
            except sqlite3.Error as exc:
                if time.monotonic() >= deadline:
                    raise QueryExecutionError('Query execution timed out. Narrow the request or simplify the calculation.') from exc
                # These are validation diagnostics; no provider response or row contents are included.
                if isinstance(exc, sqlite3.ProgrammingError):
                    message = 'Use one read-only SQL statement.'
                elif 'authoriz' in str(exc).lower() or 'prohibited' in str(exc).lower():
                    message = 'Only read-only queries against the selected uploaded tables are allowed.'
                else:
                    message = 'SQL could not be validated. Check SQLite syntax, table IDs, column IDs and functions.'
                raise QueryExecutionError(message) from exc
            finally:
                if cursor is not None:
                    cursor.close()
                self.connection.set_authorizer(None)
                self.connection.set_progress_handler(None, 0)
