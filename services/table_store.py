"""A session-local SQLite copy of the sheets that can only be read."""
from dataclasses import dataclass
import json
import re
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


def to_number(value):
    if value is None or isinstance(value, (int, float)):
        return value
    # Spaces inside digits are kept: "8235 9649 8073" is an identifier, not a number.
    text = re.sub(r'[,₹$€£%]', '', str(value)).strip()
    if text.casefold() in MISSING_MARKERS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


class TableStore:
    """Tables are named by sheet ID and keep their original column names."""
    def __init__(self, frames):
        self.frames = frames
        self.connection = None
        self.lock = threading.RLock()

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
            for name, frame in self.frames.items():
                local = frame.copy()
                for column in local.columns:
                    if local[column].dtype == 'object':
                        local[column] = local[column].map(lambda value: json.dumps(value, default=str)
                            if isinstance(value, (dict, list, tuple)) else value)
                    elif pd.api.types.is_datetime64_any_dtype(local[column]):
                        local[column] = local[column].dt.strftime('%Y-%m-%d %H:%M:%S').str.replace(' 00:00:00', '')
                local.to_sql(name, connection, index=False)
            def is_missing(value):
                return int(value is None or str(value).strip().casefold() in MISSING_MARKERS)
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
        allowed = set(self.frames) if allowed_tables is None else set(allowed_tables)
        if not allowed.issubset(self.frames):
            raise QueryExecutionError('The selected sheet scope is invalid.')
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
                    raise QueryExecutionError('A query must read at least one sheet.')
                records, size, truncated = [], 0, False
                columns = [col[0] for col in cursor.description]
                if len({c.casefold() for c in columns}) != len(columns):
                    raise QueryExecutionError('Use unique aliases for result columns.')
                for row in cursor:
                    size += sum(len(str(value).encode('utf-8')) for value in row)
                    if len(records) >= MAX_RESULT_ROWS or size > MAX_RESULT_BYTES:
                        truncated = True
                        break
                    records.append(row)
                return TableResult(pd.DataFrame.from_records(records, columns=columns),
                                   [name for name in self.frames if name in used], truncated)
            except sqlite3.Error as exc:
                if time.monotonic() >= deadline:
                    raise QueryExecutionError('Query timed out. Narrow the request or simplify the calculation.') from exc
                if isinstance(exc, sqlite3.ProgrammingError):
                    message = 'Use one read-only SELECT statement.'
                elif 'authoriz' in str(exc).lower() or 'prohibited' in str(exc).lower():
                    message = 'Only read-only queries against the loaded sheets are allowed.'
                else:
                    # SQLite diagnostics name tables/columns, never row contents.
                    message = f'SQL error: {exc}. Use sheet IDs as table names and double-quote column names.'
                raise QueryExecutionError(message) from exc
            finally:
                if cursor is not None:
                    cursor.close()
                self.connection.set_authorizer(None)
                self.connection.set_progress_handler(None, 0)
