"""Session-local full-text index. Raw SQL is never exposed to the model."""
import re
import sqlite3
import threading


class DocumentSearch:
    """Build an FTS index once on first use, then query passages instead of scanning rows."""
    def __init__(self, tables):
        self.tables = tables
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
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        try:
            connection.execute('CREATE VIRTUAL TABLE passages USING fts5('
                               'table_name UNINDEXED, row_number UNINDEXED, column_name UNINDEXED, '
                               'page UNINDEXED, chunk_number UNINDEXED, content, tokenize="unicode61")')
            def records():
                for table, frame in self.tables.items():
                    for row_number, values in enumerate(frame.itertuples(index=False, name=None), 1):
                        page = str(values[frame.columns.get_loc('page')]) if 'page' in frame else ''
                        for column, value in zip(frame.columns, values):
                            if value is None:
                                continue
                            text = str(value)
                            # Overlapping passages preserve matches near chunk boundaries.
                            for chunk, start in enumerate(range(0, len(text), 700), 1):
                                yield (table, row_number, str(column), page, chunk, text[start:start + 800])
            connection.executemany('INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)', records())
            connection.commit()
            self.connection = connection
        except Exception:
            connection.close()
            raise

    def search(self, table, query, limit=3, offset=0):
        if table not in self.tables:
            raise ValueError('Unknown table; call list_tables first.')
        if len(query) > 200:
            raise ValueError('Use up to 200 characters of search keywords.')
        terms = list(dict.fromkeys(re.findall(r'\w+', query, flags=re.UNICODE)))[:12]
        if not terms:
            raise ValueError('Provide one or more search keywords.')
        # Quote each token, so model-supplied MATCH syntax cannot become operators.
        match = ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)
        limit = max(1, min(int(limit), 3))
        offset = max(0, int(offset))
        with self.lock:
            self._build()
            rows = self.connection.execute(
                "SELECT row_number, column_name, page, chunk_number, "
                "snippet(passages, 5, '', '', ' … ', 36) "
                "FROM passages WHERE passages MATCH ? AND table_name = ? "
                "ORDER BY bm25(passages), row_number, column_name, chunk_number LIMIT ? OFFSET ?",
                (match, table, limit + 1, offset)).fetchall()
        return {'table': table, 'query': query, 'method': 'indexed keyword search',
                'has_more': len(rows) > limit, 'next_offset': offset + limit if len(rows) > limit else None,
                'passages': [{'row': row, 'column': column, 'page': page or None, 'chunk': chunk,
                              'excerpt': excerpt} for row, column, page, chunk, excerpt in rows[:limit]]}
