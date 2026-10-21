import io
import json
from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from services.context import bounded_chat
from services.data_tools import MAX_TOOL_CHARS, compact, make_tools, read_upload


class Upload(io.BytesIO):
    name = 'sales.xlsx'


class LargeExcelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.DataFrame({'product': ['A', 'B'] * 5000,
                                  'sales': range(1, 10001), 'id': range(10000)})
        buffer = Upload()
        cls.frame.to_excel(buffer, index=False)
        cls.tables = read_upload(buffer)
        cls.name = 'sales.xlsx :: Sheet1'
        cls.tools = {t.name: t for t in make_tools(cls.tables)}

    def call(self, name, **kwargs):
        raw = self.tools[name].invoke(kwargs)
        self.assertLessEqual(len(raw), MAX_TOOL_CHARS)
        return json.loads(raw)

    def test_excel_keeps_all_rows(self):
        self.assertEqual(len(self.tables[self.name]), 10000)

    def test_total_uses_all_rows(self):
        result = self.call('aggregate', table=self.name, column='sales', operation='sum')
        self.assertEqual(result['result'], 50005000)
        self.assertEqual(result['rows_scanned'], 10000)

    def test_grouped_totals(self):
        result = self.call('aggregate', table=self.name, column='sales', operation='sum', group_by='product')
        self.assertEqual({g['group']: g['result'] for g in result['groups']},
                         {'A': 25000000, 'B': 25005000})

    def test_filter_finds_last_row(self):
        result = self.call('filter_rows', table=self.name, column='sales', operator='>', value='9990')
        self.assertEqual(result['matches'], 10)
        self.assertEqual(result['rows'][-1]['sales'], 10000)

    def test_preview_is_bounded(self):
        result = self.call('preview_rows', table=self.name, count=10000)
        self.assertEqual(len(result['rows']), 20)

    def test_group_omissions_explicit(self):
        result = self.call('aggregate', table=self.name, column='sales', operation='sum', group_by='id')
        self.assertEqual(result['total_groups'], 10000)
        self.assertEqual(len(result['groups']), 20)

    def test_wide_schema_can_be_paged(self):
        tools = {t.name: t for t in make_tools({'wide': pd.DataFrame({f'c{i}': [i] for i in range(100)})})}
        result = json.loads(tools['describe_table'].invoke({'table': 'wide', 'start': 12}))
        self.assertEqual(result['columns'][0]['name'], 'c12')
        self.assertEqual(result['next_start'], 24)

    def test_oversized_response_stays_valid_json(self):
        result = compact({'rows': [{'text': 'x' * 10000} for _ in range(1000)], 'matches': 1000})
        self.assertLessEqual(len(result), MAX_TOOL_CHARS)
        self.assertTrue(json.loads(result)['truncated'])
        self.assertEqual(json.loads(result)['matches'], 1000)

    def test_history_budget(self):
        messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'x' * 10000},
                    {'role': 'user', 'content': 'current'}]
        self.assertEqual(bounded_chat(messages), messages[-1:])

    def test_app_starts_and_settings_are_session_scoped(self):
        first = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
        self.assertEqual(len(first.exception), 0)
        first.selectbox[0].select('OpenAI').run()
        first.button[0].click().run()
        self.assertEqual(first.session_state['settings']['provider'], 'OpenAI')
        second = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
        self.assertEqual(len(second.exception), 0)
        self.assertEqual(second.selectbox[0].value, 'Gemini')


if __name__ == '__main__':
    unittest.main()
