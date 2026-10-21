import asyncio
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from services.analysis import ContextBudgetError
from services.query_planner import QueryPlanError, parse_plan, run_query
from services.table_store import MAX_RESULT_ROWS, QueryExecutionError, TableStore


class Model:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    async def ainvoke(self, messages):
        self.calls.append(json.loads(json.dumps(messages)))
        response = self.responses.pop(0) if self.responses else ''
        return AIMessage(content=response if isinstance(response, str) else json.dumps(response))
    def bind_tools(self, *args, **kwargs):
        raise AssertionError('The Q&A path must not depend on native tool calls.')


class TableQueryTests(unittest.TestCase):
    def setUp(self):
        self.tables = {'records.xlsx :: Data': pd.DataFrame({
            'Label': ['One', 'Two', 'Three', 'Four'], 'Field': [None, '  ', 'provided', 'N/A'],
            'Category': ['X', 'Y', 'X', 'Y'], 'Amount': [10, 20, 30, 40]})}
        self.store = TableStore(self.tables)
    def tearDown(self):
        self.store.close()

    def test_missing_values_for_an_arbitrary_field(self):
        result = self.store.query('SELECT C0 AS Label FROM T0 WHERE is_missing(C1)=1')
        self.assertEqual(result.frame['Label'].tolist(), ['One', 'Two', 'Four'])

    def test_compound_filters_aggregate_and_group(self):
        result = self.store.query("SELECT C2 AS Category, SUM(C3) AS Total FROM T0 "
                                  "WHERE C3 >= 20 AND C2='Y' GROUP BY C2")
        self.assertEqual(result.frame.iloc[0]['Total'], 60)

    def test_joins_and_window_queries(self):
        self.store.close()
        other = dict(self.tables, **{'lookup.csv': pd.DataFrame({'Key': ['X', 'Y'], 'Description': ['East', 'West']})})
        self.store = TableStore(other)
        result = self.store.query('SELECT a.C0 AS Label, b.C1 AS Region, '
            'RANK() OVER (ORDER BY a.C3 DESC) AS Position FROM T0 a JOIN T1 b ON a.C2=b.C0 ORDER BY a.C3 DESC')
        self.assertEqual(result.frame.iloc[0].tolist(), ['Four', 'West', 1])

    def test_writes_external_access_and_multiple_statements_are_rejected(self):
        for sql in ['DELETE FROM T0', 'DROP TABLE T0', 'UPDATE T0 SET C3=0',
                    "ATTACH DATABASE ':memory:' AS other", 'PRAGMA database_list',
                    "SELECT load_extension('anything') FROM T0", 'SELECT * FROM sqlite_master',
                    'SELECT * FROM T0; DELETE FROM T0']:
            with self.subTest(sql=sql), self.assertRaises(QueryExecutionError):
                self.store.query(sql)
        self.assertEqual(self.store.query('SELECT COUNT(*) AS n FROM T0').frame.iloc[0]['n'], 4)

    def test_dataset_scope_is_enforced_even_for_same_query(self):
        self.store.query('SELECT * FROM T0')
        with self.assertRaises(QueryExecutionError):
            self.store.query('SELECT * FROM T0', allowed_tables=set())

    def test_no_invented_constant_answer_without_reading_data(self):
        with self.assertRaises(QueryExecutionError):
            self.store.query('SELECT 42 AS invented')

    def test_all_10000_rows_are_used_for_aggregations(self):
        self.store.close()
        self.store = TableStore({'large.csv': pd.DataFrame({'Amount': range(1, 10001)})})
        result = self.store.query('SELECT SUM(C0) AS Total FROM T0')
        self.assertEqual(result.frame.iloc[0]['Total'], 50005000)

    def test_large_list_results_are_explicitly_partial(self):
        self.store.close()
        self.store = TableStore({'large.csv': pd.DataFrame({'Value': range(MAX_RESULT_ROWS + 2)})})
        result = self.store.query('SELECT C0 FROM T0')
        self.assertEqual(len(result.frame), MAX_RESULT_ROWS)
        self.assertTrue(result.truncated)

    def test_expensive_queries_have_a_server_deadline(self):
        with self.assertRaisesRegex(QueryExecutionError, 'timed out'):
            self.store.query('WITH RECURSIVE x(n) AS (SELECT C3 FROM T0 UNION ALL SELECT n+1 FROM x) '
                             'SELECT SUM(n) FROM x', timeout=0.01)


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_is_dynamic_and_row_contents_do_not_enter_model_context(self):
        field = 'field_' + uuid4().hex
        tables = {'arbitrary.csv': pd.DataFrame({field: [None, 'private-cell-value']})}
        model = Model([{'action': 'sql', 'sql': 'SELECT COUNT(*) AS Missing FROM T0 WHERE is_missing(C0)=1'}])
        result = await run_query(model, tables, [{'role': 'user', 'content': f'How many records are missing {field}?'}])
        sent = json.dumps(model.calls)
        self.assertIn(field, sent)
        self.assertNotIn('private-cell-value', sent)
        self.assertEqual(result.results['Query results'].iloc[0]['Missing'], 1)
        self.assertEqual(len(model.calls), 2)

    async def test_invalid_query_is_repaired_once(self):
        tables = {'any.csv': pd.DataFrame({'Any field': [1, 2]})}
        model = Model([{'action': 'sql', 'sql': 'SELECT unknown FROM T0'},
                       {'action': 'sql', 'sql': 'SELECT SUM(C0) AS Total FROM T0'}])
        result = await run_query(model, tables, [{'role': 'user', 'content': 'What is the total?'}])
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(result.results['Query results'].iloc[0]['Total'], 3)

    async def test_repeated_invalid_plans_stop_without_claiming_a_result(self):
        model = Model(['not json', {'action': 'execute_python', 'code': 'anything'}])
        with self.assertRaises(QueryPlanError):
            await run_query(model, {'any.csv': pd.DataFrame({'x': [1]})}, [{'role': 'user', 'content': 'Count records.'}])
        self.assertEqual(len(model.calls), 2)

    async def test_schema_context_limit_is_checked_before_provider_call(self):
        model = Model([])
        with self.assertRaises(ContextBudgetError):
            await run_query(model, {'any.csv': pd.DataFrame({'x': [1]})},
                            [{'role': 'user', 'content': 'Count records.'}], input_budget=100)
        self.assertFalse(model.calls)

    async def test_ambiguity_returns_a_model_clarification(self):
        model = Model([{'action': 'clarify', 'clarification': 'Which of the two date columns do you mean?'}])
        result = await run_query(model, {'any.csv': pd.DataFrame({'Start': [1], 'End': [2]})},
                                 [{'role': 'user', 'content': 'What is the latest date?'}])
        self.assertIn('Which of the two date columns', result.answer)
        self.assertFalse(result.results)

    async def test_questions_about_dataset_structure_use_actual_headers(self):
        model = Model([{'action': 'schema'}])
        result = await run_query(model, {'data.csv': pd.DataFrame({'Completely new field': [1], 'Other field': ['x']})},
                                 [{'role': 'user', 'content': 'What columns does this file contain?'}])
        self.assertEqual(result.results['Dataset structure']['Column'].tolist(), ['Completely new field', 'Other field'])

    async def test_text_questions_still_use_indexed_search(self):
        model = Model([{'action': 'search', 'table': 'T0', 'keywords': 'cancellation'}])
        tables = {'guide.csv': pd.DataFrame({'Paragraph': ['Ordinary text'] * 999 + ['Cancellation notice is 14 days.']})}
        result = await run_query(model, tables, [{'role': 'user', 'content': 'What is the cancellation period?'}])
        self.assertIn('14 days', result.answer)
        self.assertEqual(result.results['Source excerpts'].iloc[0]['row'], 1000)

    async def test_deadline_cancels_slow_planning(self):
        class SlowModel:
            async def ainvoke(self, messages):
                await asyncio.sleep(10)
        with self.assertRaises(TimeoutError):
            await run_query(SlowModel(), {'data.csv': pd.DataFrame({'x': [1]})},
                            [{'role': 'user', 'content': 'Count.'}], timeout=0.01)

    async def test_only_selected_schema_is_exposed_and_aliases_stay_consistent(self):
        tables = {'first.csv': pd.DataFrame({'x': [1]}), 'second.csv': pd.DataFrame({'other': [2]})}
        model = Model([{'action': 'sql', 'sql': 'SELECT SUM(C0) AS Total FROM T1'}])
        result = await run_query(model, tables, [{'role': 'user', 'content': 'Total?'}], table_names=['second.csv'])
        self.assertNotIn('first.csv', model.calls[0][0]['content'])
        self.assertEqual(result.results['Query results'].iloc[0]['Total'], 2)


class QueryUIRegressionTests(unittest.TestCase):
    def test_ui_uses_the_same_general_query_flow_for_arbitrary_columns(self):
        upload = io.BytesIO(b'Asset,Inspection\nMachine-A,\nMachine-B,Complete\n')
        upload.name, upload.size, upload.file_id = 'assets.csv', len(upload.getvalue()), 'query-test'
        model = Model([{'action': 'sql', 'sql': 'SELECT C0 AS Asset FROM T0 WHERE is_missing(C1)=1'},
                       'Machine-A has no inspection recorded in assets.csv.'])
        with (patch('streamlit.file_uploader', return_value=[upload]),
              patch('agent.factory.build_model', return_value=model)):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            app.text_input[1].set_value('test-key').run()
            app.chat_input[0].set_value('Which assets have not been inspected?').run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertIn('Machine-A', app.session_state['messages'][-1]['content'])
            self.assertEqual(len(model.calls), 2)
            self.assertIn('Query results', app.session_state['query_results'])

    def test_unknown_plan_fields_and_python_are_rejected(self):
        for response in [{'action': 'sql', 'sql': 'SELECT * FROM T0', 'python': 'evil'},
                         {'action': 'sql', 'sql': 'SELECT * FROM T0', 'keywords': 'ignored filter'}]:
            with self.assertRaises(QueryPlanError):
                parse_plan(json.dumps(response))
