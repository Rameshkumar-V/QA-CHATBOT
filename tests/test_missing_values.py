"""Regression for a 43-row, 9-column student sheet with absent Aadhaar values."""
import json
import unittest

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.factory import create_data_agent
from services.analysis import analyze
from services.data_tools import MAX_TOOL_CHARS, make_tools


TABLE = 'CA-12.10.xlsx :: Sheet1'


def student_tables():
    absent = [None, '', '   ', '\t', 'N/A', 'na', 'NULL', 'not provided',
              'Not Submitted', '-', '--', 'not available']
    frame = pd.DataFrame({
        'Student Name': [f'Student {n}' for n in range(1, 44)],
        'Roll Number': range(1, 44),
        'Adhaar Number': absent + [0] + ['123456789012'] * 30,
        **{f'Other {n}': ['extra data'] * 43 for n in range(6)},
    })
    return {TABLE: frame}


class MissingValueTests(unittest.TestCase):
    def setUp(self):
        self.tools = {t.name: t for t in make_tools(student_tables())}

    def test_finds_missing_values_with_only_student_identifiers(self):
        raw = self.tools['filter_rows'].invoke({'table': TABLE, 'column': 'Adhaar Number',
              'operator': 'is_missing', 'columns': ['Student Name', 'Roll Number']})
        result = json.loads(raw)
        self.assertEqual(result['rows_scanned'], 43)
        self.assertEqual(result['matches'], 12)
        self.assertEqual(len(result['rows']), 12)
        self.assertEqual(result['rows'][-1]['Roll Number'], 12)
        self.assertNotIn('Adhaar Number', result['rows'][0])
        self.assertLessEqual(len(raw), MAX_TOOL_CHARS)

    def test_complement_includes_zero_and_provided_values(self):
        result = json.loads(self.tools['filter_rows'].invoke({
            'table': TABLE, 'column': 'Adhaar Number', 'operator': 'is_not_missing',
            'columns': ['Roll Number']}))
        self.assertEqual(result['matches'], 31)
        self.assertEqual(result['rows'][0]['Roll Number'], 13)

    def test_schema_retains_all_nine_columns_and_counts_whitespace_as_missing(self):
        result = json.loads(self.tools['describe_table'].invoke({'table': TABLE}))
        self.assertEqual(len(result['columns']), 9)
        aadhaar = next(c for c in result['columns'] if c['name'] == 'Adhaar Number')
        self.assertEqual(aadhaar['missing'], 12)
        self.assertNotIn('sample', result)

    def test_count_scans_10000_rows_even_when_only_20_are_returned(self):
        tools = {t.name: t for t in make_tools({'large': pd.DataFrame({'Name': range(10000),
                                                                   'Adhaar': [None] * 10000})})}
        result = json.loads(tools['filter_rows'].invoke({'table': 'large', 'column': 'Adhaar',
                                                       'operator': 'is_missing', 'limit': 20,
                                                       'columns': ['Name']}))
        self.assertEqual(result['matches'], 10000)
        self.assertEqual(len(result['rows']), 20)

    def test_nulls_in_numeric_and_nullable_string_columns(self):
        for values in ([1.0, float('nan'), 2.0], pd.Series(['123', pd.NA, '456'], dtype='string')):
            tools = {t.name: t for t in make_tools({'test': pd.DataFrame({'id': values})})}
            result = json.loads(tools['filter_rows'].invoke({'table': 'test', 'column': 'id',
                                                           'operator': 'is_missing'}))
            self.assertEqual(result['matches'], 1)
            self.assertIsNone(result['rows'][0]['id'])


class MissingAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_student_query_finishes_in_four_model_calls(self):
        class StudentQueryModel(BaseChatModel):
            calls: int = 0
            @property
            def _llm_type(self):
                return 'test-student-query'
            def bind_tools(self, tools, **kwargs):
                return self
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.calls += 1
                last = messages[-1]
                if isinstance(last, ToolMessage) and last.name == 'filter_rows':
                    result = json.loads(last.content)
                    output = AIMessage(f"{result['matches']} students have missing Aadhaar values. "
                                       + ', '.join(r['Student Name'] for r in result['rows']))
                else:
                    if not isinstance(last, ToolMessage):
                        name, args = 'list_tables', {}
                    elif last.name == 'list_tables':
                        name, args = 'describe_table', {'table': TABLE}
                    else:
                        schema = json.loads(last.content)
                        assert any(c['name'] == 'Adhaar Number' for c in schema['columns'])
                        name, args = 'filter_rows', {'table': TABLE, 'column': 'Adhaar Number',
                            'operator': 'is_missing', 'columns': ['Student Name', 'Roll Number']}
                    output = AIMessage('', tool_calls=[{'name': name, 'args': args, 'id': str(self.calls)}])
                return ChatResult(generations=[ChatGeneration(message=output)])
        model = StudentQueryModel()
        answer = await analyze(create_data_agent(model, student_tables()), [
            {'role': 'user', 'content': 'Is there any student not give there Adhaar Number ?'}])
        self.assertIn('12 students', answer)
        self.assertIn('Student 12', answer)
        self.assertEqual(model.calls, 4)

    async def test_fifth_model_call_is_reserved_for_answer_instead_of_another_tool(self):
        class RepeatingModel(BaseChatModel):
            calls: int = 0
            choice: str | None = None
            @property
            def _llm_type(self):
                return 'test-final-step'
            def bind_tools(self, tools, **kwargs):
                self.choice = kwargs.get('tool_choice')
                return self
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.calls += 1
                if self.choice == 'none':
                    output = AIMessage('The sheet contains 43 rows. Missing Aadhaar counts have not been verified.')
                else:
                    output = AIMessage('', tool_calls=[{'name': 'list_tables', 'args': {}, 'id': str(self.calls)}])
                return ChatResult(generations=[ChatGeneration(message=output)])
        model = RepeatingModel()
        answer = await analyze(create_data_agent(model, student_tables()), [{'role': 'user', 'content': 'Check Aadhaar.'}])
        self.assertEqual(model.calls, 5)
        self.assertIn('not been verified', answer)
