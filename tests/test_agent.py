import asyncio
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from streamlit.testing.v1 import AppTest

from agent.context_budget import fit_request, request_size
from agent.factory import DEFAULT_MODEL, build_model
from agent.qa_agent import answer_question, create_qa_agent
from services.analysis import ContextBudgetError, MAX_MODEL_CALLS, friendly_error
from services.dataframe_tools import Toolkit
from services.workbook import Workbook

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


class ScriptedModel(BaseChatModel):
    """Plays back tool calls, then answers from the last tool result."""
    script: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    tool_choices: list = Field(default_factory=list)
    choice: object = None

    @property
    def _llm_type(self):
        return 'scripted'

    def bind_tools(self, tools, **kwargs):
        self.choice = kwargs.get('tool_choice')
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        self.tool_choices.append(self.choice)
        if self.script and self.choice != 'none':
            name, args = self.script.pop(0)
            output = AIMessage('', tool_calls=[{'name': name, 'args': args, 'id': f'call{len(self.seen)}'}])
        else:
            result = json.loads(messages[-1].content) if isinstance(messages[-1], ToolMessage) else {}
            output = AIMessage(f'Answer from data: {json.dumps(result)}')
        return ChatResult(generations=[ChatGeneration(message=output)])


def student_workbook():
    absent = [None, '', '   ', 'N/A', 'na', 'NULL', 'not provided', '-']
    return Workbook({'class.xlsx :: Sheet1': pd.DataFrame({
        'Student Name': [f'Student {n}' for n in range(1, 1201)],
        'Roll Number': range(1, 1201),
        'Aadhaar Number': absent + [0] + ['123456789012'] * 1191})})


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_answers_large_sheet_with_queries_not_full_reads(self):
        workbook = student_workbook()
        model = ScriptedModel(script=[
            ('describe_sheet', {'sheet': 'class_sheet1', 'columns': ['Aadhaar Number']}),
            ('find_rows', {'sheet': 'class_sheet1', 'columns': ['Student Name', 'Roll Number'],
                           'filters': [{'column': 'Aadhaar Number', 'op': 'is_missing'}]})])
        progress = []
        answer = await answer_question(model, workbook, [{'role': 'user', 'content': 'Who has no Aadhaar?'}],
                                       progress.append)
        self.assertIn('"matched": 8', answer.text)
        self.assertEqual(len(model.seen), 3)
        sent = ''.join(str(m.content) for call in model.seen for m in call)
        self.assertNotIn('Student 1100', sent)  # rows beyond the bounded results never reach the model
        self.assertIn('1,200 rows', sent)  # the overview is in the system prompt
        self.assertEqual(len(answer.steps[-1].frame), 8)
        self.assertTrue(any('8 of 1,200 rows matched' in line for line in progress))
        workbook.close()

    async def test_last_call_is_forced_to_answer(self):
        workbook = student_workbook()
        model = ScriptedModel(script=[('read_rows', {'sheet': 'class_sheet1'})] * 20)
        answer = await answer_question(model, workbook, [{'role': 'user', 'content': 'Read everything'}])
        self.assertEqual(len(model.seen), MAX_MODEL_CALLS)
        self.assertEqual(model.tool_choices[-1], 'none')
        self.assertTrue(answer.text.startswith('Answer from data'))
        workbook.close()

    async def test_deadline_cancels_a_stalled_model(self):
        class SlowModel(ScriptedModel):
            async def _agenerate(self, *args, **kwargs):
                await asyncio.sleep(10)
        workbook = student_workbook()
        with self.assertRaises(TimeoutError):
            await answer_question(SlowModel(), workbook, [{'role': 'user', 'content': 'x'}], timeout=0.05)
        workbook.close()


class ContextTests(unittest.TestCase):
    def request(self, messages):
        from langchain.agents.middleware.types import ModelRequest
        workbook = student_workbook()
        toolkit = Toolkit(workbook)
        with patch('agent.qa_agent.create_agent', side_effect=lambda **kwargs: kwargs):
            setup = create_qa_agent(None, toolkit)
        workbook.close()
        return ModelRequest(model=None, messages=messages, system_prompt=setup['system_prompt'], tools=setup['tools'])

    def test_older_tool_output_is_retired_to_fit(self):
        messages = [HumanMessage('Current question')]
        for n in range(4):
            messages += [AIMessage('', tool_calls=[{'id': str(n), 'name': 'read_rows', 'args': {'sheet': 'x'}}]),
                         ToolMessage(json.dumps({'rows': [['x' * 100] * 3 for _ in range(25)]}), tool_call_id=str(n))]
        bounded = fit_request(self.request(messages), 6000)
        self.assertLessEqual(request_size(bounded, bounded.messages), 6000)
        self.assertEqual(bounded.messages[-1].tool_call_id, '3')
        for message in bounded.messages:
            if isinstance(message, ToolMessage):
                json.loads(message.content)

    def test_oversized_question_fails_before_calling_the_provider(self):
        with self.assertRaises(ContextBudgetError):
            fit_request(self.request([HumanMessage('long question ' * 10000)]), 6000)


class ModelTests(unittest.TestCase):
    def test_groq_client_is_bounded(self):
        model = build_model(DEFAULT_MODEL, 'test-key')
        self.assertEqual(model.max_retries, 0)
        self.assertEqual(model.reasoning_effort, 'low')
        self.assertIn('groq.com', str(model.openai_api_base))

    def test_provider_errors_do_not_leak_details(self):
        for status, expected in [(401, 'API key'), (429, 'rate limit'), (404, 'not available'), (503, 'temporarily')]:
            exc = RuntimeError('secret-key and private cell value')
            exc.status_code = status
            self.assertIn(expected, friendly_error(exc))
            self.assertNotIn('secret-key', friendly_error(exc))


def csv_upload(text, name='assets.csv'):
    upload = io.BytesIO(text.encode())
    upload.name, upload.size, upload.file_id = name, len(upload.getvalue()), name
    return upload


class UITests(unittest.TestCase):
    def test_empty_state_has_no_errors(self):
        app = AppTest.from_file(APP).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual(app.selectbox[0].value, DEFAULT_MODEL)

    def test_question_is_answered_with_result_table(self):
        model = ScriptedModel(script=[('find_rows', {'sheet': 'assets', 'columns': ['Asset'],
                                                     'filters': [{'column': 'Inspection', 'op': 'is_missing'}]})])
        upload = csv_upload('Asset,Inspection\nMachine-A,\nMachine-B,Complete\n')
        with patch('streamlit.file_uploader', return_value=[upload]), \
             patch('agent.factory.build_model', return_value=model):
            app = AppTest.from_file(APP).run(timeout=30)
            self.assertEqual(len(app.button), 2)  # generic starter questions for a text-only sheet
            app.text_input(key='api_key').set_value('test-key').run(timeout=30)
            app.chat_input[0].set_value('Which assets were not inspected?').run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            reply = app.session_state['messages'][-1]
            self.assertIn('Machine-A', reply['content'])
            self.assertEqual(reply['steps'][0].frame['Asset'].tolist(), ['Machine-A'])
            self.assertTrue(any('Download CSV' in b.label for b in app.get('download_button')))

    def test_failed_question_keeps_data_and_history_clean(self):
        upload = csv_upload('a\n1\n')
        with patch('streamlit.file_uploader', return_value=[upload]), \
             patch('agent.qa_agent.answer_question', side_effect=TimeoutError()), \
             patch('agent.factory.build_model', return_value=None):
            app = AppTest.from_file(APP).run(timeout=30)
            app.text_input(key='api_key').set_value('test-key').run(timeout=30)
            app.chat_input[0].set_value('Total?').run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertIn('too long', app.error[0].value)
            self.assertTrue(app.session_state['workbook'])
            self.assertEqual(app.session_state['messages'], [])

    def test_starter_question_runs_without_typing(self):
        model = ScriptedModel()
        with patch('streamlit.file_uploader', return_value=[csv_upload('Item,Qty\nA,1\nB,2\n')]), \
             patch('agent.factory.build_model', return_value=model):
            app = AppTest.from_file(APP).run(timeout=30)
            app.text_input(key='api_key').set_value('test-key').run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state['messages'][0]['content'], 'Give me a quick overview of assets.csv')
            self.assertEqual(len(app.session_state['messages']), 2)


if __name__ == '__main__':
    unittest.main()
