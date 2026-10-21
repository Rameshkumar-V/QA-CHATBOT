import asyncio
import io
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from streamlit.testing.v1 import AppTest

from agent.context_budget import fit_request, request_size
from agent.factory import build_model, create_data_agent
from services.analysis import ContextBudgetError, analyze, friendly_error
from services.data_tools import MAX_TEXT_CHARS, MAX_TOOL_CHARS, make_tools, read_upload
from services.search import DocumentSearch


class SearchTests(unittest.TestCase):
    def test_query_finds_last_of_1000_rows_and_reuses_index(self):
        rows = ['Ordinary handbook information. ' * 10 for _ in range(1000)]
        rows[-1] = 'The cancellation policy requires notice of 14 days.'
        index = DocumentSearch({'handbook': pd.DataFrame({'text': rows})})
        tools = {tool.name: tool for tool in make_tools(index.tables, index)}
        result = json.loads(tools['search_documents'].invoke({'table': 'handbook', 'query': 'cancellation'}))
        self.assertEqual(result['passages'][0]['row'], 1000)
        self.assertIn('14 days', result['passages'][0]['excerpt'])
        connection = index.connection
        self.assertFalse(index.search('handbook', 'nonexistentkeyword')['passages'])
        self.assertIs(connection, index.connection)
        self.assertLessEqual(len(json.dumps(result)), MAX_TOOL_CHARS)
        index.close()

    def test_query_finds_match_deep_in_one_long_cell(self):
        index = DocumentSearch({'doc': pd.DataFrame({'text': ['ordinary words ' * 2000 + 'criticalneedle 42 days']})})
        result = index.search('doc', 'criticalneedle')
        self.assertIn('42 days', result['passages'][0]['excerpt'])
        self.assertGreater(result['passages'][0]['chunk'], 1)
        index.close()

    def test_1000_text_lines_not_silently_cut_at_old_limit(self):
        upload = io.BytesIO(('normal words ' * 10 + '\n') .encode() * 999 + b'lastrowneedle final answer')
        upload.name = 'guide.txt'
        tables = read_upload(upload)
        self.assertEqual(len(tables['guide.txt']), 1000)
        index = DocumentSearch(tables)
        self.assertEqual(index.search('guide.txt', 'lastrowneedle')['passages'][0]['row'], 1000)
        index.close()

    def test_oversized_text_rejected_explicitly(self):
        upload = io.BytesIO(b'a' * (MAX_TEXT_CHARS + 1))
        upload.name = 'guide.txt'
        with self.assertRaisesRegex(ValueError, 'split'):
            read_upload(upload)

    def test_pagination_and_sources(self):
        index = DocumentSearch({'doc': pd.DataFrame({'page': list(range(1, 11)), 'text': ['refund policy'] * 10})})
        first = index.search('doc', 'refund')
        second = index.search('doc', 'refund', offset=first['next_offset'])
        self.assertTrue(first['has_more'])
        self.assertEqual(first['passages'][0]['page'], '1')
        self.assertNotEqual(first['passages'][0]['row'], second['passages'][0]['row'])
        index.close()


class ContextTests(unittest.TestCase):
    def request(self, messages):
        with patch('agent.factory.create_agent', side_effect=lambda **kwargs: kwargs):
            setup = create_data_agent(None, {})
        return ModelRequest(model=None, messages=messages, system_prompt=setup['system_prompt'], tools=setup['tools'])

    def test_real_schema_and_search_result_fit_default_budget(self):
        request = self.request([HumanMessage('What is the cancellation policy?'),
                               AIMessage('', tool_calls=[{'id': 'one', 'name': 'search_documents', 'args': {'table': 'doc', 'query': 'cancellation'}}]),
                               ToolMessage(json.dumps({'passages': [{'row': n, 'excerpt': 'x' * 300} for n in range(3)]}), tool_call_id='one')])
        bounded = fit_request(request, 6000)
        self.assertLessEqual(request_size(bounded, bounded.messages), 6000)
        self.assertEqual(len(json.loads(bounded.messages[-1].content)['passages']), 3)

    def test_accumulated_context_keeps_pairs_and_valid_json(self):
        messages = [HumanMessage('Current question')]
        for n in range(4):
            messages += [AIMessage('', tool_calls=[{'id': str(n), 'name': 'preview_rows', 'args': {'table': 'doc'}}]),
                         ToolMessage(json.dumps({'rows': [{'text': 'x' * 300} for _ in range(20)]}), tool_call_id=str(n))]
        request = self.request(messages)
        bounded = fit_request(request, 6000)
        self.assertLessEqual(request_size(bounded, bounded.messages), 6000)
        tool_call_ids = {call['id'] for msg in bounded.messages if isinstance(msg, AIMessage)
                         for call in msg.tool_calls}
        for retained in bounded.messages:
            if isinstance(retained, ToolMessage):
                json.loads(retained.content)
                self.assertIn(retained.tool_call_id, tool_call_ids)
        self.assertEqual(bounded.messages[-1].tool_call_id, '3')
        self.assertEqual(bounded.messages[0].content, 'Current question')

    def test_fails_before_provider_if_question_cannot_fit(self):
        request = self.request([HumanMessage('longquestion' * 5000)])
        with self.assertRaises(ContextBudgetError):
            fit_request(request, 6000)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_agent_queries_document_and_returns_evidence(self):
        class QueryModel(BaseChatModel):
            seen: list = Field(default_factory=list)
            @property
            def _llm_type(self):
                return 'test-query-model'
            def bind_tools(self, tools, **kwargs):
                return self
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.seen.append(messages)
                last = messages[-1]
                if isinstance(last, ToolMessage) and last.name == 'search_documents':
                    evidence = json.loads(last.content)['passages'][0]
                    output = AIMessage(f"Row {evidence['row']}: {evidence['excerpt']}")
                else:
                    name = 'search_documents' if isinstance(last, ToolMessage) else 'list_tables'
                    args = {'table': 'handbook', 'query': 'cancellation'} if name == 'search_documents' else {}
                    output = AIMessage('', tool_calls=[{'name': name, 'args': args, 'id': str(len(self.seen))}])
                return ChatResult(generations=[ChatGeneration(message=output)])
        model = QueryModel()
        tables = {'handbook': pd.DataFrame({'text': ['General information'] * 999 + ['Cancellation notice is 14 days.']})}
        index = DocumentSearch(tables)
        agent = create_data_agent(model, tables, search_index=index)
        answer = await analyze(agent, [{'role': 'user', 'content': 'What is the cancellation policy?'}])
        self.assertIn('1000', answer)
        self.assertIn('14 days', answer)
        self.assertEqual(len(model.seen), 3)
        index.close()

    async def test_overall_deadline_cancels_stalled_agent(self):
        class SlowAgent:
            cancelled = False
            async def astream(self, *args, **kwargs):
                try:
                    await asyncio.sleep(10)
                    yield {}
                finally:
                    self.cancelled = True
        agent = SlowAgent()
        with self.assertRaises(TimeoutError):
            await analyze(agent, [], timeout=0.02)
        self.assertTrue(agent.cancelled)

    async def test_progress_and_final_answer(self):
        class Agent:
            async def astream(self, *args, **kwargs):
                yield {'model': {'messages': [AIMessage('', tool_calls=[{'id': '1', 'name': 'aggregate', 'args': {}}])]}}
                yield {'tools': {'messages': [ToolMessage('42', tool_call_id='1', name='aggregate')]}}
                yield {'model': {'messages': [AIMessage('The total is 42.')]}}
        progress = []
        self.assertEqual(await analyze(Agent(), [], progress.append), 'The total is 42.')
        self.assertTrue(any('aggregate' in item for item in progress))

    async def test_errors_propagate_for_actionable_ui_message(self):
        class Agent:
            async def astream(self, *args, **kwargs):
                raise ConnectionError('private key and document text')
                yield {}
        with self.assertRaises(ConnectionError):
            await analyze(Agent(), [])


class ErrorTests(unittest.TestCase):
    def test_ui_timeout_keeps_files_and_excludes_failed_turn(self):
        upload = io.BytesIO(b'text\ncancellation policy\n')
        upload.name, upload.size, upload.file_id = 'doc.csv', len(upload.getvalue()), 'test-file'
        with patch('streamlit.file_uploader', return_value=[upload]), \
             patch('services.query_planner.run_query', side_effect=TimeoutError()), \
             patch('agent.factory.build_model', return_value=None), \
             patch('agent.factory.create_data_agent', return_value=None):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            app.text_input[1].set_value('test-key').run()
            app.chat_input[0].set_value('What is the policy?').run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn('timed out', app.error[0].value)
            self.assertTrue(app.session_state['tables'])
            self.assertEqual(app.session_state['messages'], [])

    def test_provider_status_errors_do_not_leak_details(self):
        for status, expected in [(401, 'API key'), (429, 'quota'), (404, 'unavailable'), (503, 'temporarily')]:
            exc = RuntimeError('secret-api-key and sensitive uploaded row')
            exc.status_code = status
            message = friendly_error(exc)
            self.assertIn(expected, message)
            self.assertNotIn('secret-api-key', message)
        self.assertIn('timed out', friendly_error(TimeoutError()))
        self.assertIn('context', friendly_error(ContextBudgetError()))

    def test_groq_defaults_use_bounded_requests(self):
        model = build_model('GroqCloud', 'openai/gpt-oss-120b', 'test-key')
        self.assertEqual(model.max_retries, 0)
        self.assertEqual(model.max_tokens, 1200)
        self.assertEqual(model.request_timeout, 20)
        self.assertEqual(model.reasoning_effort, 'low')
