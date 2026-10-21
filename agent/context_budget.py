"""Enforce a total input token budget before every model request."""
import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage, convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool

from services.analysis import ContextBudgetError, MAX_MODEL_CALLS
from services.data_tools import compact

DEFAULT_INPUT_TOKENS = 6000


def request_size(request, messages):
    """Conservative UTF-8 byte budget, including schema and message overhead.

    Byte-level tokenizers cannot use more tokens than their input bytes. Additional
    per-message/tool overhead is reserved rather than relying on a model tokenizer.
    """
    all_messages = ([request.system_message] if request.system_message else []) + messages
    payload = {'messages': convert_to_openai_messages(all_messages),
               'tools': [convert_to_openai_tool(t) for t in request.tools]}
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8')) + 512


def fit_request(request, budget):
    messages = list(request.messages)
    # Retain the current question and valid tool-call/result pairs. Evict old turns.
    last_user = max((i for i, msg in enumerate(messages) if msg.type == 'human'), default=0)
    if request_size(request, messages) > budget and last_user:
        messages = messages[last_user:]
    # Retire old tool bodies, keeping IDs and the latest round intact.
    latest_call = max((i for i, msg in enumerate(messages)
                       if msg.type == 'ai' and getattr(msg, 'tool_calls', None)), default=-1)
    for index, msg in enumerate(messages):
        if request_size(request, messages) <= budget:
            break
        if isinstance(msg, ToolMessage) and index < latest_call:
            messages[index] = msg.model_copy(update={
                'content': '{"omitted": true, "reason": "Older tool output retired for context budget; rerun if needed."}'})
    # Remove complete old exchanges before sacrificing the latest query's evidence.
    while request_size(request, messages) > budget:
        call_indexes = [i for i, msg in enumerate(messages)
                        if msg.type == 'ai' and getattr(msg, 'tool_calls', None)]
        if len(call_indexes) < 2:
            break
        first, following = call_indexes[:2]
        exchange = messages[first:following]
        expected = {call['id'] for call in messages[first].tool_calls}
        actual = {msg.tool_call_id for msg in exchange if isinstance(msg, ToolMessage)}
        if expected != actual or any(msg.type not in ('ai', 'tool') for msg in exchange):
            break
        del messages[first:following]
    # Reduce the latest results structurally, never by cutting JSON mid-record.
    for index, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        target = len(str(msg.content)) // 2
        while request_size(request, messages) > budget and target >= 300:
            try:
                content = compact(json.loads(msg.content), max_chars=target)
            except (ValueError, TypeError):
                content = '{"error": "Tool output too large. Request a narrower result."}'
            messages[index] = msg.model_copy(update={'content': content})
            target //= 2
    if request_size(request, messages) > budget:
        raise ContextBudgetError('Input budget exceeded before calling the provider.')
    return request.override(messages=messages)


class ContextBudgetMiddleware(AgentMiddleware):
    def __init__(self, budget=DEFAULT_INPUT_TOKENS):
        self.budget = budget

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))

    def prepare(self, request):
        if request.state.get('run_model_call_count', 0) >= MAX_MODEL_CALLS - 1:
            prompt = request.system_message.content if request.system_message else ''
            request = request.override(tool_choice='none', system_message=SystemMessage(
                content=prompt + ' This is the final step. Answer from verified tool results now. '
                'If the evidence is insufficient, explain precisely what remains unknown. Do not call tools.'))
        return fit_request(request, self.budget)
