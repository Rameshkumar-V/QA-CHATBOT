"""Bounded execution and safe, actionable errors for the chat UI."""
import asyncio

ANALYSIS_TIMEOUT = 60
REQUEST_TIMEOUT = 20
MAX_MODEL_CALLS = 5


class ContextBudgetError(ValueError):
    pass


async def analyze(agent, chat, progress=None, timeout=ANALYSIS_TIMEOUT):
    """Cancel an asynchronous agent run at the overall deadline, including retries."""
    last = None
    calls = 0
    async with asyncio.timeout(timeout):
        async for update in agent.astream({'messages': chat},
                                         config={'recursion_limit': MAX_MODEL_CALLS * 4 + 4}, stream_mode='updates'):
            for node, value in update.items():
                if not isinstance(value, dict):
                    continue
                for message in value.get('messages', []):
                    if getattr(message, 'type', None) == 'ai':
                        calls += 1
                        last = message
                        if progress:
                            progress(f'Model response {calls}/{MAX_MODEL_CALLS}')
                        for call in getattr(message, 'tool_calls', []):
                            if progress:
                                progress(f"Running {call['name']} on the server…")
                    elif getattr(message, 'type', None) == 'tool' and progress:
                        progress(f"Completed {message.name or 'data tool'}")
    if last is None or getattr(last, 'tool_calls', None):
        raise ValueError('The model did not finish an answer. Try a more specific question.')
    content = last.content
    if isinstance(content, list):
        content = '\n'.join(block.get('text', '') for block in content if isinstance(block, dict))
    if not content or not str(content).strip():
        raise ValueError('The model returned an empty answer. Try a more specific question.')
    return str(content)


def friendly_error(exc):
    """Never expose raw provider responses, which may contain keys or uploaded data."""
    kind = type(exc).__name__
    status = getattr(exc, 'status_code', None)
    # Inspect provider text only for classification; do not display it.
    detail = str(exc).lower()
    if isinstance(exc, TimeoutError) or 'timeout' in kind.lower():
        return 'Analysis timed out. Ask one focused question or try again. No automatic retry was made.'
    if isinstance(exc, ContextBudgetError) or any(term in detail for term in
            ('context_length', 'context window', 'maximum context', 'too many tokens', 'request too large')):
        return 'The model context or token limit was reached. Clear the conversation and ask a shorter, more specific question, or choose a model with more context.'
    if status in (401, 403) or 'authentication' in kind.lower():
        return 'The provider rejected access. Check the API key, selected provider, and model permissions in Settings.'
    if status == 429 or 'ratelimit' in kind.lower() or 'quota' in detail:
        return 'The provider rate limit or quota was reached. Wait before retrying, or check your API account quota and billing.'
    if status == 404 or 'model_not_found' in detail or 'model_decommissioned' in detail:
        return 'This model is unavailable. Enter a current model ID supported by your provider account in Settings.'
    if 'connection' in kind.lower() or status and status >= 500:
        return 'The AI service could not be reached or is temporarily unavailable. Try again shortly.'
    if kind in ('GraphRecursionError', 'ModelCallLimitExceededError'):
        return 'The analysis step limit was reached before a final answer was produced. Your files are still loaded; retry the question or clear the conversation.'
    if status == 400 or 'tool' in detail and any(word in detail for word in ('failed', 'invalid', 'support')):
        return 'The model could not complete a valid request or tool call. Try a simpler question or another model that supports tool calling.'
    if isinstance(exc, ValueError) and str(exc).startswith(('Please shorten', 'The model')):
        return str(exc)
    return 'Analysis failed unexpectedly. Try a simpler question, check the provider settings, and retry.'
