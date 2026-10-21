"""Time limits and safe, actionable errors for the chat UI."""
ANALYSIS_TIMEOUT = 90
REQUEST_TIMEOUT = 30
MAX_MODEL_CALLS = 6


class ContextBudgetError(ValueError):
    pass


def friendly_error(exc):
    """Never expose raw provider responses, which may contain keys or uploaded data."""
    kind = type(exc).__name__
    status = getattr(exc, 'status_code', None)
    # Inspect provider text only for classification; do not display it.
    detail = str(exc).lower()
    if isinstance(exc, TimeoutError) or 'timeout' in kind.lower():
        return 'The answer took too long. Ask one focused question or try again.'
    if isinstance(exc, ContextBudgetError) or any(term in detail for term in
            ('context_length', 'context window', 'maximum context', 'too many tokens', 'request too large')):
        return 'The context limit was reached. Start a new chat, focus on one sheet, or raise the context budget under Model › Advanced.'
    if status in (401, 403) or 'authentication' in kind.lower():
        return 'GroqCloud rejected the API key. Check the key in the sidebar.'
    if status == 429 or 'ratelimit' in kind.lower() or 'quota' in detail:
        return 'GroqCloud rate limit reached. Wait a minute and retry, or pick a smaller model.'
    if status == 404 or 'model_not_found' in detail or 'model_decommissioned' in detail:
        return 'This model is not available on your GroqCloud account. Pick another model in the sidebar.'
    if 'connection' in kind.lower() or status and status >= 500:
        return 'GroqCloud could not be reached or is temporarily unavailable. Try again shortly.'
    if kind in ('GraphRecursionError', 'ModelCallLimitExceededError'):
        return 'The agent ran out of steps before answering. Try a more specific question.'
    if status == 400 or 'tool' in detail and any(word in detail for word in ('failed', 'invalid', 'support')):
        return 'The model made an invalid tool call. Retry, or switch to openai/gpt-oss-120b.'
    if isinstance(exc, ValueError) and str(exc).startswith(('Please shorten', 'The model', 'Enter your')):
        return str(exc)
    return 'Something went wrong while answering. Retry, or try a simpler question.'
