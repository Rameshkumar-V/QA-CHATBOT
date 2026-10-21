"""Keep previous turns bounded independently of dataset size."""
MAX_QUESTION_CHARS = 4000
MAX_HISTORY_CHARS = 6000


def bounded_chat(messages):
    """Retain the current question and recent complete question/answer pairs."""
    current = messages[-1]
    if len(current['content']) > MAX_QUESTION_CHARS:
        raise ValueError('Please shorten your question to 4,000 characters or fewer.')
    history = []
    used = 0
    previous = messages[:-1]
    for index in range(len(previous) - 2, -1, -2):
        pair = previous[index:index + 2]
        if [m['role'] for m in pair] != ['user', 'assistant']:
            break
        size = sum(len(m['content']) for m in pair)
        if used + size > MAX_HISTORY_CHARS or len(history) >= 4:
            break
        history = pair + history
        used += size
    return history + [current]
