"""GroqCloud model catalog and client construction."""
from langchain_openai import ChatOpenAI

from services.analysis import REQUEST_TIMEOUT

GROQ_BASE_URL = 'https://api.groq.com/openai/v1'
GROQ_KEY_URL = 'https://console.groq.com/keys'
# Tool-calling models from https://console.groq.com/docs/models; the first is the default.
MODELS = {
    'openai/gpt-oss-120b': 'Recommended: best tool use',
    'openai/gpt-oss-20b': 'Faster, lower cost',
    'llama-3.3-70b-versatile': 'General purpose',
    'qwen/qwen3.8-27b': 'Preview',
}
DEFAULT_MODEL = next(iter(MODELS))
MAX_OUTPUT_TOKENS = 4096


def build_model(model_name, api_key):
    extra = {}
    if model_name.startswith('openai/gpt-oss'):
        # Medium reasoning produces noticeably fewer malformed tool calls than low.
        extra = {'reasoning_effort': 'medium'}
    return ChatOpenAI(model=model_name, api_key=api_key, base_url=GROQ_BASE_URL, use_responses_api=False,
                      temperature=0.2, timeout=REQUEST_TIMEOUT, max_retries=0, max_tokens=MAX_OUTPUT_TOKENS,
                      **extra)
