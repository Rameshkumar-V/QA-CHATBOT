"""Personal AI Data Agent: Streamlit + LangChain 1.x."""
import json
import os

import inspect
import asyncio
import time

import streamlit as st

from services.data_tools import read_upload, sanitize_dataframe, MAX_TOTAL_BYTES
from agent.factory import create_data_agent, build_model
from services.context import bounded_chat
from services.analysis import analyze, friendly_error, ANALYSIS_TIMEOUT
from services.search import DocumentSearch

def _width_kwarg(stretch: bool = True) -> dict:
    if 'width' in inspect.signature(st.button).parameters:
        return {'width': 'stretch' if stretch else 'content'}
    return {'use_container_width': stretch}

WIDE = _width_kwarg(True)

DEFAULT_MODELS = {'GroqCloud': 'openai/gpt-oss-20b', 'Gemini': 'gemini-2.5-flash', 'OpenAI': 'gpt-4.1-mini', 'Grok (xAI)': 'grok-4-1-fast-non-reasoning'}
SERVICE_KEYS = {'Gemini': 'gemini', 'OpenAI': 'openai', 'Grok (xAI)': 'grok', 'GroqCloud': 'groqcloud'}


def load_config():
    return st.session_state.get('settings', {})


def save_config(config):
    st.session_state.settings = config


def load_key(provider):
    if st.session_state.get('temp_keys', {}).get(provider):
        return st.session_state.temp_keys[provider]
    secret_names = {'Gemini': 'GOOGLE_API_KEY', 'OpenAI': 'OPENAI_API_KEY',
                    'Grok (xAI)': 'XAI_API_KEY', 'GroqCloud': 'GROQ_API_KEY'}
    name = secret_names[provider]
    try:
        secret = st.secrets.get(name, '')
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        secret = ''
    if secret or os.environ.get(name):
        return secret or os.environ[name]
    if os.environ.get('ENABLE_OS_KEYRING') != '1':
        return ''
    try:
        import keyring
        return keyring.get_password('PersonalDataAgent', SERVICE_KEYS[provider]) or ''
    except Exception:
        return ''


def save_key(provider, key):
    import keyring
    keyring.set_password('PersonalDataAgent', SERVICE_KEYS[provider], key)


st.set_page_config(page_title='Personal AI Data Agent', page_icon='📊', layout='wide')
st.title('📊 Personal AI Data Agent')
st.caption('Ask questions about CSV, Excel, JSON, TXT, Markdown and text-based PDF files with Gemini, OpenAI, Grok (xAI) or GroqCloud.')

if 'tables' not in st.session_state:
    st.session_state.tables = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'temp_keys' not in st.session_state:
    st.session_state.temp_keys = {}
if 'dataset_signature' not in st.session_state:
    st.session_state.dataset_signature = None

config = load_config()
with st.sidebar:
    st.header('⚙️ Settings')
    providers = list(DEFAULT_MODELS)
    provider = st.selectbox('AI provider', providers, index=providers.index(config.get('provider', 'GroqCloud')) if config.get('provider') in providers else 0)
    model_name = st.text_input('Model name', value=config.get('models', {}).get(provider, DEFAULT_MODELS[provider]), key=f'model_{provider}', help='Enter a model ID available on your API account.')
    input_budget = st.select_slider('Input context budget (tokens)', options=[6000, 12000, 24000], value=6000,
                                   help='Conservative UTF-8 byte estimate of input tokens, including tools and system prompt. Allow at least 1,200 more tokens for the response. Larger settings need a larger context model.')
    if provider == 'GroqCloud':
        st.caption('GroqCloud uses api.groq.com, not the xAI Grok endpoint. Create keys at console.groq.com.')
    existing_key = load_key(provider)
    api_input = st.text_input('API key', type='password', key=f'key_{provider}', placeholder='Saved securely' if existing_key else 'Paste API key here', help='Never paste your API key into the chat.')
    store_key = False
    if os.environ.get('ENABLE_OS_KEYRING') == '1':
        store_key = st.checkbox('Save API key in operating-system credential store', value=False)
    if st.button('Save settings', **WIDE):
        config['provider'] = provider
        config.setdefault('models', {})[provider] = model_name.strip()
        save_config(config)
        if api_input.strip():
            st.session_state.temp_keys[provider] = api_input.strip()
            if store_key:
                try:
                    save_key(provider, api_input.strip())
                    st.success('Settings saved; API key stored in the OS credential manager.')
                except Exception as exc:
                    st.warning(f'Settings saved. Credential storage failed: {exc}. Key is only held in this session.')
            else:
                st.success('Settings saved. Key is stored only for this session.')
        else:
            st.success('Settings saved.')
    api_key = api_input.strip() or load_key(provider)
    st.divider()
    st.header('📁 Files')
    uploads = st.file_uploader('Upload documents', type=['csv', 'xlsx', 'xls', 'json', 'txt', 'md', 'pdf'], accept_multiple_files=True)
    st.caption('Full datasets are analyzed on the app server. Only bounded tool results are sent to your AI provider. Settings and entered keys stay in this browser session.')
    st.caption('10,000+ rows work with small models that support tool calling. Calculations use every row, not just the preview.')
    st.caption('Documents use indexed keyword queries with source excerpts. Analysis stops after 60 seconds or 5 model calls.')
    if st.button('Clear conversation', **WIDE):
        st.session_state.messages = []
        st.rerun()

if uploads:
    sig = tuple((f.name, f.size, f.file_id) for f in uploads)
    if sig != st.session_state.dataset_signature:
        try:
            if sum(f.size for f in uploads) > MAX_TOTAL_BYTES:
                raise ValueError('Combined uploads must be 60 MB or smaller.')
            all_tables = {}
            for upload in uploads:
                for name, frame in read_upload(upload).items():
                    if name in all_tables:
                        raise ValueError(f'Duplicate table name: {name}; rename one file.')
                    all_tables[name] = frame
                    if sum(int(df.memory_usage(deep=True).sum()) for df in all_tables.values()) > 200 * 1024 * 1024:
                        raise ValueError('Loaded datasets exceed the 200 MB session memory limit; upload fewer files.')
            st.session_state.tables = all_tables
            if st.session_state.get('search_index'):
                st.session_state.search_index.close()
            st.session_state.search_index = DocumentSearch(all_tables)
            st.session_state.dataset_signature = sig
            st.session_state.messages = []
        except Exception as e:
            if st.session_state.get('search_index'):
                st.session_state.search_index.close()
            st.session_state.search_index = None
            st.session_state.tables = {}
            st.session_state.dataset_signature = None
            st.session_state.messages = []
            st.error(f'File loading failed: {e}')
else:
    if st.session_state.get('search_index'):
        st.session_state.search_index.close()
    st.session_state.search_index = None
    st.session_state.tables = {}
    st.session_state.dataset_signature = None

if st.session_state.tables:
    st.subheader('Loaded datasets')
    for name, df in st.session_state.tables.items():
        with st.expander(f'{name} — {len(df):,} rows × {len(df.columns)} columns'):
            st.dataframe(sanitize_dataframe(df.head(100)), **WIDE)
else:
    st.info('Upload a file using the left sidebar to start.')

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

question = st.chat_input('Example: Which product has the highest total sales?')
if question:
    if not st.session_state.tables:
        st.warning('Please upload at least one file first.')
    elif not api_key:
        st.warning('Enter your API key in Settings.')
    elif not model_name.strip():
        st.warning('Enter a model name in Settings.')
    else:
        pending = st.session_state.messages + [{'role': 'user', 'content': question}]
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            started = time.monotonic()
            status = st.status(f'Analyzing — maximum {ANALYSIS_TIMEOUT} seconds', expanded=True)
            try:
                chat = bounded_chat(pending)
                model = build_model(provider, model_name.strip(), api_key)
                agent = create_data_agent(model, st.session_state.tables, input_budget,
                                          st.session_state.get('search_index'))
                def progress(message):
                    status.write(f'{time.monotonic() - started:.0f}s · {message}')
                answer = asyncio.run(analyze(agent, chat, progress))
                status.update(label=f'Completed in {time.monotonic() - started:.1f}s', state='complete', expanded=False)
                st.markdown(answer)
                st.session_state.messages = pending
                st.session_state.messages.append({'role': 'assistant', 'content': answer})
            except Exception as exc:
                status.update(label='Analysis stopped', state='error', expanded=False)
                st.error(friendly_error(exc))
                st.caption('Your files are still loaded. You can submit the question again; failed turns are not added to model history.')

if st.session_state.messages:
    st.download_button('⬇️ Export conversation', json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
                       'data_agent_chat.json', mime='application/json')
