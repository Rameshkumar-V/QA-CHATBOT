"""Personal AI Data Agent: Streamlit + LangChain 1.x."""
import json
import os

import inspect

import streamlit as st

from services.data_tools import read_upload, sanitize_dataframe, MAX_TOTAL_BYTES
from agent.factory import create_data_agent, build_model
from services.context import bounded_chat

def _width_kwarg(stretch: bool = True) -> dict:
    if 'width' in inspect.signature(st.button).parameters:
        return {'width': 'stretch' if stretch else 'content'}
    return {'use_container_width': stretch}

WIDE = _width_kwarg(True)

DEFAULT_MODELS = {'Gemini': 'gemini-2.5-flash', 'OpenAI': 'gpt-4.1-mini', 'Grok (xAI)': 'grok-4-1-fast-non-reasoning', 'GroqCloud': 'llama-3.3-70b-versatile'}
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
    provider = st.selectbox('AI provider', providers, index=providers.index(config.get('provider', 'Gemini')) if config.get('provider') in providers else 0)
    model_name = st.text_input('Model name', value=config.get('models', {}).get(provider, DEFAULT_MODELS[provider]), help='Enter a model ID available on your API account.')
    if provider == 'GroqCloud':
        st.caption('GroqCloud uses api.groq.com, not the xAI Grok endpoint. Create keys at console.groq.com.')
    existing_key = load_key(provider)
    api_input = st.text_input('API key', type='password', placeholder='Saved securely' if existing_key else 'Paste API key here', help='Never paste your API key into the chat.')
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
            st.session_state.dataset_signature = sig
            st.session_state.messages = []
        except Exception as e:
            st.session_state.tables = {}
            st.session_state.dataset_signature = None
            st.session_state.messages = []
            st.error(f'File loading failed: {e}')
else:
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
        st.session_state.messages.append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            try:
                model = build_model(provider, model_name.strip(), api_key)
                agent = create_data_agent(model, st.session_state.tables)
                chat = bounded_chat(st.session_state.messages)
                with st.spinner('Analyzing files...'):
                    answer_state = agent.invoke({'messages': chat}, config={'recursion_limit': 18})
                last = answer_state['messages'][-1]
                answer = last.content
                if not isinstance(answer, str):
                    answer = '\n'.join(str(block.get('text', '')) for block in answer if isinstance(block, dict))
                answer = answer or 'No textual response was returned.'
                st.markdown(answer)
                st.session_state.messages.append({'role': 'assistant', 'content': answer})
            except Exception as exc:
                st.error(f'Agent error: {exc}')

if st.session_state.messages:
    st.download_button('⬇️ Export conversation', json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
                       'data_agent_chat.json', mime='application/json')
