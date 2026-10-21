"""Personal AI Data Agent: Streamlit + LangChain 1.x."""
import json
import os

import inspect
import asyncio
import time

import streamlit as st

from services.data_tools import read_upload, sanitize_dataframe, MAX_TOTAL_BYTES, excel_previews
from openpyxl.utils import get_column_letter
from agent.factory import build_model
from services.context import bounded_chat
from services.analysis import friendly_error, ANALYSIS_TIMEOUT
from services.search import DocumentSearch
from services.query_planner import run_query
from services.table_store import TableStore

def _width_kwarg(stretch: bool = True) -> dict:
    if 'width' in inspect.signature(st.button).parameters:
        return {'width': 'stretch' if stretch else 'content'}
    return {'use_container_width': stretch}

WIDE = _width_kwarg(True)

DEFAULT_MODELS = {'GroqCloud': 'openai/gpt-oss-120b', 'Gemini': 'gemini-2.5-flash', 'OpenAI': 'gpt-4.1-mini', 'Grok (xAI)': 'grok-4-1-fast-non-reasoning'}
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
                                   help='Conservative UTF-8 byte estimate including schema, question and prompt. Allow at least 1,200 more tokens for the response. Larger settings need a larger context model.')
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
    st.caption('Your model receives column names, types, questions and bounded chat history. Queries and calculations run over the full data on the app server. Settings and entered keys stay in this browser session.')
    st.caption('10,000+ rows do not need to fit in the model context. The model plans a query; calculations use every relevant row.')
    st.caption('The model plans a query from your column names. The server validates and executes it over all rows. Analysis has a 60-second deadline.')
    if st.button('Clear conversation', **WIDE):
        st.session_state.messages = []
        st.session_state.query_results = {}
        st.rerun()

excel_options = {}
if uploads:
    preview_cache = st.session_state.setdefault('excel_header_previews', {})
    active_previews = {(f.name, f.size, f.file_id) for f in uploads if f.name.lower().endswith(('.xlsx', '.xls'))}
    for cached in list(preview_cache):
        if cached not in active_previews:
            del preview_cache[cached]
    for upload in uploads:
        if not upload.name.lower().endswith(('.xlsx', '.xls')):
            continue
        cache_key = (upload.name, upload.size, upload.file_id)
        try:
            if cache_key not in preview_cache:
                preview_cache[cache_key] = excel_previews(upload)
            with st.sidebar.expander(f'Excel header settings: {upload.name}'):
                st.caption('Preview shows original Excel row numbers. Change the header row if the sheet starts with titles or merged cells.')
                for sheet, preview in preview_cache[cache_key].items():
                    st.markdown(f'**{sheet}**')
                    key = f'excel_{upload.file_id}_{sheet}'
                    mode = st.selectbox('Header mode', ['Automatic', 'Select header row', 'No header'], key=key + '_mode')
                    row, depth = 'auto', 1
                    if mode == 'Select header row':
                        row = int(st.number_input('Header starts at Excel row', min_value=1, max_value=1000,
                                                  value=1, key=key + '_row')) - 1
                        depth = int(st.number_input('Number of header rows', min_value=1, max_value=3,
                                                    value=1, key=key + '_depth'))
                    elif mode == 'No header':
                        row = None
                    excel_options.setdefault(upload.file_id, {})[sheet] = {'header_row': row, 'header_rows': depth}
                    visible = preview.copy()
                    visible.columns = [get_column_letter(i + 1) for i in range(len(visible.columns))]
                    visible.index = range(1, len(visible) + 1)
                    visible.index.name = 'Excel row'
                    st.dataframe(sanitize_dataframe(visible), **WIDE)
        except Exception as exc:
            st.sidebar.warning(f'Excel header preview unavailable for {upload.name}: {exc}')
    sig = (tuple((f.name, f.size, f.file_id) for f in uploads), json.dumps(excel_options, sort_keys=True))
    if sig != st.session_state.dataset_signature:
        try:
            if sum(f.size for f in uploads) > MAX_TOTAL_BYTES:
                raise ValueError('Combined uploads must be 60 MB or smaller.')
            all_tables = {}
            for upload in uploads:
                for name, frame in read_upload(upload, excel_headers=excel_options.get(upload.file_id)).items():
                    if name in all_tables:
                        raise ValueError(f'Duplicate table name: {name}; rename one file.')
                    all_tables[name] = frame
                    if sum(int(df.memory_usage(deep=True).sum()) for df in all_tables.values()) > 200 * 1024 * 1024:
                        raise ValueError('Loaded datasets exceed the 200 MB session memory limit; upload fewer files.')
            st.session_state.tables = all_tables
            if st.session_state.get('search_index'):
                st.session_state.search_index.close()
            st.session_state.search_index = DocumentSearch(all_tables)
            if st.session_state.get('table_store'):
                st.session_state.table_store.close()
            st.session_state.table_store = TableStore(all_tables)
            st.session_state.dataset_signature = sig
            st.session_state.messages = []
            st.session_state.query_results = {}
        except Exception as e:
            if st.session_state.get('search_index'):
                st.session_state.search_index.close()
            st.session_state.search_index = None
            if st.session_state.get('table_store'):
                st.session_state.table_store.close()
            st.session_state.table_store = None
            st.session_state.tables = {}
            st.session_state.dataset_signature = None
            st.session_state.messages = []
            st.session_state.query_results = {}
            st.error(f'File loading failed: {e}')
else:
    st.session_state.excel_header_previews = {}
    if st.session_state.get('search_index'):
        st.session_state.search_index.close()
    st.session_state.search_index = None
    if st.session_state.get('table_store'):
        st.session_state.table_store.close()
    st.session_state.table_store = None
    st.session_state.tables = {}
    st.session_state.dataset_signature = None
    st.session_state.query_results = {}

if st.session_state.tables:
    with st.sidebar:
        scope_options = [None] + list(st.session_state.tables)
        dataset_scope = st.selectbox('Query dataset', scope_options,
                                    format_func=lambda value: 'All uploaded datasets' if value is None else value,
                                    help='Narrow the schema sent to the model when you have many tables or columns.')
    st.subheader('Loaded datasets')
    for name, df in st.session_state.tables.items():
        with st.expander(f'{name} — {len(df):,} rows × {len(df.columns)} columns'):
            if 'header_row' in df.attrs:
                header = df.attrs['header_row']
                st.caption(f"Headers: Excel row {header}, {df.attrs['header_rows']} row(s), {df.attrs['header_mode']}."
                           if header is not None else 'No header detected/selected; all data rows are retained with positional column names.')
                if df.attrs.get('generic_headers'):
                    st.caption('Blank header cells use positional names: ' + ', '.join(df.attrs['generic_headers'])
                               + '. Adjust Excel header settings in the sidebar if the real labels are on another row.')
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
    else:
        pending = st.session_state.messages + [{'role': 'user', 'content': question}]
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            started = time.monotonic()
            status = st.status(f'Analyzing — maximum {ANALYSIS_TIMEOUT} seconds', expanded=True)
            try:
                st.session_state.query_results = {}
                chat = bounded_chat(pending)
                if not api_key:
                    raise ValueError('Enter your API key in Settings.')
                if not model_name.strip():
                    raise ValueError('Enter a model name in Settings.')
                model = build_model(provider, model_name.strip(), api_key)
                def progress(message):
                    status.write(f'{time.monotonic() - started:.0f}s · {message}')
                result = asyncio.run(run_query(model, st.session_state.tables, chat, progress,
                                              input_budget=input_budget,
                                              search_index=st.session_state.get('search_index'),
                                              store=st.session_state.get('table_store'),
                                              table_names=[dataset_scope] if dataset_scope is not None else None))
                answer = result.answer
                st.session_state.query_results = result.results
                status.update(label=f'Completed in {time.monotonic() - started:.1f}s', state='complete', expanded=False)
                st.markdown(answer)
                st.session_state.messages = pending
                st.session_state.messages.append({'role': 'assistant', 'content': answer})
            except Exception as exc:
                status.update(label='Analysis stopped', state='error', expanded=False)
                st.error(friendly_error(exc))
                st.caption('Your files are still loaded. You can submit the question again; failed turns are not added to model history.')

for index, (source, result) in enumerate(st.session_state.get('query_results', {}).items()):
    st.caption(f'{source} — {len(result):,} result rows')
    st.dataframe(sanitize_dataframe(result), **WIDE)
    st.download_button('Download query results (CSV)', result.to_csv(index=False),
                       f'query_results_{index + 1}.csv', mime='text/csv', key=f'query_result_{index}')

if st.session_state.messages:
    st.download_button('⬇️ Export conversation', json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
                       'data_agent_chat.json', mime='application/json')
