# Personal AI Data Agent

Local Streamlit UI + LangChain read-only data agent using user-provided Gemini, OpenAI, xAI (Grok) or GroqCloud API keys.

Default: **GroqCloud** with **`openai/gpt-oss-120b`**, using a Groq API key. Other providers and models remain selectable. Groq model and tool support: https://console.groq.com/docs/tool-use/overview

## Deploy on Streamlit Community Cloud

1. Push this folder to your GitHub repository. Do not commit `.venv`, API keys, or uploaded datasets.
2. At https://share.streamlit.io choose **Create app**, select your repository and branch `main`, and set the main file to **app.py**.
3. In **Advanced settings**, choose **Python 3.12**. Dependencies install from `requirements.txt`; no Windows packages are needed.
4. Optionally add provider keys in the Cloud **Secrets** settings using `.streamlit/secrets.toml.example`. These keys are shared with all app visitors and consume the owner's quota. Otherwise each visitor enters a key in the sidebar.
5. Deploy. Choose a provider and a model that supports tool calling, upload an Excel workbook, and ask a question.

Deployment reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Large Excel files and small models

10,000 rows are processed by pandas on the app server. The model receives table metadata, bounded samples, and calculated results; the whole workbook is never inserted into a prompt. Totals, counts, filters, and grouped calculations scan the full table. Preview rows are only for inspection. Grouped results return the top 20 groups with the total group count, so a partial listing is explicit.

For questions such as “Which students have not given their Aadhaar number?”, `filter_rows` supports `is_missing` and `is_not_missing` with no comparison value. Missing includes nulls, blank/whitespace cells and explicit markers (`-`, `--`, `NA`, `N/A`, `NaN`, `None`, `null`, `not provided`, `not submitted`, `not available`), ignoring case. Zero is not assumed missing. The tool scans the full column, returns the exact match count, and can project only student name/roll columns rather than returning every field. Lists remain bounded to at most 20 matching rows. Schema inspection reports both null and broader missing counts; full statistics/samples are only added for an explicitly requested column.

Straightforward missing-Aadhaar questions also have a **direct local query path**, including “Which Students does not have an Adaar?” and Aadhaar/Adhaar/Aadhar/Adaar spelling variants. This path does not call an AI model and works without an API key. It checks all rows, reports counts and up to 20 identifiers in chat, and displays **all** matching identifiers in per-source tables with CSV downloads. Aadhaar values themselves are not included in those result tables. Source data row positions start at 1, excluding the header. If more than one Aadhaar column could apply, the app asks for the exact column. Questions with additional filters (such as section or unpaid fees) stay with the general agent; those conditions are never silently ignored. Missing Aadhaar headers receive explicit guidance rather than a provider error.

Each tool response is limited to 3,000 characters of valid JSON. Oversized results are marked as truncated; long cell text is marked when shortened. Schema inspection supports 12-column pages or a single requested column. Recent conversation context is limited to 6,000 characters plus the current question (maximum 4,000 characters).

Before **every** model call, middleware checks the combined system prompt, tool schemas, question, history, tool calls and results. The default input budget is 6,000 conservative token units, estimated using serialized UTF-8 bytes plus an overhead reserve, rather than assuming English characters equal tokens. Old conversation turns and tool bodies are retired first; oversized results are reduced as valid JSON with omission metadata. If input still exceeds the budget, the request is stopped locally with an actionable error. Allow at least 1,200 additional output tokens; select a higher input budget only for larger-context models. Arbitrarily tiny context windows are not supported.

## Query documents instead of sending all rows

`search_documents` uses a session-local SQLite FTS5 index, built once on its first query and reused for later questions. Text cells are indexed in overlapping 800-character passages. The agent supplies a few keywords; the index ranks matches and returns up to 3 excerpts with table, row, column, chunk and PDF page references. This works for documents with 1,000+ rows and for answers deep inside a long cell. Source rows are one-based DataFrame positions, excluding an Excel/CSV header.

Only the index builder reads the content once. Subsequent queries use the index; the model never receives the full document. Search uses literal keyword tokens with OR matching and BM25 ranking, not semantic embeddings or whole-document summarization. A lack of matches is not proof that the topic is absent; try other terms. Tools expose constrained queries, not arbitrary SQL or Python execution. SQLite must include FTS5 (standard Community Cloud Python supports this).

## Analysis progress and errors

Analysis shows model/tool progress and elapsed time. Each provider request has a 20-second timeout and no automatic retries. The async agent has a 60-second overall deadline and a maximum of 5 model calls, with output limited to 1,200 tokens per call. The final call disables further tool calls and requests an answer from verified evidence, or a precise explanation of what is still unknown. The graph step budget accounts for middleware and tool nodes as well as model calls. Old complete tool exchanges are removed before shrinking the latest query results when context is tight. Slow network calls are cancelled at the deadline; a synchronous server-side tool already running may finish its current work during cancellation. File parsing occurs before the analysis deadline.

Invalid credentials, unavailable models, rate limits/quota, context overflow, connection problems, step limits, empty answers and timeouts show recovery guidance without exposing raw provider responses. Failed questions do not enter subsequent model history, and loaded files remain available to retry.

Limits: 20 MB per file, 60 MB combined uploads, 200,000 rows per table, 30 sheets per workbook, and 200 MB estimated DataFrame memory per session. Cloud resources are shared across users; large compressed workbooks can use substantial memory during parsing before the memory check. Uploaded tables, settings and entered keys are held per session. Refreshing or reconnecting may reset them. Analysis occurs on the Cloud server when deployed, and bounded tool results are sent to the selected provider.

Run the regression checks with `python -m unittest discover -s tests -v`. They verify 10,000-row Excel calculations, indexed retrieval across 1,000 document rows, complete agent query flow, context budgets, deadlines, safe errors and session isolation without provider API calls.

## Run locally (Windows)

1. Install Python 3.11 or 3.12.
2. Open a terminal in this folder.
3. `py -m venv .venv`
4. `.venv\Scripts\activate`
5. `pip install -r requirements.txt`
6. `run_windows.bat` (or `streamlit run app.py`)
7. In sidebar settings choose provider, model name and API key. Upload files and ask questions.

Current starter model IDs are only examples; replace them with an enabled model from your provider account if an API reports unavailable model.

## Make Windows EXE

On Windows, run `build_windows.bat` from this folder. The resulting executable is at `dist\PersonalDataAgent\PersonalDataAgent.exe`. Keep the whole `PersonalDataAgent` folder together, including its `_internal` contents. Double-clicking the EXE starts Streamlit locally and opens a browser tab. It is **not** a native desktop window. Windows EXEs must be built on Windows.

PyInstaller + Streamlit bundling can need adjustments for specific versions/environment; treat build.bat as a starting point, not a guaranteed one-click build on all systems. A simple reliable alternative is shipping a .bat that runs `run_windows.bat` (or `streamlit run app.py`) on machines with Python installed.

## Functionality and limits

- CSV, XLSX, XLS, JSON, TXT, MD and text-based PDF (not scanned/OCR).
- Excel workbook sheets are loaded as separate named datasets.
- Agent tools: indexed document search, list tables, inspect columns and stats, preview rows, value counts, aggregate metrics, filter rows.
- Uploads are read locally into pandas, and relevant tool outputs are sent to the selected cloud LLM. No arbitrary Python execution and no modification of source files.
- Max 20 MB/file; 200,000 rows/table; up to 30 Excel sheets; text files/PDF text max 1,000,000 characters; PDF max 100 pages. Oversized text/PDF files are rejected with split-file guidance rather than silently dropping the end. Avoid opening files that contain highly sensitive information unless your chosen AI provider's data policy is acceptable.
- API keys and provider/model settings stay in the browser session by default. For a trusted local desktop only, install `requirements-desktop.txt` and set `ENABLE_OS_KEYRING=1` to enable optional OS credential storage. Keep this option disabled on shared servers.
- Agent tools return bounded data samples; for complex joins, plotting, pivots and larger-scale analysis, add explicit constrained tools.
- Works online only: API calls to Gemini, xAI, GroqCloud or OpenAI require internet access, API key, and available quota/billing. Provider model IDs can be changed in Settings.

## Actual folder structure

```text
Personal_AI_Data_Agent_Windows/
├── app.py                    # Streamlit UI and settings
├── requirements.txt
├── run_windows.bat           # Run from source
├── build_windows.bat         # Windows 10/11 EXE packaging
├── README.md
├── agent/
│   ├── __init__.py
│   └── factory.py            # Gemini / OpenAI / Grok / GroqCloud and LangChain agent
├── services/
│   ├── __init__.py
│   └── data_tools.py         # CSV / Excel / JSON / PDF tools
└── desktop/
    └── launcher.py           # Streamlit EXE launcher
```

Build from Windows 10 x64 using Python 3.11 to target Windows 10 and Windows 11 x64. Python is bundled in the distribution, but keep the entire `dist/PersonalDataAgent` directory. This is not a tested/prebuilt exe; build and smoke-test on both OS versions. The UI runs in your default web browser on localhost.

## GroqCloud vs xAI Grok

GroqCloud and xAI Grok are separate services with incompatible API keys. Select **GroqCloud** for keys from https://console.groq.com/keys (endpoint `https://api.groq.com/openai/v1`); select **Grok (xAI)** for keys from https://console.x.ai (endpoint `https://api.x.ai/v1`). Set the model to one supported by your provider. GroqCloud default: `openai/gpt-oss-120b`.
