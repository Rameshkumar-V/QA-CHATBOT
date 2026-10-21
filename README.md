# Personal AI Data Agent

General-purpose Streamlit Q&A over uploaded data using user-provided Gemini, OpenAI, xAI (Grok) or GroqCloud API keys.

Default: **GroqCloud** with **`openai/gpt-oss-120b`**, using a Groq API key. Other providers and models remain selectable. Groq model and tool support: https://console.groq.com/docs/tool-use/overview

## Deploy on Streamlit Community Cloud

1. Push this folder to your GitHub repository. Do not commit `.venv`, API keys, or uploaded datasets.
2. At https://share.streamlit.io choose **Create app**, select your repository and branch `main`, and set the main file to **app.py**.
3. In **Advanced settings**, choose **Python 3.12**. Dependencies install from `requirements.txt`; no Windows packages are needed.
4. Optionally add provider keys in the Cloud **Secrets** settings using `.streamlit/secrets.toml.example`. These keys are shared with all app visitors and consume the owner's quota. Otherwise each visitor enters a key in the sidebar.
5. Deploy. Choose a provider and model that can produce JSON query plans, upload an Excel workbook, and ask a question.

Deployment reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## General Q&A over uploaded data

Excel headers are detected from a bounded preview of the first 30 rows, so sparse title/blank rows above a dense header are not automatically used as field names. The sidebar has **Excel header settings** for each workbook/sheet: Automatic, Select header row (Excel row numbers start at 1), or No header. Manual selection supports one to three header rows and combines merged group/child labels. A preview shows original row numbers. Review or override automatic selection for ambiguous or headerless layouts; automatic inference cannot distinguish every all-text layout.

Header whitespace/newlines are cleaned, duplicates receive unique suffixes, and genuinely blank headers with data get positional labels such as Column D. Only completely empty columns and rows are removed. Data under a blank header is retained. The loaded-table caption shows the selected header row, and changes rebuild both query/search indexes and clear stale chat/results. The same cleaned column names and header metadata are passed to the Q&A planner. No subject-specific field names are used for detection.

Questions use one schema-driven flow for every dataset. There are no field-name dictionaries, typo lists, topic-specific question handlers or preset answers. The model sees the actual uploaded column names and types and generates a JSON plan containing a read-only SQL query. The server validates and executes it against a session-local SQLite copy, then sends only a bounded computed result to the model for a concise answer. The result table and CSV are also available directly.

This supports filters with multiple conditions, missing/present values, totals and averages, grouped comparisons, sorting, top-N rankings, duplicate detection, date expressions, subqueries, window functions and joins across uploaded sheets/files. Questions requiring unavailable information or ambiguous fields should receive a clarification instead of an invented result. Native provider tool calling is not required.

The internal table and column IDs are generated from each upload (T0, T1, C0, C1, ...). Original labels are included in the model schema; SQL results use meaningful aliases. Arbitrary field names, including labels not anticipated by the application, go through the same planner. The Query dataset selector can narrow the schema for large workbooks. Cross-file queries require All uploaded datasets.

The SQLite connection is read-only after loading and authorizes reads only from uploaded tables in the selected scope. Writes, external database/file access, extension loading and multiple SQL statements are rejected. Each query has a three-second execution budget after the data is loaded. The engine provides generic is_missing(value) and to_number(value) functions: missing includes nulls, blank/whitespace and explicit absence markers; zero is not assumed missing. to_number returns NULL for invalid numeric text.

## Large files and small models

10,000 rows do not need to fit inside the model context. The model gets the schema and question, not the entire workbook. SQL calculations operate over all relevant rows. Query results are limited to 10,000 rows or 10 MB for display/export, with partial results explicitly marked. Chat shows at most 20 result rows and bounds long text. Counts, sums and other SQL aggregates are computed before the result-display limit applies.

Input size is checked before each planning request, including the schema, question and history. The default budget is 6,000 conservative token units estimated from serialized UTF-8 bytes plus overhead. Older chat turns are removed first. Oversized schemas produce guidance to select fewer datasets or increase the budget. Allow at least 1,200 additional output tokens. A model still needs enough context for the schema and reliable JSON/SQL planning.

For free-text questions, the planner can query a session-local SQLite FTS5 index. It is built once on first use over overlapping passages and reused. Up to three ranked excerpts carry table, row, column and page references. This is keyword retrieval rather than semantic embeddings; no match is not proof that a topic is absent.

## Progress, deadlines and errors

The UI shows planning, validation/execution and answer-writing progress. Each provider request has a 20-second timeout with automatic retries disabled. Planning can correct an invalid query once; optional answer writing has a ten-second timeout. Overall analysis has a 60-second deadline. If answer writing fails, the verified calculated result remains available. Loading/indexing work already executing synchronously may finish during cancellation; file parsing happens before analysis.

Provider errors and malformed plans receive actionable messages without raw provider responses or uploaded cell contents. Failed questions do not enter subsequent model history. Uploaded tables and valid query results stay within the current session.

Limits: 20 MB/file, 60 MB combined uploads, 200,000 rows/table, 30 sheets/workbook, 200 MB estimated DataFrame memory/session; SQL and text indexes use additional memory. Text/PDF text is limited to 1,000,000 characters; PDFs to 100 pages. Oversized files receive guidance rather than silently dropping the end. API keys and provider/model selections are session-scoped by default.

Run checks with `python -m unittest discover -s tests -v`. Tests cover arbitrary schemas, compound filters, joins, window functions, 10,000-row calculations, read-only enforcement, output limits, query correction, indexed retrieval, deadlines and the UI without provider API calls. Live model interpretation still requires validation with an enabled API key and model.

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
- Schema-planned read-only SQL for tabular questions, indexed retrieval for text questions, and bounded model explanations.
- Uploads are read locally into pandas, and relevant tool outputs are sent to the selected cloud LLM. No arbitrary Python execution and no modification of source files.
- Max 20 MB/file; 200,000 rows/table; up to 30 Excel sheets; text files/PDF text max 1,000,000 characters; PDF max 100 pages. Oversized text/PDF files are rejected with split-file guidance rather than silently dropping the end. Avoid opening files that contain highly sensitive information unless your chosen AI provider's data policy is acceptable.
- API keys and provider/model settings stay in the browser session by default. For a trusted local desktop only, install `requirements-desktop.txt` and set `ENABLE_OS_KEYRING=1` to enable optional OS credential storage. Keep this option disabled on shared servers.
- Queries support joins and analytical SQL; chart generation, source-file editing and information outside uploaded data are not implemented.
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
