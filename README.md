# Data QA Agent

Ask questions about Excel and CSV files in plain language. A GroqCloud model plans queries with tool calls; the queries run over every row on your machine (or your Streamlit server), and only small, relevant results go back to the model.

Default model: **`openai/gpt-oss-120b`** on GroqCloud. Other Groq tool-calling models are in the sidebar picker, and you can type any model ID enabled on your account. Model list: https://console.groq.com/docs/models

## How the agent works

Each sheet becomes a pandas DataFrame with a short ID (`sales.xlsx › Orders` → `sales_orders`). The model gets a compact overview (sheet IDs, row counts, column names and types) and five tools:

| Tool | What it does | What the model receives |
|---|---|---|
| `describe_sheet` | Column types, missing counts, min/max/mean, most common values, 3 sample rows | A summary, never the rows |
| `find_rows` | Filters every row (`=`, `!=`, `>`, `between`, `in`, `contains`, `is_missing`, …), optional sort for top-N | Exact match count + at most 25 rows |
| `aggregate` | count, count_distinct, sum, mean, median, min, max, std; filtered and grouped, with day/week/month/quarter/year buckets | The computed numbers, at most 30 groups |
| `read_rows` | Peeks at a few rows by position | At most 25 rows |
| `sql_query` | Read-only SQLite for joins, calculated columns and window functions | At most 25 result rows |

A 10,000-row sheet is never sent to the model. Counts and totals always cover all matching rows, and when more rows match than the model saw, the answer says so. You get the full result table (up to 10,000 rows) under the answer, with a CSV download and a chart for grouped results.

The tools are forgiving: column names match regardless of case and punctuation, an unknown column returns "Did you mean …?" so the agent can correct itself, and `==`/`avg`-style aliases are accepted. Text comparisons ignore case. Dates are compared as dates, and a date-only bound such as `<= 2024-02-29` covers the whole day.

Safety and limits:
- No arbitrary Python runs. SQL is read-only: writes, `ATTACH`, pragmas, extension loading and multiple statements are rejected, and each query has a 3-second budget.
- Each answer can use up to 6 model calls; the last one must answer without tools. A whole answer has a 90-second deadline and each request has a 30-second timeout.
- Before each request the context is trimmed to the budget (default 12,000 tokens, under Model › Advanced). Older tool results are retired first.
- Cell values are treated as data, never instructions. Provider error messages are never shown raw.

## Loading data

- CSV/TSV (delimiter auto-detected: `,` `;` tab `|`), XLSX, XLSM and XLS. Each Excel sheet is its own table.
- Text dates become real dates when the format is unambiguous (ISO, or day/month order provable from the data). Formatted numbers such as `₹1,200` become numbers. Codes with leading zeros stay text.
- Excel headers are detected from the first 30 rows, so title rows above the real header are skipped. Use **Excel header settings** in the sidebar to pick the header row (1 to 3 rows, merged labels combined) or choose "No header".
- Limits: 20 MB per file, 60 MB in total, 200,000 rows per sheet, 30 sheets per workbook, 200 MB of DataFrame memory.

## Run locally (Windows)

1. Install Python 3.11 or 3.12.
2. `py -m venv .venv`
3. `.venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. `run_windows.bat` (or `streamlit run app.py`)
6. Paste a GroqCloud key (https://console.groq.com/keys) in the sidebar, upload files, and ask.

Run the tests with `python -m unittest discover -s tests -v`. They cover the tools on 10,000-row sheets, SQL safety, loading, the agent loop with a scripted model, and the UI, without calling GroqCloud.

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub. Do not commit `.venv`, API keys or data files.
2. At https://share.streamlit.io choose **Create app**, select the repository, branch `main` and main file **app.py**, and pick Python 3.12 under **Advanced settings**.
3. Optionally set `GROQ_API_KEY` in **Secrets** (see `.streamlit/secrets.toml.example`). That key is shared by every visitor; otherwise each visitor pastes their own.

## Windows EXE

Run `build_windows.bat` on Windows. The executable is `dist\PersonalDataAgent\PersonalDataAgent.exe`; keep the whole folder together. It starts Streamlit locally and opens a browser tab. For a trusted single-user machine you can install `requirements-desktop.txt` and set `ENABLE_OS_KEYRING=1` to save the key in the Windows credential store.

## Folder structure

```text
app.py                      # Streamlit UI
agent/
  factory.py                # GroqCloud model list and client
  qa_agent.py               # Agent prompt, tool loop, progress and deadline
  context_budget.py         # Per-request context trimming and forced final answer
services/
  data_tools.py             # CSV/Excel loading and type cleanup
  excel_headers.py          # Header row detection
  workbook.py               # Sheet IDs and the overview the model sees
  dataframe_tools.py        # The agent's DataFrame tools
  table_store.py            # Read-only SQLite copy for sql_query
  analysis.py, context.py   # Time limits, error messages, chat history bounds
desktop/launcher.py         # EXE entry point
```
