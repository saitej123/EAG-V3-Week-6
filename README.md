# E‑Commerce Concierge — Session 6 cognitive agent

Python agent that combines **Indian e‑commerce price analysis** use cases with the **four-role architecture**: **Memory**, **Perception**, **Decision**, and **Action (MCP)**. All cross-role boundaries are **Pydantic v2** models in **`schemas.py`**.

**Deep dive:** see **`MODULE_REFERENCE.md`** (assignment ↔ modules ↔ types).

---

## Architecture (four layers + loop)

| Layer | Module | Responsibility |
|--------|--------|------------------|
| **Contracts** | **`schemas.py`** | `MemoryItem`, `Goal`, `Observation`, `ToolCall`, `DecisionOutput`, LLM payload models (`PerceptionLLMResponse`, `DecisionLLMFlat`, `MemoryClassifyLLM`, …), commerce DB rows |
| **Memory** | **`memory.py`** (`MemoryService` / `MemoryManager`) | **`read()`** — keyword-ranked recall (no LLM); **`remember()`** — one structured classification LLM call; **`record_outcome()`** — append **`tool_outcome`** rows; **`query_products` / `upsert_product`** — SQLite catalog |
| **Artifacts** | **`artifact_store.py`** | **`ArtifactStore`** — content-addressable **`art:<sha256-prefix>`** (`.bin` + `.json`); reads legacy **`art:*.txt`**; large MCP results offload above **4 KiB** |
| **Perception** | **`perception.py`** | **`observe(query, hits, history, prior_goals, run_id) → Observation`** — ordered goals; model emits **`artifact_index`** only (no free-form `art:` handles) |
| **Decision** | **`decision.py`** | **`next_step(...) → DecisionOutput`** — **`answer`** *or* **`tool_call`** (wire format **`DecisionLLMFlat`** → mapped in code) |
| **Action** | **`action.py`** | **`execute(ToolCall, store=ArtifactStore) → (descriptor, artifact_id?)`** — MCP **stdio** (`mcp_server.py`); **`gemini_live_search`**; blocks **`art:`** in tool arguments |
| **Config / timeouts** | **`llm_env.py`** | API keys, model list, **`shared_gemini_client()`**, agent timeouts / iteration cap |
| **Orchestration** | **`agent6.py`** | **`remember(user_query)`** once, then each iteration: **`read → observe → next_step → execute → record_outcome`**; **`history: list[dict]`** |

**Web UI:** **`app.py`** — FastAPI, **`GET /`**, **`POST /run-agent`**, **`GET /stream-logs`** (SSE). One run at a time → **429** if busy. Successful final answers also emit **`[UI_RESULT_JSON]`** for the markdown result panel in **`templates/index.html`**.

**LLM calls:** **`google-genai`** with **`response_schema`** (structured JSON validated by Pydantic). There is **no** separate HTTP Gateway process in this repo (only **`shared_gemini_client()`**).

---

## Proof-of-Prompt (PoP)

| File | Contents |
|------|----------|
| **`pop/perception_pop.json`** | Perception schema name (`PerceptionLLMResponse`), temperature **1.0**, constraints |
| **`pop/decision_pop.json`** | Decision wire schema (`DecisionLLMFlat`), mapping to `DecisionOutput` |

---

## Requirements

- **`uv`** for dependencies and **`uv run`** (no manual `venv` activation required)
- **Python 3.12+**

---

## Setup

```bash
uv sync
```

Create **`.env`** in the repo root (same directory as **`agent6.py`** — loaded via absolute path):

```env
GEMINI_API_KEY=
# Prefer one of:
GEMINI_MODEL=
# or comma-separated fallbacks:
GEMINI_MODELS=

TAVILY_API_KEY=

# Optional tuning:
AGENT_MAX_ITERATIONS=5
AGENT_RUN_MAX_SECONDS=900
AGENT_LLM_STEP_TIMEOUT_SEC=120
```

- **`GEMINI_MODEL` / `GEMINI_MODELS`**: model IDs; extra comma-separated IDs are **fallbacks** only.
- **`TAVILY_API_KEY`**: optional; search can fall back to DuckDuckGo when Tavily is unavailable or over cap.
- **`AGENT_MAX_ITERATIONS`**: max perceive→decide→act loops (default **5**, clamped 1–50).
- **`AGENT_RUN_MAX_SECONDS`**: wall-clock cap for **`app.py`** agent jobs (default **900**).
- **`AGENT_LLM_STEP_TIMEOUT_SEC`**: budget for each Perception / Decision LLM call.

---

## Run (CLI)

```bash
uv run python agent6.py "Your query here"
```

On startup the loop **always** calls **`memory.remember(..., source="user_query", ...)`** so durable facts in the utterance are classified and stored (supports **Query C** across runs).

---

## Run (Web UI)

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/** — **Run agent**, watch **Live console**; formatted answers appear in **Agent result** when **`>>> FINAL ANSWER <<<`** / **`[UI_RESULT_JSON]`** are emitted.

---

## On-disk state (`state/` — gitignored)

| Path | Role |
|------|------|
| **`state/memory.json`** | **`{"items": [ ... MemoryItem ... ]}`** (legacy **`{"facts":[]}`** is migrated on load) |
| **`state/commerce.db`** | Cached product rows (optional PDP catalog) |
| **`state/artifacts/``** | **`ArtifactStore`** blobs + metadata (and legacy **`art:*.txt`**) |

### Clean slate (assignment / grading)

```bash
rm -rf state/
```

The next run recreates directories as needed.

---

## Expected console patterns (current code)

Illustrative lines you should see (exact wording varies by model and tools):

- **`[memory.remember]`** — after classification of the raw user query (`kind=`, `keywords=`).
- **`[Iteration i/N]`** — **`N`** from **`AGENT_MAX_ITERATIONS`** (Web UI uses the same loop).
- **`[memory.read] K ranked hits`** — keyword-ranked **`MemoryItem`** pool.
- **`[done=true|false] …`** — goal **`text`** from **`Observation`**; optional **`attach=art:…`**.
- **`[decision] ANSWER recorded`** — partial answer for the current goal; Perception marks **`done`** on a later iteration using history.
- **`-> TOOL_CALL …`** then MCP previews — tool execution; **`record_outcome`** writes episodic memory with optional **`artifact_id`** (content-addressable **`art:<sha256-prefix>`** for large payloads).
- **`>>> FINAL ANSWER <<<`** — when all goals done (last **`answer`** in history) or after **max-iteration** markdown wrap-up.

For submissions, **re-run all four queries from a clean `state/`** on your machine and paste **real** terminal transcripts into this README (or an appendix) as required by your instructor.

---

## Four target queries (commands)

Re-run after **`rm -rf state/`** when you need reproducible traces.

### Query A — Shannon Wikipedia (artifact attach)

```bash
uv run python agent6.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
```

Expect: **`fetch_url`** (or **`fetch_urls`**) → large payload stored as artifact → later iteration attaches bytes → **`Decision`** answers from attached content without refetching.

### Query B — Tokyo activities + weather

```bash
uv run python agent6.py "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."
```

Expect: discovery tools → forecast fetch → final **`answer`** comparing activities to weather via **`memory.read`** hits + history.

### Query C — Durable memory (two runs)

**Run 1:**

```bash
uv run python agent6.py "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day."
```

Expect: **`remember`** classification persists **`MemoryItem`** facts; **`create_file`** (or similar) via MCP sandbox rules.

**Run 2** (same **`state/`**, do **not** delete between Run 1 and Run 2):

```bash
uv run python agent6.py "When is mom's birthday?"
```

Expect: **`memory.read`** surfaces stored facts; short **`answer`** grounded on durable memory.

### Query D — Asyncio multi-source synthesis

```bash
uv run python agent6.py "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."
```

Expect: **`web_search`** → fetches per URL (or **`fetch_urls`**) → artifacts recorded → synthesis goal with attachment → consolidated **`answer`**.

---

## MCP server

**`mcp_server.py`** — stdio tools (**`web_search`**, **`fetch_url`**, **`fetch_urls`**, **`query_database`**, **`analyze_image_url`**, sandbox file tools, **`get_time`**, **`currency_convert`**, …). Requires matching **`uv`** deps (**`crawl4ai`**, Tavily/DDG stack as configured).

---

## Repo layout (quick reference)

| Path | Role |
|------|------|
| **`schemas.py`** | All boundary Pydantic models |
| **`agent6.py`** | Cognitive loop |
| **`memory.py`**, **`artifact_store.py`**, **`perception.py`**, **`decision.py`**, **`action.py`** | Four layers + artifact store |
| **`mcp_server.py`** | MCP tool implementations |
| **`app.py`**, **`templates/index.html`** | Web UI + SSE |
| **`llm_env.py`** | Env-driven LLM client + timeouts |
| **`MODULE_REFERENCE.md`** | Session 6 ↔ implementation matrix |
