# Week 1 — `/ask` Demo (5 stages)

Build a typed LLM endpoint step by step. Each stage is a standalone FastAPI app you can run and compare.

## Setup

```bash
cp .env.example .env          # OPENAI_API_KEY=sk-...
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo stages

| Stage | File | What you learn |
|-------|------|----------------|
| 1 | `serve_stage1.py` | Bare `/ask` — string answer + `tokens_used` |
| 2 | `serve_stage2.py` | Structured output via Pydantic + `completions.parse` |
| 3 | `serve_stage3.py` | Validation guardrail + retry (`force_bad` demo knob) |
| 4 | `serve_stage4.py` | Per-request `model` override + `latency_ms` |
| 5 | `serve_stage5.py` / `main.py` | Full system + `cost_usd` readout |

Run one stage at a time (only one server on port 8000):

```bash
uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
# or the full system:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Streamlit demo runner

Interactive UI for all five stages:

```bash
streamlit run demo_page.py
```

Open http://localhost:8501. Set **API base URL** to `http://127.0.0.1:8000` and start the matching stage server in another terminal.

## Test with curl

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Estimate 3-month WTI call option?"}'
```

Stage 5 example (model + cost):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "model": "gpt-4o-mini"}'
```

## Smoke-test all stages

Requires `.venv` and a valid `OPENAI_API_KEY`:

```bash
python test_all_stages.py
```

## Pinecone vector store

The API embeds both document text and search questions with
`text-embedding-3-small` with a 1024-dimensional output. Create a Pinecone
index before using these endpoints:

- dimensions: `1024`
- metric: `cosine`

Set these values in a local `.env` file (copied from `.env.example`):

```dotenv
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=your-index-name
PINECONE_NAMESPACE=default
OPENAI_EMBEDDING_DIMENSIONS=1024
INGEST_CHUNK_SIZE=800
INGEST_CHUNK_OVERLAP=100
```

`PINECONE_NAMESPACE` is optional and defaults to `default`. The chunk settings
are optional and default to `800` characters with `100` characters of overlap.
`OPENAI_EMBEDDING_DIMENSIONS` must match the Pinecone index dimension and
defaults to `1024`.
Do not commit `.env`.

On Render, add the same variables under **Dashboard → your service → Environment**.
Set `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME` as secret
environment variables. Add `PINECONE_NAMESPACE` only if you do not want the
default namespace. Add `INGEST_CHUNK_SIZE` and `INGEST_CHUNK_OVERLAP` only to
override their defaults. Set `OPENAI_EMBEDDING_DIMENSIONS=1024` to match the
index. Render injects these values at runtime, so no `.env` file should be
deployed.

After deployment, call `GET /health/pinecone` to confirm that the configured
index is reachable. It returns the index readiness and vector count without
returning credentials. The same check is available to Python callers as
`pinecone_health()` in `main.py`.

Ingest documents:

```bash
curl -X POST http://127.0.0.1:8000/vectors/upsert \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"id":"intro","text":"Pinecone is a managed vector database.","metadata":{"source":"example"}}]}'
```

Query the configured namespace:

```bash
curl -X POST http://127.0.0.1:8000/vectors/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Pinecone?","top_k":3}'
```

Inspect retrieval without calling a generation model:

```bash
curl "http://127.0.0.1:8000/debug/retrieve?q=How%20many%20remote-work%20days%20are%20allowed%3F"
```

Chunk and ingest a document:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"document_id":"readme-v1","text":"Pinecone is a managed vector database.","metadata":{"source":"README.md"}}'
```

## Project layout

```
week-1/
├── main.py              # Full system (stages 1–5 combined)
├── serve_stage1.py … serve_stage5.py
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
