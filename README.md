# pdf-chat 📄💬

A local RAG (Retrieval-Augmented Generation) chatbot that lets you chat with your PDF documents. Drop PDFs in a folder, run one command to ingest them, and get a ChatGPT-like interface powered by your own documents.

**Stack:**
- 🗄️ **Oracle Database 26ai Free** — stores PDF chunks as vectors (AI Vector Search) + chat history
- 🔢 **MiniLM-L6-v2** — local embeddings (runs on your CPU, no API needed, ~90 MB)
- 🧠 **Groq** (`llama-3.3-70b-versatile`) — LLM generation via OpenAI-compatible API
- 🔗 **LangChain** — RAG chain orchestration + message history
- 🌐 **Open WebUI** — polished ChatGPT-like browser UI
- ⚡ **FastAPI** — adapter that connects Open WebUI to the RAG chain

---

## Architecture

```
Browser (http://localhost:3000)
    └── Open WebUI
            └── POST /v1/chat/completions
                    └── FastAPI adapter (port 8000)
                            ├── LangChain RAG chain
                            │       ├── Oracle 26ai (vector search → top 5 PDF chunks)
                            │       ├── Oracle 26ai (load/save chat history)
                            │       └── Groq API (generate answer with citations)
                            └── Response streams back token by token
```

---

## Prerequisites

- Docker (running) — `docker --version`
- Python 3.10+ — `python3 --version`
- Groq API key — get one free at [console.groq.com/keys](https://console.groq.com/keys)

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/post-avantika/pdf-to-chat.git pdf-chat
cd pdf-chat

cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 2. Create Python virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install -e .
```

> First install downloads `sentence-transformers` (~90 MB model weights). Subsequent installs are instant.

### 3. Start Oracle + Open WebUI via Docker

```bash
docker compose up -d
```

Oracle takes ~90 seconds on first boot. Check status:
```bash
docker compose ps          # both should show "healthy"
docker compose logs oracle # view Oracle startup logs
```

### 4. Set up Oracle app user (first time only)

```bash
python -m pdf_chat.setup_db
```

> This creates the `PDF_CHAT` Oracle user with the correct tablespace and grants.

### 5. Verify the full stack

```bash
python verify.py
```

Expected output:
```
verify: OK (db, vector, inference)
```

### 6. Add your PDFs and ingest them

```bash
# Drop your PDFs into data/pdfs/
cp ~/Downloads/my_document.pdf data/pdfs/

# Ingest (embed + store in Oracle)
python -m pdf_chat.ingest
```

### 7. Start the FastAPI adapter

```bash
python -m pdf_chat.adapter
```

The adapter is now running at `http://localhost:8000`.

### 8. Open the chat UI

Go to **http://localhost:3000** in your browser.

Open WebUI is already configured to talk to your local adapter. Select the `pdf-chat` model and start asking questions about your PDFs!

---

## Project Structure

```
pdf-chat/
├── docker-compose.yml          Oracle 26ai Free + Open WebUI
├── .env                        Secrets (GROQ_API_KEY, DB passwords) — never commit!
├── .env.example                Safe template for .env
├── pyproject.toml              Python dependencies
├── verify.py                   End-to-end smoke test
│
├── data/pdfs/                  Drop your PDF files here
├── init-schema/                SQL files Oracle runs on first boot (optional)
├── migrations/
│   └── 001_chat_history.sql    Creates the chat_history table
│
└── src/pdf_chat/
    ├── __init__.py             Package definition
    ├── _monkeypatch.py         Fix for langchain-oracledb metadata parsing bug
    ├── store.py                Oracle connection + OracleVS vector store layer
    ├── history.py              OracleChatHistory — persists conversations in Oracle
    ├── ingest.py               PDF → chunks → embeddings → Oracle
    ├── chain.py                RAG chain: retrieve → prompt → Groq → stream
    └── adapter.py              FastAPI: /v1/chat/completions for Open WebUI
```

---

## How It Works

### Ingestion (runs once per PDF)

```
PDF file
  └── pypdf extracts text page by page
        └── each page split into 800-word overlapping chunks
              └── MiniLM converts each chunk → 384-dimensional vector
                    └── stored in Oracle (PDF_CHAT_DOCUMENTS table)
```

### Query (happens on every chat message)

```
Your question
  └── MiniLM converts question → 384-dimensional vector
        └── Oracle VECTOR_DISTANCE() finds 5 most similar chunks
              └── chunks formatted with citations [filename:p.N]
                    └── [history + context + question] sent to Groq
                          └── answer streams back with citations
```

---

## Common Commands

```bash
# Start everything
docker compose up -d && python -m pdf_chat.adapter

# Ingest new PDFs
python -m pdf_chat.ingest

# Run smoke tests
python verify.py

# View Oracle logs
docker compose logs oracle

# Stop everything
docker compose down          # keeps data
docker compose down -v       # deletes all data (nuclear option)

# Re-ingest all PDFs from scratch
rm data/.ingested.json
python -m pdf_chat.ingest
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key (`gsk_...`) | required |
| `LLM_BASE_URL` | Groq API base URL | `https://api.groq.com/openai` |
| `LLM_MODEL` | Groq model name | `llama-3.3-70b-versatile` |
| `EMBED_MODEL` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `DB_USER` | Oracle app user | `PDF_CHAT` |
| `DB_PASSWORD` | Oracle app user password | set in `.env` |
| `DB_DSN` | Oracle connection string | `localhost:1521/FREEPDB1` |
| `ORACLE_PWD` | Oracle admin password | generated |
| `PDF_DIR` | Folder to scan for PDFs | `data/pdfs` |

---

## Why Oracle 26ai?

Unlike using a separate vector database (Pinecone, ChromaDB, pgvector), Oracle 26ai stores everything in one place:

- ✅ **PDF vectors** — in `PDF_CHAT_DOCUMENTS` (VECTOR column, COSINE distance)
- ✅ **Chat history** — in `chat_history` (JSON CLOB, validated at DB level)
- ✅ **JSON metadata** — per chunk (filename, page, ingested_at)

No extra services to manage. One `docker compose up -d` starts everything.

---

## Upgrading to pgvector (after you're done learning Oracle)

The architecture is designed for easy swapping. To switch to PostgreSQL + pgvector:
1. Replace `langchain-oracledb` → `langchain-postgres` in `pyproject.toml`
2. Replace `OracleVS` → `PGVector` in `store.py`
3. Replace `oracledb.connect(...)` → `psycopg.connect(...)` in `store.py` and `history.py`
4. Update `docker-compose.yml` to use `pgvector/pgvector` image

`chain.py`, `adapter.py`, `ingest.py` — **unchanged**.

---

## Screenshot

> _Drop a 30-second demo GIF here: `docs/demo.gif`_
> 
> Suggested flow: drag PDF into data/pdfs/ → run ingest → open Open WebUI → ask a question → show cited answer

---

Built with the [oracle-ai-developer-hub](https://github.com/oracle-devrel/oracle-ai-developer-hub) build-paths skill.
