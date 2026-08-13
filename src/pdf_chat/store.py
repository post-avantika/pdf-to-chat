"""
Oracle vector store layer for pdf_chat.

This module is the ONLY place in the project that talks directly to Oracle.
Everything else (chain.py, ingest.py, adapter.py) imports from here.

Owns:
    - Database connection lifecycle (lazy singleton — created once, reused)
    - Embedder factory (lazy singleton — MiniLM loaded once, ~90 MB weights)
    - OracleVS wrapper per collection (DOCUMENTS, CONVERSATIONS)
    - bootstrap() — ensures tables exist before ingest or search

Collections:
    PDF_CHAT_DOCUMENTS    — stores PDF chunk embeddings
    PDF_CHAT_CONVERSATIONS — reserved for future conversation-level summaries

Table naming convention: <PROJECT_PREFIX>_<KIND>
    Two projects on the same Oracle DB won't collide because each has a unique prefix.

Cites:
    shared/references/langchain-oracledb.md
"""

# MUST be the very first import — patches langchain-oracledb before it's used.
# If this line moves below the OracleVS import, the patch won't apply in time.
from . import _monkeypatch  # noqa: F401
# ↑ The "noqa: F401" comment tells linters: "I know this import looks unused
#   (we never reference _monkeypatch by name), but it IS used — its side effect
#   of patching runs at import time. Don't flag it as an unused import."

import os
# ↑ Python's built-in module for reading environment variables.
#   os.environ["KEY"] reads from the shell environment (loaded from .env by dotenv).

import oracledb
# ↑ Oracle's official Python database driver (thin mode — pure Python, no Oracle Client).
#   We use: oracledb.connect(), connection.cursor(), cursor.execute()

from dotenv import load_dotenv
# ↑ Reads .env file and loads all KEY=VALUE pairs into os.environ.
#   Must be called before any os.environ["..."] access.

from langchain_oracledb.vectorstores.oraclevs import OracleVS
# ↑ LangChain's Oracle vector store class.
#   Handles: table creation, embedding insertion, vector similarity search.
#   Does NOT handle: chat history (we write that in history.py ourselves).

from langchain_community.vectorstores.utils import DistanceStrategy
# ↑ Enum defining how to measure vector distance.
#   langchain-oracledb uses DistanceStrategy but does NOT re-export it —
#   you MUST import it from langchain_community. This is a known gotcha.
#   DistanceStrategy.COSINE = measure angle between vectors (best for text).

from langchain_huggingface import HuggingFaceEmbeddings
# ↑ LangChain wrapper around sentence-transformers.
#   Takes a HuggingFace model name, downloads it once, runs locally.
#   Implements LangChain's Embeddings interface (embed_query, embed_documents).

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_PREFIX = "PDF_CHAT"
# ↑ Prefixes all Oracle table names for this project.
#   PDF_CHAT_DOCUMENTS, PDF_CHAT_CONVERSATIONS
#   Change this if you rename the project to avoid table name collisions.

EXPECTED_DIM = 384
# ↑ The number of dimensions (floats) that MiniLM-L6-v2 produces per text chunk.
#   This is hard-coded because OracleVS creates the VECTOR(384) column on first use.
#   If you switch to a different embedder (e.g. Cohere = 1024 dims),
#   you must drop the table and re-bootstrap — the column size can't change.

COLLECTIONS = ["DOCUMENTS", "CONVERSATIONS"]
# ↑ The two vector store collections (= Oracle tables) this project uses.
#   DOCUMENTS: stores your PDF chunks
#   CONVERSATIONS: reserved (used by advanced tier for conversation-level embeddings)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
# "Lazy" means: don't create until first use. "Singleton" means: create only once.
# This pattern avoids paying the connection + model-load cost at import time.

_conn: oracledb.Connection | None = None
_embedder: HuggingFaceEmbeddings | None = None


def get_connection() -> oracledb.Connection:
    """
    Return the shared Oracle connection, creating it if needed.

    Uses the thin driver (pure Python — no Oracle Client installation required).
    Reads DB_USER, DB_PASSWORD, DB_DSN from environment (loaded from .env).

    The .ping() check detects a stale/dropped connection and reconnects.
    """
    global _conn
    load_dotenv()  # safe to call multiple times — only loads once
    if _conn is None:
        _conn = oracledb.connect(
            user=os.environ["DB_USER"],
            # ↑ The app user we created: PDF_CHAT
            #   Never connect as SYSTEM here — SYSTEM's tablespace lacks ASSM
            #   and OracleVS's JSON metadata column would fail with ORA-43853.
            password=os.environ["DB_PASSWORD"],
            dsn=os.environ["DB_DSN"],
            # ↑ Format: "localhost:1521/FREEPDB1"
            #   host:port/service_name
        )
    else:
        try:
            _conn.ping()
            # ↑ Sends a lightweight roundtrip to Oracle to verify the connection
            #   is still alive. If the DB restarted while we were idle, this
            #   raises an exception and we reconnect below.
        except oracledb.Error:
            _conn = oracledb.connect(
                user=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
                dsn=os.environ["DB_DSN"],
            )
    return _conn


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Return the shared MiniLM embedder, loading it if needed.

    First call: downloads model weights (~90 MB) to ~/.cache/huggingface/
    and loads them into memory. Takes ~5-10s on first run.
    Subsequent calls: returns the already-loaded model instantly.

    The model name is read from EMBED_MODEL env var so you can swap it
    without changing code (but remember: changing the model requires
    re-ingesting all PDFs because embedding spaces are incompatible).
    """
    global _embedder
    load_dotenv()
    if _embedder is None:
        model_name = os.environ.get(
            "EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",  # fallback default
        )
        print(f"[store] Loading embedder: {model_name} (first call — may take a moment)")
        _embedder = HuggingFaceEmbeddings(model_name=model_name)
        print(f"[store] Embedder ready. Dim={EXPECTED_DIM}")
    return _embedder


def get_store(kind: str) -> OracleVS:
    """
    Return an OracleVS instance for the given collection kind.

    Args:
        kind: One of "DOCUMENTS" or "CONVERSATIONS".
              Case-insensitive — will be uppercased automatically.

    Returns:
        OracleVS connected to the table PDF_CHAT_<KIND>.

    Note: This does NOT guarantee the table exists. Call bootstrap() first.
    """
    table = f"{PROJECT_PREFIX}_{kind.upper()}"
    # ↑ e.g. kind="DOCUMENTS" → table="PDF_CHAT_DOCUMENTS"
    #   Oracle table names are uppercase by convention.

    return OracleVS(
        client=get_connection(),
        # ↑ The shared Oracle connection from get_connection().

        embedding_function=get_embedder(),
        # ↑ The embedder used to convert text → vectors.
        #   OracleVS calls embedder.embed_query() for searches
        #   and embedder.embed_documents() for insertions.

        table_name=table,
        # ↑ The Oracle table to store/search vectors in.
        #   OracleVS creates this table automatically on first add_texts() call.

        distance_strategy=DistanceStrategy.COSINE,
        # ↑ How to measure similarity between vectors.
        #   COSINE = angle between vectors (0 = same direction = most similar).
        #   All searches on this table will use VECTOR_DISTANCE(..., COSINE) SQL.
    )


def bootstrap() -> None:
    """
    Ensure all collection tables exist in Oracle.

    How: Insert a dummy document → OracleVS creates the table → delete the dummy.
    This is the "bootstrap dance" — it's idempotent (safe to call repeatedly).

    Why not just CREATE TABLE manually?
        OracleVS.from_texts() creates the table with the correct VECTOR(384) schema.
        If we wrote the DDL ourselves, we'd have to match its exact schema — fragile.
        Let the library do it correctly, we just trigger it.

    Call this once at app startup (in adapter.py) before any ingest or search.
    """
    print("[store] Bootstrapping collections...")
    for kind in COLLECTIONS:
        store = get_store(kind)
        # Insert a dummy text to trigger table creation
        dummy_ids = store.add_texts(
            ["__bootstrap__"],
            metadatas=[{"_skip": True}],
        )
        # Immediately delete it — we don't want dummy data in the store
        store.delete(dummy_ids)
        print(f"[store]   {PROJECT_PREFIX}_{kind} ✓")
    print("[store] Bootstrap complete.")
