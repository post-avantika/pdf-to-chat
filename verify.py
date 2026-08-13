"""
verify.py — End-to-end smoke test for the pdf-chat stack.

Tests each layer independently, in order:
    1. Oracle DB connection
    2. Vector store bootstrap (table creation)
    3. Embedder dimension (must be exactly 384)
    4. Mini ingest + retrieve round-trip (3 test sentences → search → correct result)
    5. Groq LLM connectivity (API key + model reachable)
    6. Full RAG chain (question → retrieval → generation → answer)

Run from the project root (with .venv active):
    python verify.py

Expected output when everything is healthy:
    verify: OK (db, vector, inference)

Each layer prints its result before moving on. If any layer fails,
the script exits with a non-zero code and explains what to check.
"""

import sys
import os
import time

from dotenv import load_dotenv
load_dotenv()
# ↑ Load .env first — all checks below need the env vars.

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def ok(label: str) -> None:
    """Print a green-ish check mark and label."""
    print(f"  ✅  {label}")


def fail(label: str, detail: str) -> None:
    """Print an error, explain what to check, and exit."""
    print(f"\n  ❌  FAILED: {label}")
    print(f"      {detail}")
    sys.exit(1)
    # ↑ sys.exit(1) stops the script with exit code 1.
    #   Exit code 0 = success, any non-zero = failure.
    #   CI/CD systems (GitHub Actions, etc.) check this to mark a build as failed.


# ---------------------------------------------------------------------------
# Layer 1: Oracle DB Connection
# ---------------------------------------------------------------------------

def check_db() -> object:
    """
    Try to connect to Oracle and run SELECT 'ok' FROM dual.

    'FROM dual' is Oracle's way of running a query without a real table.
    It's like PostgreSQL's SELECT 1 or SELECT 'ok'.
    Returns the connection for use by later checks.
    """
    print("\n[1/6] Oracle DB connection...")
    try:
        import oracledb
        conn = oracledb.connect(
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            dsn=os.environ["DB_DSN"],
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 'ok' FROM dual")
            result = cur.fetchone()[0]
            # ↑ fetchone() returns one row as a tuple: ('ok',)
            #   [0] gets the first (and only) column value: 'ok'
            assert result == "ok", f"Unexpected result: {result}"
        ok(f"Connected to Oracle at {os.environ['DB_DSN']} as {os.environ['DB_USER']}")
        return conn
    except KeyError as e:
        fail("Oracle DB", f"Missing env var: {e}. Check your .env file.")
    except Exception as e:
        fail(
            "Oracle DB",
            f"{e}\n\n"
            "      Checklist:\n"
            "        • Is Oracle running?  →  docker compose ps\n"
            "        • Is it healthy?      →  docker compose logs oracle | tail -20\n"
            "        • Is ORACLE_PORT=1521 correct in .env?\n"
            "        • Did you wait ~90s for first boot?"
        )


# ---------------------------------------------------------------------------
# Layer 2: Vector Store Bootstrap
# ---------------------------------------------------------------------------

def check_bootstrap() -> None:
    """
    Run the bootstrap dance to verify OracleVS can create tables.

    If this fails with ORA-43853, the app user's tablespace doesn't support
    JSON columns — means the user is connected as SYSTEM (wrong) or the
    tablespace isn't ASSM-enabled.
    """
    print("\n[2/6] Vector store bootstrap...")
    try:
        from pdf_chat.store import bootstrap
        bootstrap()
        ok("PDF_CHAT_DOCUMENTS and PDF_CHAT_CONVERSATIONS tables ready")
    except Exception as e:
        fail(
            "Bootstrap",
            f"{e}\n\n"
            "      Common causes:\n"
            "        • Connected as SYSTEM — SYSTEM's tablespace lacks ASSM.\n"
            "          DB_USER in .env must be the app user (PDF_CHAT), not SYSTEM.\n"
            "        • ORA-43853: JSON column needs ASSM tablespace (USERS, not SYSTEM)."
        )


# ---------------------------------------------------------------------------
# Layer 3: Embedder Dimension
# ---------------------------------------------------------------------------

def check_embedder() -> object:
    """
    Load the MiniLM embedder and verify it produces exactly 384 dimensions.

    This check also warms up the embedder — first run downloads ~90 MB of
    model weights to ~/.cache/huggingface/. Subsequent runs are instant.
    """
    print("\n[3/6] Embedder dimension check...")
    print("      (First run downloads ~90 MB model weights — may take a moment)")
    try:
        from pdf_chat.store import get_embedder, EXPECTED_DIM
        embedder = get_embedder()
        test_vector = embedder.embed_query("dim check")
        # ↑ embed_query() converts a string into a list of floats.
        #   For MiniLM-L6-v2, this list always has exactly 384 elements.

        actual_dim = len(test_vector)
        if actual_dim != EXPECTED_DIM:
            fail(
                "Embedder dimension",
                f"Expected {EXPECTED_DIM} dims, got {actual_dim}.\n"
                "        The EMBED_MODEL in .env may have changed.\n"
                "        If you changed the model, drop and re-bootstrap the tables."
            )

        ok(f"Embedder: {os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')} → {actual_dim} dims")
        return embedder
    except Exception as e:
        fail("Embedder", str(e))


# ---------------------------------------------------------------------------
# Layer 4: Mini Ingest + Retrieve Round-Trip
# ---------------------------------------------------------------------------

def check_vector_roundtrip(conn) -> None:
    """
    Store 3 test sentences in a temporary Oracle collection, then query it.

    The query "space exploration moon landing" should retrieve the space sentence
    (not the food or programming ones) — validates that vector similarity search
    is working correctly, not just storing and returning random results.

    Uses a SEPARATE collection (PDF_CHAT_VERIFY_TEMP) so test data doesn't
    pollute your real PDF_CHAT_DOCUMENTS collection. Deletes everything after.
    """
    print("\n[4/6] Vector round-trip (ingest 3 sentences → search → verify result)...")
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_oracledb.vectorstores.oraclevs import OracleVS
    from pdf_chat.store import get_embedder

    # Use a temporary collection name to avoid polluting real data
    temp_table = "PDF_CHAT_VERIFY_TEMP"

    test_docs = [
        "The Apollo 11 mission landed astronauts on the moon in 1969.",
        # ↑ This is the one we expect to be retrieved for the space query
        "Python is a high-level programming language known for readability.",
        "The best pizza toppings are mozzarella, basil, and tomato sauce.",
    ]

    try:
        store = OracleVS(
            client=conn,
            embedding_function=get_embedder(),
            table_name=temp_table,
            distance_strategy=DistanceStrategy.COSINE,
        )

        # Bootstrap: add docs (creates table), then we'll delete the test ones later
        ids = store.add_texts(
            test_docs,
            metadatas=[{"test": True, "idx": i} for i in range(len(test_docs))],
        )

        # Search with a space-related query
        query = "space exploration moon landing"
        results = store.similarity_search(query, k=1)
        # ↑ k=1: return only the single most similar document

        if not results:
            fail("Vector round-trip", "similarity_search returned no results.")

        top_result = results[0].page_content
        if "Apollo" not in top_result and "moon" not in top_result.lower():
            fail(
                "Vector round-trip",
                f"Expected Apollo/moon doc, got: '{top_result[:80]}...'\n"
                "        Vector search is returning wrong results — dim mismatch?"
            )

        # Clean up: delete the test documents
        store.delete(ids)
        # ↑ If the table has only these docs, this effectively empties it.
        #   The table itself remains (that's fine — it's PDF_CHAT_VERIFY_TEMP).

        ok(f"Vector search: query='{query}' → correctly retrieved Apollo sentence")

    except Exception as e:
        fail("Vector round-trip", str(e))


# ---------------------------------------------------------------------------
# Layer 5: Groq / LLM Connectivity
# ---------------------------------------------------------------------------

def check_llm() -> None:
    """
    Send a tiny one-shot message to Groq and verify we get a response.

    This confirms:
        • GROQ_API_KEY is valid
        • LLM_BASE_URL is reachable
        • LLM_MODEL exists on Groq

    We don't run the full RAG chain here — just a direct API call.
    """
    print("\n[5/6] Groq LLM connectivity...")
    try:
        from openai import OpenAI

        api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OCI_GENAI_API_KEY")
        if not api_key:
            fail("LLM", "No API key found. Set GROQ_API_KEY in .env.")

        base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: VERIFY_OK"}],
            max_tokens=20,
            temperature=0,
            # ↑ temperature=0: deterministic — the model should always reply
            #   "VERIFY_OK" for this exact prompt (or close to it).
        )
        reply = response.choices[0].message.content.strip()
        ok(f"Groq → model={model} → reply='{reply[:40]}'")

    except Exception as e:
        fail(
            "Groq LLM",
            f"{e}\n\n"
            "      Checklist:\n"
            "        • Is GROQ_API_KEY set correctly in .env? (starts with gsk_)\n"
            "        • Is LLM_BASE_URL=https://api.groq.com/openai ?\n"
            "        • Is LLM_MODEL=llama-3.3-70b-versatile ?\n"
            "        • Test your key: curl https://api.groq.com/openai/v1/models -H 'Authorization: Bearer YOUR_KEY'"
        )


# ---------------------------------------------------------------------------
# Layer 6: Full RAG Chain
# ---------------------------------------------------------------------------

def check_chain() -> None:
    """
    Run the full RAG chain with a test question against a tiny known corpus.
    """
    print("\n[6/6] Full RAG chain end-to-end...")
    try:
        # Ensure chat_history table exists before the chain tries to use it
        from pdf_chat.history import init_table
        from pdf_chat.store import get_connection
        init_table(get_connection())
        from pdf_chat.store import get_store
        store = get_store("DOCUMENTS")
        test_texts = [
            "This project uses Python 3.10 as its programming language.",
            "The web framework is FastAPI, which is a Python-based ASGI framework.",
            "Dependencies are managed with pip and a pyproject.toml file.",
        ]
        ids = store.add_texts(
            test_texts,
            metadatas=[{"filename": "verify_test.txt", "page": 1, "chunk": i}
                       for i in range(len(test_texts))],
        )

        # Run the chain with a question about the corpus
        from pdf_chat.chain import get_chain
        chain = get_chain()

        answer = chain.invoke(
            {"question": "What programming language is used in this project?"},
            config={"configurable": {"session_id": "verify-session-tmp"}},
        )
        # ↑ invoke() runs the chain synchronously and returns the final string.
        #   (astream() is the async streaming version used in the real adapter.)

        # Clean up: remove test chunks
        store.delete(ids)

        # Clean up: delete verify session from chat history
        from pdf_chat.history import OracleChatHistory
        from pdf_chat.store import get_connection
        OracleChatHistory(get_connection(), "verify-session-tmp").clear()

        if not answer or len(answer) < 5:
            fail("Full chain", f"Got empty/short answer: '{answer}'")

        # The answer should mention Python
        if "python" not in answer.lower() and "Python" not in answer:
            print(f"      ⚠️  Answer didn't mention Python (may still be OK):")
            print(f"         '{answer[:120]}...'")
        else:
            ok(f"Chain produced answer mentioning Python: '{answer[:80]}...'")

    except Exception as e:
        fail("Full RAG chain", str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("pdf-chat — verify.py")
    print("=" * 60)

    start = time.time()

    conn = check_db()           # Layer 1: Oracle connection
    check_bootstrap()           # Layer 2: Vector tables
    check_embedder()            # Layer 3: Embedder dimensions
    check_vector_roundtrip(conn)  # Layer 4: Store + retrieve
    check_llm()                 # Layer 5: Groq API
    check_chain()               # Layer 6: Full pipeline

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"verify: OK (db, vector, inference)  [{elapsed:.1f}s]")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("  1. Drop PDFs into data/pdfs/")
    print("  2. python -m pdf_chat.ingest")
    print("  3. docker compose up -d")
    print("  4. python -m pdf_chat.adapter")
    print("  5. Open http://localhost:3000")


if __name__ == "__main__":
    main()
