"""
PDF ingestion pipeline for pdf_chat.

What this script does:
    1. Walks data/pdfs/ and finds all .pdf files
    2. Skips files already recorded in data/.ingested.json (idempotent)
    3. For each new PDF:
       a. Extracts text from every page using pypdf
       b. Splits long pages into overlapping 800-token chunks
       c. Embeds each chunk via MiniLM-L6-v2 (via store.get_embedder())
       d. Stores chunks + metadata in Oracle (PDF_CHAT_DOCUMENTS table)
       e. Records the file in the ledger so it's skipped on future runs
    4. Reports a summary

Usage:
    # From the project root, with .venv active:
    python -m pdf_chat.ingest

    # Or with a custom PDF folder:
    PDF_DIR=/path/to/my/pdfs python -m pdf_chat.ingest

Drop your PDF files into data/pdfs/ and run this once before starting the server.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
# ↑ pathlib.Path is the modern Python way to work with file system paths.
#   It's object-oriented: Path("data/pdfs") / "report.pdf" builds paths safely
#   across operating systems (Windows uses \, Linux uses / — Path handles both).

from dotenv import load_dotenv
from pypdf import PdfReader
# ↑ pypdf is a pure-Python PDF library. PdfReader opens a PDF file and lets
#   you iterate over pages, extracting text. Lightweight — no system dependencies.

from pdf_chat.store import bootstrap, get_store
# ↑ Our store layer:
#   bootstrap() — ensures PDF_CHAT_DOCUMENTS table exists before we insert
#   get_store("DOCUMENTS") — returns an OracleVS instance for that collection

# ---------------------------------------------------------------------------
# Configuration constants (can be overridden via environment variables)
# ---------------------------------------------------------------------------

load_dotenv()

PDF_DIR = Path(os.environ.get("PDF_DIR", "data/pdfs"))
# ↑ Directory where users drop their PDFs.
#   Default: data/pdfs/ (relative to wherever you run the script from).
#   Override: PDF_DIR=/other/path python -m pdf_chat.ingest

LEDGER_FILE = PDF_DIR.parent / ".ingested.json"
# ↑ The idempotency ledger. Stored at data/.ingested.json (one level above pdfs/).
#   Uses Python's / operator on Path objects to build: data/.ingested.json

CHUNK_SIZE = 800
# ↑ Maximum number of words per chunk. 800 words ≈ 1000 tokens (roughly).
#   Why words not tokens? Counting exact tokens requires loading the tokenizer
#   (slow). Word count is a fast, good-enough approximation.
#   800 words gives ~1000 tokens with 384-dim MiniLM's vocabulary.

CHUNK_OVERLAP = 150
# ↑ How many words are shared between consecutive chunks.
#   Overlap ensures that a sentence at the boundary of two chunks
#   appears fully in at least one of them — prevents missing context.


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping word-based chunks.

    Args:
        text:       The full text of one PDF page (or any string).
        chunk_size: Max words per chunk.
        overlap:    Words shared between consecutive chunks.

    Returns:
        List of text chunks. If the text is shorter than chunk_size,
        returns a single-element list.

    Example:
        text = "word1 word2 ... word1200"  (1200 words)
        chunk_size=800, overlap=150

        → chunk 1: words 0–799
        → chunk 2: words 650–1199   (starts 150 words before chunk 1 ends)
    """
    words = text.split()
    # ↑ str.split() without arguments splits on any whitespace (spaces, newlines, tabs)
    #   and removes empty strings. "hello   world\n" → ["hello", "world"]

    if len(words) <= chunk_size:
        # Short page — no need to split, return as-is
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        # ↑ words[start:end] is list slicing — takes elements from index start up to (not including) end
        chunks.append(chunk)
        start += chunk_size - overlap
        # ↑ Move start forward by (chunk_size - overlap) so the next chunk
        #   overlaps with the last `overlap` words of the current chunk.
        #   e.g. chunk_size=800, overlap=150 → step forward by 650 words each time

    return chunks


# ---------------------------------------------------------------------------
# Ledger helpers (idempotency)
# ---------------------------------------------------------------------------

def load_ledger() -> dict:
    """Load the ingestion ledger from disk. Returns empty dict if not found."""
    if LEDGER_FILE.exists():
        return json.loads(LEDGER_FILE.read_text())
    return {}


def save_ledger(ledger: dict) -> None:
    """Write the updated ledger to disk."""
    LEDGER_FILE.write_text(json.dumps(ledger, indent=2))


def ledger_key(pdf_path: Path) -> str:
    """
    Fingerprint for a PDF: name + size in bytes.

    Using just the filename isn't enough — if someone replaces a PDF with
    a different file of the same name, we'd miss the update.
    Filename + file size is a lightweight fingerprint that catches most cases.
    (A full SHA256 hash would be more robust but slower for large files.)
    """
    return f"{pdf_path.name}:{pdf_path.stat().st_size}"


# ---------------------------------------------------------------------------
# Core ingestion function
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, store) -> int:
    """
    Ingest one PDF file into Oracle vector store.

    Args:
        pdf_path: Path to the .pdf file.
        store:    OracleVS instance (from get_store("DOCUMENTS")).

    Returns:
        Number of chunks ingested.
    """
    print(f"  📄 Ingesting: {pdf_path.name}")
    reader = PdfReader(str(pdf_path))
    # ↑ Opens the PDF. PdfReader parses the PDF binary format and gives us
    #   access to individual pages via reader.pages

    total_chunks = 0
    texts = []
    metadatas = []

    for page_num, page in enumerate(reader.pages, start=1):
        # ↑ enumerate(reader.pages, start=1) gives us (1, page1), (2, page2), ...
        #   start=1 because humans count pages from 1, not 0.

        raw_text = page.extract_text() or ""
        # ↑ page.extract_text() returns the page's text content as a string.
        #   Returns None for scanned/image-only PDFs (no OCR here — out of scope).
        #   The `or ""` converts None → empty string so we don't crash.

        if not raw_text.strip():
            # Skip blank pages (common in PDFs with section breaks or images)
            print(f"     ⚠️  Page {page_num}: no text extracted (image-only page?), skipping")
            continue

        # Split this page into chunks if it's long
        page_chunks = chunk_text(raw_text)

        for chunk_idx, chunk_text_content in enumerate(page_chunks):
            if not chunk_text_content.strip():
                continue  # skip whitespace-only chunks

            texts.append(chunk_text_content)
            metadatas.append({
                "filename": pdf_path.name,
                # ↑ Used for citations: [report.pdf:p.3]
                "page": page_num,
                # ↑ The PDF page number (1-indexed)
                "chunk": chunk_idx,
                # ↑ Which chunk within the page (0 for first chunk)
                "source": str(pdf_path),
                # ↑ Full path — useful for opening the file directly
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                # ↑ ISO 8601 timestamp: "2026-08-06T05:00:00+00:00"
                #   Stored as metadata so you can filter by recency later.
            })
            total_chunks += 1

    if texts:
        # Store all chunks for this PDF in one batch call
        # (more efficient than one call per chunk)
        store.add_texts(texts, metadatas=metadatas)
        # ↑ OracleVS.add_texts():
        #   1. Calls embedder.embed_documents(texts) → list of 384-dim vectors
        #   2. Runs INSERT INTO PDF_CHAT_DOCUMENTS (...) for each chunk
        #   Returns: list of IDs assigned to each chunk (we don't need them here)

    print(f"     ✅ {total_chunks} chunks stored")
    return total_chunks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the ingestion pipeline.

    Called when you run: python -m pdf_chat.ingest
    """
    print("=" * 60)
    print("pdf-chat — PDF Ingestion Pipeline")
    print("=" * 60)

    # 1. Verify the PDF directory exists
    if not PDF_DIR.exists():
        print(f"\n❌ PDF directory not found: {PDF_DIR}")
        print(f"   Create it and drop your PDFs inside:")
        print(f"   mkdir -p {PDF_DIR}")
        sys.exit(1)

    # 2. Find all PDFs
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    # ↑ Path.glob("*.pdf") returns all files matching the pattern.
    #   sorted() gives consistent ordering (alphabetical by filename).

    if not pdf_files:
        print(f"\n⚠️  No PDF files found in {PDF_DIR}")
        print(f"   Drop some PDFs there and run again.")
        sys.exit(0)

    print(f"\nFound {len(pdf_files)} PDF(s) in {PDF_DIR}")

    # 3. Load ledger to check what's already been ingested
    ledger = load_ledger()
    new_files = [p for p in pdf_files if ledger_key(p) not in ledger]
    skipped = len(pdf_files) - len(new_files)

    if skipped:
        print(f"Skipping {skipped} already-ingested file(s) (in .ingested.json)")
    if not new_files:
        print("\n✅ Nothing new to ingest. All PDFs already processed.")
        print(f"   To re-ingest, delete {LEDGER_FILE} and run again.")
        sys.exit(0)

    print(f"Processing {len(new_files)} new file(s)...\n")

    # 4. Bootstrap Oracle tables (idempotent — safe to call every run)
    print("[store] Ensuring Oracle tables exist...")
    bootstrap()

    # 5. Get the vector store for documents
    store = get_store("DOCUMENTS")

    # 6. Ingest each new PDF
    total_ingested = 0
    for pdf_path in new_files:
        try:
            num_chunks = ingest_pdf(pdf_path, store)
            ledger[ledger_key(pdf_path)] = {
                "filename": pdf_path.name,
                "size_bytes": pdf_path.stat().st_size,
                "chunks": num_chunks,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            save_ledger(ledger)
            # ↑ Save ledger after EACH file (not at the end).
            #   If the script crashes halfway through, already-processed
            #   files won't be re-ingested on the next run.
            total_ingested += num_chunks
        except Exception as e:
            print(f"  ❌ Failed to ingest {pdf_path.name}: {e}")
            print(f"     Skipping this file. Fix the error and re-run.")
            # Don't add to ledger — it will be retried next run

    # 7. Summary
    print("\n" + "=" * 60)
    print(f"✅ Ingestion complete.")
    print(f"   Total chunks stored: {total_ingested}")
    print(f"   Ledger updated:      {LEDGER_FILE}")
    print(f"\nNext step: start the adapter and Open WebUI:")
    print(f"   docker compose up -d")
    print(f"   python -m pdf_chat.adapter")
    print("=" * 60)


if __name__ == "__main__":
    # ↑ This block only runs when the file is executed directly:
    #       python src/pdf_chat/ingest.py
    #   It does NOT run when the file is imported as a module:
    #       from pdf_chat.ingest import ingest_pdf
    #
    # python -m pdf_chat.ingest also triggers this because Python
    # sets __name__ = "__main__" when a module is run as the main entry point.
    main()
