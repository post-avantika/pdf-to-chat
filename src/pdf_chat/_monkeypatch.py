"""
langchain-oracledb stores metadata as JSON strings, but its similarity_search
return path doesn't always parse them back. This monkeypatch makes the parsing
consistent. MUST be imported before any OracleVS instantiation in your app.

Source: shared/references/langchain-oracledb.md § "Metadata-as-string fix"

WHY THIS EXISTS
---------------
Oracle stores OracleVS metadata as VARCHAR2/JSON. When read back, oracledb
returns it as a Python *string*, but the langchain-oracledb internals expect a
dict. Filtered retrievals then fail with `AttributeError: 'str' object has no
attribute 'pop'` — silent in some code paths, loud in others.

This patch sits in front of `_read_similarity_output` (the module-level
function — patching the class no-ops) and parses the metadata column to a dict
before the library sees it.

Apply ONCE, near the top of your project's main module or `store.py`. Importing
this file is enough — the patch runs at import time.
"""

from __future__ import annotations
# ↑ Allows modern type hint syntax (e.g. list[str] instead of List[str])
#   on Python 3.10. It makes the file forward-compatible.

import json
# ↑ Python's built-in JSON parser.
#   json.loads("...") = string → Python dict
#   json.dumps({...}) = Python dict → string

try:
    import langchain_oracledb.vectorstores.oraclevs as _vs_module
    # ↑ Import the internal oraclevs module directly.
    #   We need to reach inside the library to patch a module-level function.
    #   The underscore prefix in _vs_module means "we're treating this as private".

    _orig_read_similarity_output = _vs_module._read_similarity_output
    # ↑ Save a reference to the ORIGINAL (broken) function before we replace it.
    #   This is important: we still call the original at the end of our
    #   fixed version — we only fix the data going INTO it, not replace it entirely.

    def _fixed_read_similarity_output(
        results,
        has_similarity_score: bool = False,
        has_embeddings: bool = False,
    ):
        # This function receives raw rows from Oracle.
        # Each `row` is a tuple: (content_text, metadata, [score], [embedding])
        # The bug: `metadata` (index 1) comes back as a JSON *string* from Oracle.
        # The fix: parse it into a Python dict before passing it to the original function.

        fixed = []
        for row in results:
            if len(row) >= 2:
                row_list = list(row)        # tuples are immutable → convert to list
                metadata = row_list[1]      # grab the metadata column

                if isinstance(metadata, str):
                    # It's a string — try to parse it as JSON
                    try:
                        row_list[1] = json.loads(metadata)
                        # json.loads converts: '{"page": 3}' → {"page": 3}
                    except Exception:
                        # If it's somehow invalid JSON, leave it as-is.
                        # Better to have a string than crash the whole search.
                        pass

                fixed.append(tuple(row_list))   # convert back to tuple
            else:
                fixed.append(row)               # rows with < 2 columns, leave alone

        # Call the original function with our fixed rows
        return _orig_read_similarity_output(fixed, has_similarity_score, has_embeddings)

    # THE ACTUAL PATCH: replace the library's function with our fixed version
    _vs_module._read_similarity_output = _fixed_read_similarity_output
    # ↑ From this point on, whenever langchain-oracledb calls _read_similarity_output
    #   internally, it runs OUR version (which fixes the metadata), not the original.

except Exception as e:  # noqa: BLE001 — this must never crash the host process
    # If for any reason the patch fails (e.g. future version changes the function name),
    # we print a warning but DO NOT crash. The app will still start — you'll just
    # see metadata errors if they occur. This is a deliberate safety choice.
    print(f"[pdf-chat] failed to apply OracleVS metadata monkeypatch: {e}")
