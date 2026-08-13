"""
Oracle-backed chat message history for LangChain.

`langchain-oracledb` does not ship a chat-history class — its submodules are
only document_loaders, embeddings, retrievers, utilities, vectorstores.
We roll a small one on top of raw `oracledb` + LangChain's BaseChatMessageHistory.

REQUIRES: The chat_history table must exist before use.
Run at startup:
    from pdf_chat.history import init_table
    from pdf_chat.store import get_connection
    init_table(get_connection())

Or apply migrations/001_chat_history.sql manually via SQLcl/SQL*Plus.

USAGE:
    from pdf_chat.history import OracleChatHistory, get_history_factory
    from pdf_chat.store import get_connection

    # Direct use:
    history = OracleChatHistory(get_connection(), session_id="user-42")
    history.add_user_message("What is in my PDFs?")
    print(history.messages)   # reloads from Oracle — survives restarts

    # LangChain chain wiring (used in chain.py):
    factory = get_history_factory(get_connection())
    # factory("user-42") returns an OracleChatHistory for that session
"""

from __future__ import annotations

import json
# ↑ For serializing/deserializing LangChain message objects to/from JSON strings.

import oracledb
# ↑ Oracle's Python driver. Used directly here (no LangChain wrapper needed)
#   because we're doing plain INSERT/SELECT/DELETE — not vector operations.

from langchain_core.chat_history import BaseChatMessageHistory
# ↑ LangChain's abstract base class for chat history backends.
#   Defines the interface we must implement: messages, add_message(), clear().
#   By inheriting from it, our class works with RunnableWithMessageHistory
#   and any other LangChain component that accepts a history backend.

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
# ↑ BaseMessage: the parent class for HumanMessage, AIMessage, SystemMessage, etc.
#   messages_to_dict([msg])  → serializes Python message objects to JSON-able dicts
#   messages_from_dict(rows) → deserializes JSON-able dicts back to message objects


class OracleChatHistory(BaseChatMessageHistory):
    """
    Persist LangChain chat message history in Oracle's chat_history table.

    Each conversation has a unique session_id. Messages are stored as JSON CLOBs,
    ordered by the auto-incrementing `seq` column, so history is always
    replayed in the correct order.

    The table schema is defined in migrations/001_chat_history.sql.
    """

    def __init__(
        self,
        conn: oracledb.Connection,
        session_id: str,
        table_name: str = "chat_history",
    ):
        """
        Args:
            conn:       Shared Oracle connection (from store.get_connection()).
            session_id: Unique ID for this conversation thread.
                        Open WebUI sends a conversation UUID; we use it directly.
            table_name: Override for testing (default: "chat_history").
        """
        self.conn = conn
        self.session_id = session_id
        self.table = table_name

    @property
    def messages(self) -> list[BaseMessage]:
        """
        Load all messages for this session from Oracle, in order.

        This is a @property, not a method — so callers write:
            history.messages        (not history.messages())

        LangChain calls this before each chain invocation to build
        the conversation context to send to the LLM.

        Returns:
            List of BaseMessage objects (HumanMessage, AIMessage, etc.)
            in the order they were added.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                # Named bind variable :sid (Oracle uses :name, not %s like Postgres)
                # This is parameterized — NOT string interpolation — so it's
                # safe from SQL injection. Oracle replaces :sid with the actual value.
                f"SELECT payload FROM {self.table} "
                f"WHERE session_id = :sid ORDER BY seq",
                sid=self.session_id,
            )
            rows = []
            for (payload,) in cur.fetchall():
                # oracledb 4.x may auto-parse IS JSON CLOB columns to dicts.
                # Older driver versions return a LOB object, bytes, or a plain string.
                # We handle all four cases so this works regardless of driver version.
                if isinstance(payload, dict):
                    # Already a dict — oracledb did the JSON parsing for us
                    rows.append(payload)
                    continue
                # LOB object (Oracle Large Object): needs .read() to get the string
                raw = payload.read() if hasattr(payload, "read") else payload
                # If it came back as bytes, decode to string first
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                # Parse the JSON string → Python dict
                rows.append(json.loads(raw))

        # Convert the list of dicts → list of LangChain message objects
        return messages_from_dict(rows)

    def add_message(self, message: BaseMessage) -> None:
        """
        Save one message to Oracle.

        LangChain calls this automatically after the human sends a message
        and after the AI responds. You don't call this directly in your app.

        Args:
            message: Any LangChain message (HumanMessage, AIMessage, etc.)
        """
        # Convert the LangChain message object → JSON string
        # messages_to_dict returns a list, so we take index [0] (just this one message)
        payload = json.dumps(messages_to_dict([message])[0])
        # Example payload: '{"type": "human", "data": {"content": "hi", "type": "human"}}'

        with self.conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table} (session_id, payload) VALUES (:sid, :p)",
                sid=self.session_id,
                p=payload,
                # ↑ Named bind variables — :sid and :p are replaced safely by oracledb.
                #   Note: seq and created_at are NOT listed — Oracle fills them
                #   automatically (GENERATED ALWAYS AS IDENTITY and DEFAULT SYSTIMESTAMP).
            )
        self.conn.commit()
        # ↑ Without commit(), the INSERT is pending in a transaction.
        #   commit() flushes it to disk. If the server crashes before commit(),
        #   the message is lost — acceptable for a chat history.

    def clear(self) -> None:
        """
        Delete all messages for this session.

        Useful for "New Chat" functionality or resetting a conversation.
        Does NOT delete messages from other sessions.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table} WHERE session_id = :sid",
                sid=self.session_id,
            )
        self.conn.commit()


# ---------------------------------------------------------------------------
# DDL helper — run once at startup
# ---------------------------------------------------------------------------

# Plain CREATE TABLE DDL — no PL/SQL wrapper needed.
# oracledb's thin driver doesn't support PL/SQL anonymous blocks via
# cursor.execute(). Instead we catch ORA-00955 at the Python level.
_CREATE_DDL = """\
CREATE TABLE chat_history (
    session_id VARCHAR2(120) NOT NULL,
    seq        NUMBER GENERATED ALWAYS AS IDENTITY,
    payload    CLOB CHECK (payload IS JSON),
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
    PRIMARY KEY (session_id, seq)
)
"""


def init_table(conn: oracledb.Connection) -> None:
    """
    Create the chat_history table if it doesn't already exist.

    Idempotent — safe to call on every app startup. If the table already
    exists (ORA-00955), the error is silently swallowed at the Python level
    (same logic as the PL/SQL EXCEPTION block, just written in Python).

    Call this in adapter.py before serving any requests.
    """
    with conn.cursor() as cur:
        try:
            cur.execute(_CREATE_DDL)
            conn.commit()
            print("[history] chat_history table created.")
        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args else 0
            if error_code == 955:
                # ORA-00955: table already exists — that's fine, nothing to do
                print("[history] chat_history table already exists.")
            else:
                raise


# ---------------------------------------------------------------------------
# LangChain factory helper — used by chain.py
# ---------------------------------------------------------------------------

def get_history_factory(conn: oracledb.Connection):
    """
    Return a callable that creates an OracleChatHistory for a given session_id.

    LangChain's RunnableWithMessageHistory expects a function with signature:
        (session_id: str) -> BaseChatMessageHistory

    Usage in chain.py:
        from pdf_chat.history import get_history_factory
        from pdf_chat.store import get_connection

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_history_factory(get_connection()),   # ← pass the factory here
            input_messages_key="question",
            history_messages_key="history",
        )
    """
    def _factory(session_id: str) -> OracleChatHistory:
        return OracleChatHistory(conn, session_id)
    return _factory
