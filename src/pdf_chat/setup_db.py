"""
setup_db.py — One-time Oracle database setup for pdf-chat.

What this script does:
    1. Connects to Oracle as SYSTEM (the admin user)
    2. Creates the PDF_CHAT application user with DEFAULT TABLESPACE USERS
    3. Grants the minimum required privileges to PDF_CHAT
    4. Grants EXECUTE on SYS.DBMS_VECTOR (needed for vector operations)
    5. Smoke-tests the new app user connection

Run ONCE after first `docker compose up -d`:
    python -m pdf_chat.setup_db

Safe to re-run — if the user already exists, it is dropped and recreated cleanly.

Why a separate app user (not SYSTEM)?
    • Security: PDF_CHAT can only access its own tables — not system tables.
    • ASSM: Oracle's JSON validation (CHECK payload IS JSON) requires Automatic
      Segment Space Management. The SYSTEM tablespace lacks ASSM → ORA-43853.
      The USERS tablespace has ASSM → works correctly.
    • Best practice: never run application code as a database admin user.
"""

from __future__ import annotations

import os
import sys

import oracledb
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(msg: str) -> None:
    print(f"  ✅  {msg}")

def fail(msg: str, detail: str = "") -> None:
    print(f"\n  ❌  {msg}")
    if detail:
        print(f"      {detail}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Connect as SYSTEM (admin)
# ---------------------------------------------------------------------------

def connect_as_system() -> oracledb.Connection:
    """
    Open a connection to Oracle using the SYSTEM admin account.

    SYSTEM is Oracle's built-in DBA account. We use it only here to
    create the app user — nowhere else in the application.

    Reads SYS_USER and SYS_PASSWORD from .env (both written by initial setup).
    """
    print("\n[1/4] Connecting to Oracle as SYSTEM...")
    try:
        conn = oracledb.connect(
            user=os.environ.get("SYS_USER", "SYSTEM"),
            password=os.environ["SYS_PASSWORD"],
            # ↑ SYS_PASSWORD = same value as ORACLE_PWD in .env.
            #   Oracle sets the same password for SYSTEM, SYS, and PDBADMIN
            #   from the ORACLE_PWD env var when the container first boots.
            dsn=os.environ["DB_DSN"],
            # ↑ "localhost:1521/FREEPDB1"
            #   We connect to FREEPDB1 (the pluggable database), not the CDB root.
            #   App users are created inside FREEPDB1.
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 'ok' FROM dual")
            assert cur.fetchone()[0] == "ok"
        ok(f"Connected as SYSTEM to {os.environ['DB_DSN']}")
        return conn
    except KeyError as e:
        fail(f"Missing env var: {e}", "Check SYS_PASSWORD and DB_DSN in your .env file.")
    except oracledb.DatabaseError as e:
        fail(
            f"Cannot connect as SYSTEM: {e}",
            "Is Oracle running and healthy?\n"
            "        Check: docker compose ps\n"
            "        Logs:  docker compose logs oracle | tail -30"
        )


# ---------------------------------------------------------------------------
# Step 2: Create the app user
# ---------------------------------------------------------------------------

def create_app_user(conn: oracledb.Connection) -> None:
    """
    Drop (if exists) and recreate the PDF_CHAT application user.

    Why DROP + CREATE instead of just CREATE?
        Idempotency — if the user already exists with wrong settings,
        we start clean. Safe because this is a dev-only local database.

    Why DEFAULT TABLESPACE USERS?
        The USERS tablespace uses Automatic Segment Space Management (ASSM).
        Oracle's CHECK (payload IS JSON) constraint requires ASSM.
        Without it, creating chat_history would fail with ORA-43853.
    """
    app_user = os.environ.get("DB_USER", "PDF_CHAT").upper()
    app_pwd = os.environ["DB_PASSWORD"]

    print(f"\n[2/4] Creating app user: {app_user}...")

    with conn.cursor() as cur:

        # Drop the user if it already exists (idempotent)
        try:
            cur.execute(f"DROP USER {app_user} CASCADE")
            # ↑ CASCADE = also drops all tables, indexes, and objects owned by this user.
            #   Without CASCADE, DROP USER fails if the user owns any objects.
            print(f"      (Dropped existing {app_user} user — recreating cleanly)")
        except oracledb.DatabaseError as e:
            if "ORA-01918" in str(e):
                # ORA-01918: user does not exist — that's fine, nothing to drop
                pass
            else:
                raise

        # Create the app user with correct tablespace
        cur.execute(
            f'CREATE USER {app_user} IDENTIFIED BY "{app_pwd}" '
            f"DEFAULT TABLESPACE USERS "
            # ↑ All tables created by PDF_CHAT go into USERS tablespace (has ASSM)
            f"QUOTA UNLIMITED ON USERS"
            # ↑ No disk quota limit — for a local dev DB, unlimited is fine.
            #   In production you'd set a realistic limit (e.g. QUOTA 5G ON USERS).
        )

        # Grant minimum required privileges
        cur.execute(
            f"GRANT CONNECT, RESOURCE, CREATE SESSION, "
            f"CREATE TABLE, CREATE VIEW, CREATE PROCEDURE "
            f"TO {app_user}"
            # ↑ CONNECT:           allows connecting to the database
            #   RESOURCE:           allows creating tables, sequences, triggers
            #   CREATE SESSION:     explicitly allows opening a session (belt+suspenders)
            #   CREATE TABLE/etc.:  needed to create vector store tables and chat_history
        )

        cur.execute(f"GRANT CREATE MINING MODEL TO {app_user}")
        # ↑ Needed for registering ONNX models (used in the intermediate tier).
        #   Harmless to grant here even though beginner tier doesn't use it.

    conn.commit()
    ok(f"Created user {app_user} with DEFAULT TABLESPACE USERS")


# ---------------------------------------------------------------------------
# Step 3: Grant DBMS_VECTOR execute (requires SYSDBA)
# ---------------------------------------------------------------------------

def grant_vector_privileges(dsn: str, sys_pwd: str, app_user: str) -> None:
    """
    Grant EXECUTE on SYS.DBMS_VECTOR to the app user.

    This grant requires SYSDBA mode — a normal SYSTEM connection isn't
    privileged enough in Oracle 26ai Free to grant SYS package execution.

    DBMS_VECTOR is Oracle's built-in package for vector operations.
    Even though beginner tier doesn't call it directly, langchain-oracledb
    may use it internally for OracleVS operations.
    """
    print(f"\n[3/4] Granting vector privileges (requires SYSDBA)...")
    try:
        sysdba_conn = oracledb.connect(
            user="SYS",
            password=sys_pwd,
            dsn=dsn,
            mode=oracledb.AUTH_MODE_SYSDBA,
            # ↑ SYSDBA mode = Oracle's highest privilege level.
            #   It's like sudo — only needed for system-level operations.
            #   After this setup script, nothing connects as SYSDBA again.
        )
        with sysdba_conn.cursor() as cur:
            try:
                cur.execute(
                    f"GRANT EXECUTE ON SYS.DBMS_VECTOR TO {app_user}"
                )
                sysdba_conn.commit()
                ok(f"Granted EXECUTE ON SYS.DBMS_VECTOR to {app_user}")
            except oracledb.DatabaseError as e:
                if "ORA-01720" in str(e) or "ORA-04042" in str(e):
                    # DBMS_VECTOR might not exist in all Oracle Free versions
                    print(f"      ⚠️  SYS.DBMS_VECTOR not available — skipping (OK for beginner tier)")
                else:
                    raise
        sysdba_conn.close()
    except oracledb.DatabaseError as e:
        # SYSDBA connection can fail in some Oracle Docker configs
        # It's not fatal for the beginner tier — log and continue
        print(f"      ⚠️  SYSDBA grant skipped: {e}")
        print(f"         (Not required for beginner tier — continuing)")


# ---------------------------------------------------------------------------
# Step 4: Smoke test the app user
# ---------------------------------------------------------------------------

def smoke_test_app_user() -> None:
    """
    Connect as the new PDF_CHAT user and verify it works.

    This is the connection that ALL other modules (store.py, history.py,
    ingest.py, adapter.py) will use at runtime. If this works, everything
    will work.
    """
    app_user = os.environ.get("DB_USER", "PDF_CHAT").upper()
    print(f"\n[4/4] Smoke-testing {app_user} connection...")

    try:
        conn = oracledb.connect(
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            dsn=os.environ["DB_DSN"],
        )
        with conn.cursor() as cur:
            cur.execute("SELECT USER FROM dual")
            # ↑ SELECT USER returns the name of the currently connected user.
            #   Verifies we're connected as PDF_CHAT, not SYSTEM.
            current_user = cur.fetchone()[0]
            assert current_user.upper() == app_user, \
                f"Expected {app_user}, connected as {current_user}"

            # Also verify we can create a table (tests RESOURCE grant)
            cur.execute("""
                BEGIN
                    EXECUTE IMMEDIATE 'CREATE TABLE setup_test (id NUMBER)';
                EXCEPTION WHEN OTHERS THEN
                    IF SQLCODE != -955 THEN RAISE; END IF;
                END;
            """)
            cur.execute("DROP TABLE setup_test")
            # ↑ Create and immediately drop a test table.
            #   Verifies: RESOURCE grant, USERS tablespace access, DDL works.

        conn.commit()
        conn.close()
        ok(f"{app_user} can connect and create tables in USERS tablespace")

    except Exception as e:
        fail(
            f"App user smoke test failed: {e}",
            "The user was created but can't connect or create tables.\n"
            "        Check DB_USER and DB_PASSWORD in .env match what was just created."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("pdf-chat — Oracle Database Setup")
    print("=" * 60)
    print("\nThis script creates the PDF_CHAT Oracle user.")
    print("Run once after: docker compose up -d\n")

    # Check required env vars upfront
    for key in ["SYS_PASSWORD", "DB_USER", "DB_PASSWORD", "DB_DSN"]:
        if not os.environ.get(key):
            fail(f"Missing required env var: {key}",
                 "Make sure .env is filled in correctly.")

    app_user = os.environ.get("DB_USER", "PDF_CHAT").upper()
    sys_pwd = os.environ["SYS_PASSWORD"]
    dsn = os.environ["DB_DSN"]

    conn = connect_as_system()               # Step 1
    create_app_user(conn)                    # Step 2
    conn.close()                             # close SYSTEM connection

    grant_vector_privileges(dsn, sys_pwd, app_user)  # Step 3 (SYSDBA)
    smoke_test_app_user()                    # Step 4

    print(f"\n{'=' * 60}")
    print(f"oracle-db-setup: OK")
    print(f"  user:   {app_user}")
    print(f"  dsn:    {dsn}")
    print(f"  next:   python verify.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
