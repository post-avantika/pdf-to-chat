-- =============================================================================
-- Migration: 001_chat_history.sql
-- Purpose: Creates the chat_history table in Oracle 26ai to store conversation
--          transcripts for LangChain's RunnableWithMessageHistory.
--
-- Why PL/SQL anonymous block (BEGIN ... EXCEPTION ... END)?
-- Oracle SQL does NOT have a native "CREATE TABLE IF NOT EXISTS" syntax.
-- To make this script idempotent (safe to run multiple times without error),
-- we wrap the DDL in a PL/SQL block and catch error ORA-00955 ("name is already
-- used by an existing object").
-- =============================================================================

BEGIN
    EXECUTE IMMEDIATE q'[
        CREATE TABLE chat_history (
            session_id VARCHAR2(120) NOT NULL,
            seq        NUMBER GENERATED ALWAYS AS IDENTITY,
            payload    CLOB CHECK (payload IS JSON),
            created_at TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            PRIMARY KEY (session_id, seq)
        )
    ]';
EXCEPTION
    WHEN OTHERS THEN
        -- ORA-00955 = table or view already exists.
        -- If it's already there, swallow the error and succeed silently.
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;
/
