"""
FastAPI adapter — exposes /v1/chat/completions for Open WebUI.

This is the "front door" of the application. Open WebUI sends chat requests
in OpenAI's API format; this adapter translates them into LangChain chain
calls and streams the response back in OpenAI format.

Architecture:
    Open WebUI (port 3000)
        → POST http://host.docker.internal:8000/v1/chat/completions
        → adapter.py (this file, FastAPI on port 8000)
        → chain.get_chain().stream(...)
        → Oracle (vector search + history) + Groq (generation)
        → SSE stream back to Open WebUI

Run:
    python -m pdf_chat.adapter

Endpoints:
    GET  /                        health check
    GET  /v1/models               list available models (Open WebUI uses this)
    POST /v1/chat/completions     the main chat endpoint (streaming)
    GET  /docs                    auto-generated API docs (FastAPI built-in)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
# ↑ FastAPI: the web framework.
#   HTTPException: raise this to return HTTP errors (400, 404, 500, etc.)

from fastapi.middleware.cors import CORSMiddleware
# ↑ CORS (Cross-Origin Resource Sharing) middleware.
#   Browsers block requests between different origins (e.g. localhost:3000 → localhost:8000)
#   unless the server explicitly allows it. This middleware adds the necessary headers.

from fastapi.responses import StreamingResponse
# ↑ A special FastAPI response type for server-sent events.
#   Instead of sending one JSON blob, it streams text line by line.

from pydantic import BaseModel
# ↑ Pydantic is Python's most popular data validation library.
#   BaseModel: define a class with typed fields, Pydantic validates automatically.
#   FastAPI uses Pydantic for all request/response models.

import uvicorn
# ↑ ASGI server that runs FastAPI. Think of it as the engine that
#   listens on port 8000 and hands requests to FastAPI.

from pdf_chat.chain import get_chain
from pdf_chat.history import init_table
from pdf_chat.store import bootstrap, get_connection

load_dotenv()

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="pdf-chat adapter",
    description="OpenAI-compatible adapter for the pdf-chat RAG system",
    version="0.1.0",
)
# ↑ Creates the FastAPI application instance.
#   title, description, version appear in the auto-docs at http://localhost:8000/docs

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # ↑ Allow requests from any origin.
    #   In production, replace "*" with your specific frontend URL for security.
    #   For local development, "*" is fine.
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],   # allow any request headers (including Authorization)
)


# ---------------------------------------------------------------------------
# Startup: ensure Oracle tables exist before serving any requests
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts, before accepting any requests.

    Creates:
        - PDF_CHAT_DOCUMENTS table (via bootstrap dance)
        - PDF_CHAT_CONVERSATIONS table (via bootstrap dance)
        - chat_history table (via init_table)

    Safe to call even if tables already exist (idempotent).
    """
    print("[adapter] Starting up — bootstrapping Oracle tables...")
    try:
        bootstrap()                         # creates vector store tables
        init_table(get_connection())        # creates chat_history table
        get_chain()                         # pre-warms the chain + LLM client
        print("[adapter] Ready. Listening on http://localhost:8000")
    except Exception as e:
        print(f"[adapter] WARNING: startup error — {e}")
        print("[adapter] Server will start anyway; errors may occur at request time.")
        # Don't raise — let the server start even if Oracle isn't up yet.
        # The first actual request will surface the real error.


# ---------------------------------------------------------------------------
# Pydantic models — define the shape of requests and responses
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """One message in a conversation."""
    role: str
    # ↑ "system", "user", or "assistant"
    content: str
    # ↑ The message text


class ChatCompletionRequest(BaseModel):
    """
    OpenAI /v1/chat/completions request body.
    Open WebUI sends exactly this format.
    """
    model: str = "pdf-chat"
    # ↑ Model name — we accept any value and ignore it (we always use Groq).
    #   The default lets requests omit this field.

    messages: list[ChatMessage]
    # ↑ The conversation so far. At minimum: [{"role": "user", "content": "..."}]

    stream: bool = True
    # ↑ Whether to stream the response. Open WebUI always sends stream=true.

    temperature: float = 0.2
    max_tokens: int = 1024


# ---------------------------------------------------------------------------
# Helper: extract session_id from the request
# ---------------------------------------------------------------------------

def _get_session_id(messages: list[ChatMessage]) -> str:
    """
    Derive a stable session identifier from the conversation.

    Open WebUI doesn't send an explicit session_id. We use the content of the
    first user message as a session key. This is simple and good enough for
    single-user local use — different conversations start with different first
    messages.

    In a multi-user system you'd use a real UUID per user session passed in
    a custom header or cookie.
    """
    for msg in messages:
        if msg.role == "user":
            # Use first 60 chars of first user message as session key
            return msg.content[:60].strip().replace(" ", "_")
    return "default-session"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check():
    """
    Health check endpoint.
    Open WebUI sometimes polls this to verify the backend is alive.
    """
    return {"status": "ok", "service": "pdf-chat-adapter"}


@app.get("/v1/models")
async def list_models():
    """
    Return a fake model list in OpenAI format.

    Open WebUI calls GET /v1/models on startup to populate the model dropdown.
    We return one entry so Open WebUI has something to select.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "pdf-chat",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Main chat endpoint — translates OpenAI format → LangChain chain → SSE stream.

    Open WebUI sends:
        POST /v1/chat/completions
        {"model": "pdf-chat", "messages": [...], "stream": true}

    We respond with a Server-Sent Events stream of:
        data: {"choices": [{"delta": {"content": "word "}, "finish_reason": null}]}
        data: {"choices": [{"delta": {"content": "by "}, "finish_reason": null}]}
        ...
        data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        data: [DONE]
    """
    # Extract the latest user message — this is what we send to the chain
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message in request")
    question = user_messages[-1].content
    # ↑ [-1] gets the LAST element of the list — the most recent user message.
    #   Earlier messages are already in Oracle history, so we only pass the new one.

    session_id = _get_session_id(request.messages)
    chain = get_chain()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    # ↑ A random ID for this completion, following OpenAI's format.
    #   uuid4() generates a random UUID. .hex gives it as a hex string.
    #   [:8] takes just the first 8 characters: "a3f9c2b1"

    async def stream_response() -> AsyncIterator[str]:
        """
        Generator that yields SSE-formatted chunks as the chain produces tokens.

        'async def' + 'yield' = an async generator function.
        FastAPI's StreamingResponse iterates over this and sends each yielded
        string to the client immediately, without buffering.
        """
        try:
            # chain.astream() is the async streaming version of chain.invoke().
            # It yields partial responses (individual tokens or small token groups)
            # as Groq generates them.
            async for chunk in chain.astream(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
                # ↑ The session_id tells RunnableWithMessageHistory which
                #   OracleChatHistory instance to load/save messages to.
            ):
                if not chunk:
                    continue  # skip empty chunks

                # Build an OpenAI-format SSE event
                event_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            # ↑ "delta" = the incremental new content.
                            #   Open WebUI appends each delta to build the full response.
                            "finish_reason": None,
                            # ↑ None means "not done yet". "stop" signals completion.
                        }
                    ],
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                # ↑ SSE format requires "data: " prefix and double newline suffix.
                #   The client's EventSource reads each "data: ..." line as an event.

            # Final chunk — signals stream is complete
            final_event = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                # ↑ Empty delta + "stop" = no more content, generation is done.
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            yield "data: [DONE]\n\n"
            # ↑ OpenAI's official signal that the stream has ended.
            #   Open WebUI stops its loading animation when it receives this.

        except Exception as e:
            # If anything goes wrong mid-stream, send an error event.
            # We can't raise an HTTP exception inside a stream that already started.
            error_event = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": f"\n\n[Error: {e}]"},
                             "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        # ↑ The Content-Type header that tells browsers/clients to treat
        #   this as a Server-Sent Events stream (not regular JSON).
        headers={
            "Cache-Control": "no-cache",
            # ↑ Prevents proxies/browsers from caching the stream.
            "X-Accel-Buffering": "no",
            # ↑ Disables Nginx's response buffering (if this runs behind Nginx).
            #   Without this, Nginx would buffer the whole response before sending it,
            #   completely defeating the purpose of streaming.
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "pdf_chat.adapter:app",
        # ↑ "module:variable" — tells uvicorn where to find the FastAPI app object.
        #   This is the string form of import. Uvicorn will import pdf_chat.adapter
        #   and look for the variable named "app".
        host="0.0.0.0",
        # ↑ Listen on all network interfaces (not just localhost).
        #   Needed so Open WebUI inside Docker can reach the adapter on the host.
        #   "0.0.0.0" = accept connections from anywhere that can reach this machine.
        port=8000,
        reload=False,
        # ↑ reload=True would auto-restart on code changes (good for dev).
        #   We set False here because the RAG chain takes a few seconds to warm up
        #   and constant reloads would be annoying.
        log_level="info",
    )
