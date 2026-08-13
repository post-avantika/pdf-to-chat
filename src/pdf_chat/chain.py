"""
RAG chain for pdf_chat.

This is the "brain" of the application. It:
    1. Takes a user question
    2. Retrieves the 5 most relevant PDF chunks from Oracle (vector search)
    3. Formats the chunks as a "context" block with citation labels
    4. Sends [system prompt + history + context + question] to Groq
    5. Returns the answer (with citations like [report.pdf:p.3])

The chain is wrapped in RunnableWithMessageHistory so that:
    - Past messages are loaded from Oracle before each call
    - The new question + AI answer are saved to Oracle after each call

Usage (in adapter.py):
    from pdf_chat.chain import get_chain

    chain = get_chain()
    answer = chain.invoke(
        {"question": "What is the dosage of aspirin?"},
        config={"configurable": {"session_id": "user-abc-123"}},
    )
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
# ↑ StrOutputParser: takes the LLM's output (an AIMessage object) and extracts
#   just the text content as a plain string. The last step in our chain.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# ↑ ChatPromptTemplate: builds a structured prompt from multiple parts
#       (system message, history placeholder, human message).
#   MessagesPlaceholder: a special slot in the template that gets filled
#       with a list of past messages (the conversation history).

from langchain_core.runnables import RunnableLambda, RunnableParallel
# ↑ RunnableLambda:    wraps any Python function as a LangChain Runnable step.
#   RunnableParallel:  runs multiple steps at the same time, combining outputs.

from langchain_core.runnables.history import RunnableWithMessageHistory
# ↑ The wrapper that adds automatic message history loading/saving to any chain.

from langchain_openai import ChatOpenAI
# ↑ LangChain's wrapper for any OpenAI-compatible chat API.
#   We point it at Groq instead of OpenAI by changing base_url and api_key.
#   It implements LangChain's BaseChatModel interface — plug-and-play.

from pdf_chat.history import get_history_factory
from pdf_chat.store import get_connection, get_store

load_dotenv()

# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: list) -> str:
    """
    Format retrieved Oracle documents into a numbered context block.

    Each document gets a citation label based on its metadata.
    The label format is [filename:p.N] so the LLM can include it in answers.

    Args:
        docs: List of LangChain Document objects returned by OracleVS.
              Each doc has .page_content (str) and .metadata (dict).

    Returns:
        A formatted string like:
            [1] [report.pdf:p.3]
            The dosage of aspirin is 500mg for adults...

            [2] [manual.pdf:p.12]
            Aspirin should not be taken on an empty stomach...

    Example usage by the LLM in its answer:
        "The recommended dosage is 500mg [report.pdf:p.3]."
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        # Extract metadata — these were set during ingestion in ingest.py
        meta = doc.metadata
        filename = meta.get("filename", "unknown")
        page = meta.get("page", "?")
        citation = f"[{filename}:p.{page}]"
        # e.g. "[aspirin_guide.pdf:p.3]"

        parts.append(f"[{i}] {citation}\n{doc.page_content.strip()}")
        # e.g. "[1] [aspirin_guide.pdf:p.3]\nThe dosage is 500mg for adults..."

    if not parts:
        return "No relevant documents found in the knowledge base."
        # ↑ Fallback if Oracle returned no results (empty store, or no similar chunks).
        #   The LLM will see this and should respond that it doesn't have information.

    return "\n\n".join(parts)
    # ↑ Join all numbered sections with a blank line between them.


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

_llm: Any = None

def get_llm() -> ChatOpenAI:
    """
    Return the LangChain Groq/OCI LLM instance (lazy singleton).

    Uses ChatOpenAI pointed at Groq's base URL. This works because Groq's
    API is 100% OpenAI-compatible — same request format, same response format.
    The only difference is the base_url and api_key.
    """
    global _llm
    if _llm is not None:
        return _llm

    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OCI_GENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY in your .env file."
        )

    base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai")
    # ↑ Default to Groq. Override via LLM_BASE_URL in .env.
    #   The OpenAI client appends /v1 automatically, so don't add it here.

    # Ensure base_url ends with /v1 (ChatOpenAI expects the full versioned path)
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    # ↑ Default Groq model. Override via LLM_MODEL in .env.
    #   Other options: llama-3.1-8b-instant (faster), mixtral-8x7b-32768 (larger context)

    _llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        # ↑ ChatOpenAI uses these two params to talk to any OpenAI-compatible endpoint.
        temperature=0.2,
        # ↑ Controls randomness: 0.0 = deterministic (always same answer for same input)
        #                         1.0 = creative/random
        #   0.2 is a good balance for factual RAG — mostly deterministic, slight variation.
        max_tokens=1024,
        # ↑ Maximum tokens in the response. 1024 ≈ ~750 words.
        #   Increase if you need longer answers.
        streaming=True,
        # ↑ Stream the response token-by-token instead of waiting for the full answer.
        #   This makes the UI feel faster — words appear as they're generated.
        #   Our adapter.py will forward the stream to Open WebUI.
    )
    return _llm


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using ONLY the provided \
PDF documents. 

Rules:
1. Base your answer EXCLUSIVELY on the numbered context sections provided below.
2. After each claim, add the citation in square brackets, e.g. [report.pdf:p.3].
3. If the answer cannot be found in the provided context, say:
   "I don't have information about that in the uploaded documents."
   Do NOT make up information.
4. If multiple documents support a claim, cite all of them.
5. Be concise and direct. Avoid repeating the question back.

Context from uploaded PDFs:
{context}
"""
# ↑ This prompt is the "system message" — it sets the AI's role and rules.
#   {context} is a placeholder filled at runtime with the Oracle-retrieved chunks.
#
# Why strict citation rules?
#   Without them, LLMs sometimes "hallucinate" — confidently state things that
#   aren't in your documents. Explicit rules + citations keep the AI grounded.


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------

_chain: Any = None

def get_chain():
    """
    Build and return the RAG chain with persistent message history.

    Returns a RunnableWithMessageHistory that:
        - Accepts: {"question": str}  with config={"configurable": {"session_id": str}}
        - Returns: str (the answer text, with citations)

    The chain is built once and cached.
    """
    global _chain
    if _chain is not None:
        return _chain

    # 1. Retriever — fetches top-5 similar chunks from Oracle
    retriever = get_store("DOCUMENTS").as_retriever(
        search_kwargs={"k": 5}
        # ↑ k=5: return the 5 most similar chunks.
        #   More chunks = more context but longer prompt and slower response.
        #   5 is a good default for single-topic questions.
    )
    # as_retriever() wraps OracleVS in LangChain's BaseRetriever interface.
    # When called with a string query, it internally runs:
    #   VECTOR_DISTANCE(embedding, embed(query), COSINE) ORDER BY ... FETCH FIRST 5

    # 2. Build the pipeline that goes from {question} → retrieved docs → formatted context
    #
    #    RunnableParallel runs both branches simultaneously:
    #      Branch A: retrieves docs AND formats them into context string
    #      Branch B: passes the question through unchanged
    #
    #    Output: {"context": "formatted text", "question": "original question", "history": [...]}
    #
    #    Note: "history" is NOT in RunnableParallel — RunnableWithMessageHistory
    #    injects it automatically before invoking the chain.

    def retrieve_and_format(inputs: dict) -> str:
        """Retrieve docs for the question and format them with citations."""
        question = inputs["question"]
        docs = retriever.invoke(question)
        return _format_docs(docs)
    # ↑ This function becomes one branch of RunnableParallel below.
    #   It takes the full inputs dict, extracts "question", retrieves, formats.

    context_step = RunnableLambda(retrieve_and_format)
    # ↑ RunnableLambda wraps a regular Python function as a LangChain Runnable.
    #   Now `context_step` can be used inside chains with | (pipe) operator.

    # 3. Build the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        # ↑ The system message with {context} placeholder.
        #   LangChain fills {context} from the chain's inputs automatically.

        MessagesPlaceholder(variable_name="history"),
        # ↑ This slot gets filled with the list of past messages (HumanMessage, AIMessage)
        #   loaded from Oracle by RunnableWithMessageHistory before each call.

        ("human", "{question}"),
        # ↑ The current user question — filled from the chain's "question" input key.
    ])

    # 4. Output parser — extracts plain text from the LLM's AIMessage response
    output_parser = StrOutputParser()
    # ↑ The LLM returns an AIMessage(content="...") object.
    #   StrOutputParser.parse() extracts just the content string.
    #   Without this, the chain would return an AIMessage object instead of a string.

    # 5. Assemble the full chain using LCEL (pipe operator)
    #
    #    Input dict: {"question": "...", "context": "...", "history": [...]}
    #    ↓
    #    prompt  →  fills template with all three values
    #    ↓
    #    llm     →  sends filled prompt to Groq, returns AIMessage
    #    ↓
    #    output_parser  →  extracts text string from AIMessage
    #
    raw_chain = (
        RunnableParallel(
            context=context_step,
            # ↑ Runs retrieve_and_format(inputs) → fills "context" key
            question=RunnableLambda(lambda x: x["question"]),
            # ↑ Passes through the question unchanged → fills "question" key
            history=RunnableLambda(lambda x: x.get("history", [])),
            # ↑ CRITICAL: RunnableWithMessageHistory injects "history" into the
            #   input dict BEFORE this chain runs. RunnableParallel must pass it
            #   through explicitly, otherwise it gets dropped and the prompt
            #   raises "missing variables {'history'}".
        )
        | prompt
        | get_llm()
        | output_parser
    )

    # 6. Wrap with message history (loads/saves from Oracle automatically)
    _chain = RunnableWithMessageHistory(
        raw_chain,
        get_history_factory(get_connection()),
        # ↑ The factory function: given a session_id string, returns an
        #   OracleChatHistory instance for that session.
        #   LangChain calls this before each invoke() to get the history object.

        input_messages_key="question",
        # ↑ Which key in the input dict is the new human message.
        #   This message gets saved to Oracle after the chain runs.

        history_messages_key="history",
        # ↑ Which key in the prompt template receives the history list.
        #   LangChain injects the loaded messages here automatically.
    )

    return _chain
