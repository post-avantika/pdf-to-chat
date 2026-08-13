"""
pdf_chat package.

A local RAG chatbot application over PDF documents using:
- Oracle Database 26ai (AI Vector Search & persistent chat history)
- sentence-transformers (MiniLM-L6-v2 local embeddings)
- LangChain (retrieval chain & message history orchestration)
- Groq / OpenAI-compatible API (LLM generation)
- FastAPI (adapter exposing /v1/chat/completions to Open WebUI)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
