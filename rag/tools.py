"""LangChain tool adapter for PDF retrieval."""

from langchain_core.documents import Document
from langchain_core.tools import tool

from rag.retrieval import get_default_retriever

last_retrieved_context: list[Document] = []


@tool
def collecting_info(query: str) -> str:
    """Find the most relevant passages in the indexed PDF knowledge base."""
    global last_retrieved_context
    retriever = get_default_retriever()
    last_retrieved_context = retriever.search(query)
    return retriever.format_context(last_retrieved_context)
