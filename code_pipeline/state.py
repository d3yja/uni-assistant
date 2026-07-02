# ============================================================
# state.py
# ============================================================

from typing import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Shared state used throughout the LangGraph workflow.
    """

    # User's original question
    question: str

    # Intent predicted by the classifier
    # greeting | rag | memory | unknown
    intent: str

    # Documents selected by the Document Picker
    selected_documents: list[str]

    # Retrieved document chunks from the RAG pipeline
    retrieved_documents: list[Document]

    # Final/generated answer
    answer: str

    # Whether human review is required
    needs_human_review: bool