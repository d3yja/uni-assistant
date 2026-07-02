# ============================================================
# state.py
# ============================================================

from typing import TypedDict


class GraphState(TypedDict):
    """
    Shared state used throughout the LangGraph workflow.
    """

    # User's original question
    question: str

    # Intent predicted by the classifier
    # greeting | rag | memory | unknown
    intent: str

    # Documents selected by the document picker
    selected_documents: list[str]

    # Retrieved chunks from the RAG pipeline
    retrieved_documents: list[str]

    # Final/generated answer
    answer: str

    # Whether human review is required
    needs_human_review: bool