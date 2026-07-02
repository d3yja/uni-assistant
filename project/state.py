from typing import List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """State shared across every chain in the graph."""

    query: str                 # the user's raw question
    intent: str                # output of classify_intent: one of INTENTS in config.py

    # -- RAG chain fields --
    selected_docs: List[str]
    context: str
    rag_found: bool            # False when nothing relevant was retrieved

    # -- memory chain fields --
    memory_action: str         # "read" or "write"
    memory_result: str

    answer: Optional[str]      # final answer, set by whichever chain handles the query

    # -- human-in-the-loop fields --
    draft_answer: Optional[str]    # answer proposed before human approval
    human_decision: Optional[str]  # "yes", "no", or "edit"