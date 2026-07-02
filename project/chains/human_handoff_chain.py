"""
Human handoff / review chain: handles the 'HUman needed' branch —
'Interrupt & reply w/ appropriate response'.

Per the project spec, this drafts an answer, then pauses for a human to
approve, reject, or edit it via a terminal prompt before it's shown to the
student. Maps to the spec's `human_review` node.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..llm import model
from ..state import GraphState

_draft_prompt = ChatPromptTemplate.from_template(
    """The user's request needs human attention (e.g. a complaint, an appeal,
approval for something, or a personal academic decision). Write a short,
factual draft reply based only on general policy — don't invent specific
commitments, names, or timelines. This draft will be reviewed by a human
before it's sent.

User message: {query}
"""
)

_draft_chain = _draft_prompt | model | StrOutputParser()

_REJECTED_MESSAGE = (
    "This request needs review by a human advisor before I can answer it. "
    "Someone from the relevant office will follow up with you directly."
)


def _prompt_for_approval(query: str, draft: str) -> tuple[str, str]:
    """Runs the terminal yes/no/edit prompt. Returns (final_answer, decision)."""
    print(f"\nUser: {query}")
    print(f"\nAssistant draft:\n{draft}")
    print("\nHuman approval required.")

    decision = input("Approve this answer? yes/no/edit: ").strip().lower()

    if decision == "yes":
        return draft, "yes"
    elif decision == "edit":
        edited = input("Enter the edited answer: ").strip()
        return edited, "edit"
    else:
        # anything else, including "no", is treated as a rejection
        return _REJECTED_MESSAGE, "no"


def human_handoff_node(state: GraphState) -> GraphState:
    query = state["query"]
    draft = _draft_chain.invoke({"query": query})

    final_answer, decision = _prompt_for_approval(query, draft)

    # TODO: also trigger your real escalation path here (ticket, notification, etc)
    # whenever decision == "no", since that means a human still needs to act.

    return {
        **state,
        "draft_answer": draft,
        "human_decision": decision,
        "answer": final_answer,
    }