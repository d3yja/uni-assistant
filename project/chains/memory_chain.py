"""
Memory chain: handles the 'Memory query' branch — reading or writing simple
student info during the session.

Persists to memory/memory.json using the schema from the project spec:

{
  "name": "Ali",
  "program": "BSCS",
  "section": "B",
  "interests": ["AI", "labs"],
  "previous_questions": ["...", "..."]
}

`previous_questions` is appended to on every turn by app.py, not just when
this chain runs — see app.py's main loop.
"""

import json
import os
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..llm import model
from ..state import GraphState

MEMORY_PATH = "./memory/memory.json"

_DEFAULT_MEMORY = {
    "name": None,
    "program": None,
    "section": None,
    "interests": [],
    "previous_questions": [],
}


def load_memory() -> dict:
    if not os.path.exists(MEMORY_PATH):
        return dict(_DEFAULT_MEMORY)
    with open(MEMORY_PATH, "r") as f:
        data = json.load(f)
    # backfill any keys missing from an older/partial file
    return {**_DEFAULT_MEMORY, **data}


def save_memory(data: dict) -> None:
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------
# Structured extraction: is this a recall question, or new info to store?
# --------------------------------------------------------------------------
class _MemoryExtraction(BaseModel):
    is_recall_request: bool = Field(
        description="True if the user is asking what we remember about them, "
        "rather than telling us something new."
    )
    name: Optional[str] = Field(default=None, description="The student's name, if mentioned.")
    program: Optional[str] = Field(default=None, description="The student's program, e.g. BSCS, if mentioned.")
    section: Optional[str] = Field(default=None, description="The student's section, e.g. B, if mentioned.")
    interests: List[str] = Field(
        default_factory=list, description="Any interests the student mentions, e.g. AI, labs."
    )


_extract_prompt = ChatPromptTemplate.from_template(
    """Determine whether the user is asking you to recall what you know about
them, or telling you new information about themselves to remember. Extract
any of: name, program, section, interests that are mentioned.

User message: {query}
"""
)

_extract_chain = _extract_prompt | model.with_structured_output(_MemoryExtraction)


def _describe_memory(mem: dict) -> str:
    parts = []
    if mem.get("name"):
        parts.append(f"your name is {mem['name']}")
    if mem.get("program"):
        parts.append(f"you're in {mem['program']}")
    if mem.get("section"):
        parts.append(f"section {mem['section']}")
    if mem.get("interests"):
        parts.append(f"you're interested in {', '.join(mem['interests'])}")

    if not parts:
        return "I don't have anything stored about you yet."
    return "You told me that " + ", ".join(parts) + "."


def memory_chain_node(state: GraphState) -> GraphState:
    query = state["query"]
    memory = load_memory()

    result: _MemoryExtraction = _extract_chain.invoke({"query": query})

    if result.is_recall_request:
        answer = _describe_memory(memory)
        return {**state, "memory_action": "read", "answer": answer}

    # write: merge any newly-mentioned fields into memory
    if result.name:
        memory["name"] = result.name
    if result.program:
        memory["program"] = result.program
    if result.section:
        memory["section"] = result.section
    if result.interests:
        existing = set(memory.get("interests", []))
        memory["interests"] = list(existing.union(result.interests))

    save_memory(memory)

    answer = "Got it, I'll remember that."
    return {**state, "memory_action": "write", "answer": answer}