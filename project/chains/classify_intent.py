from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..config import INTENTS
from ..llm import model
from ..state import GraphState


class IntentClassification(BaseModel):
    intent: Literal[
        "general_greeting", "memory_query", "doc_query", "human_needed", "unknown"
    ] = Field(description="The single best-matching intent for the user's message.")


_prompt = ChatPromptTemplate.from_template(
    """Classify the user's message into exactly one of the following intents:

{intent_list}

User message: {query}
"""
)

_chain = _prompt | model.with_structured_output(IntentClassification)


def classify_intent_node(state: GraphState) -> GraphState:
    intent_list = "\n".join(f"- {name}: {desc}" for name, desc in INTENTS.items())
    result: IntentClassification = _chain.invoke(
        {"intent_list": intent_list, "query": state["query"]}
    )
    return {**state, "intent": result.intent}


def route_from_intent(state: GraphState) -> str:
    """Used as the conditional edge function out of classify_intent."""
    return state["intent"]
