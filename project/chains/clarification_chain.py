"""Clarification chain: handles the 'Unknown' branch — 'Ask for clarification'."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..llm import model
from ..state import GraphState

_prompt = ChatPromptTemplate.from_template(
    """The user's intent wasn't clear. Ask a short, friendly clarifying
question that helps narrow down whether they're asking about the academic
calendar, grading, attendance, hostel rules, contact info, or something
else entirely.

User message: {query}
"""
)

_chain = _prompt | model | StrOutputParser()


def clarification_node(state: GraphState) -> GraphState:
    answer = _chain.invoke({"query": state["query"]})
    return {**state, "answer": answer}
