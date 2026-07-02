"""Simple response chain: handles the 'General greeting' branch."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..llm import model
from ..state import GraphState

_prompt = ChatPromptTemplate.from_template(
    """The user sent a general greeting or small talk. Reply briefly and
warmly, and let them know you can help with academic calendar, grading,
attendance, hostel rules, or contact info questions.

User message: {query}
"""
)

_chain = _prompt | model | StrOutputParser()


def simple_response_node(state: GraphState) -> GraphState:
    answer = _chain.invoke({"query": state["query"]})
    return {**state, "answer": answer}
