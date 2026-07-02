"""
Main graph: 'Greeting/rcv user Q' -> 'Classify intent' -> one chain per
branch -> 'Final answer'.

Each branch is its own file under project/chains/. This file just imports
the node functions and wires up the edges to match the diagram.
"""

from langgraph.graph import END, StateGraph

from .chains.classify_intent import classify_intent_node, route_from_intent
from .chains.clarification_chain import clarification_node
from .chains.human_handoff_chain import human_handoff_node
from .chains.memory_chain import memory_chain_node
from .chains.rag_chain import rag_chain_node
from .chains.simple_response_chain import simple_response_node
from .state import GraphState

# NOTE: pip install langgraph


def _route_from_rag(state: GraphState) -> str:
    """After the RAG chain runs: if nothing relevant was found, escalate to
    human review instead of showing the 'not found' message directly."""
    return "human_handoff" if not state.get("rag_found", True) else "end"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("simple_response", simple_response_node)
    graph.add_node("memory_chain", memory_chain_node)
    graph.add_node("rag_chain", rag_chain_node)
    graph.add_node("human_handoff", human_handoff_node)
    graph.add_node("clarification", clarification_node)

    graph.set_entry_point("classify_intent")

    # Matches the diamond's branches in the diagram
    graph.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {
            "general_greeting": "simple_response",
            "memory_query": "memory_chain",
            "doc_query": "rag_chain",
            "human_needed": "human_handoff",
            "unknown": "clarification",
        },
    )

    # RAG chain either ends normally, or escalates to human review when
    # nothing relevant was found in the knowledge base
    graph.add_conditional_edges(
        "rag_chain",
        _route_from_rag,
        {
            "human_handoff": "human_handoff",
            "end": END,
        },
    )

    # Every other branch feeds straight into the 'Final answer' star
    graph.add_edge("simple_response", END)
    graph.add_edge("memory_chain", END)
    graph.add_edge("human_handoff", END)
    graph.add_edge("clarification", END)

    return graph.compile()