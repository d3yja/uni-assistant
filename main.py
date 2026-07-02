"""
Main graph: 'Greeting/rcv user Q' -> 'Classify intent' -> one chain per
branch -> 'Final answer'.

Each branch is its own file under project/chains/. This file just imports
the node functions and wires up the edges to match the diagram.
"""

from langgraph.graph import END, StateGraph

from project.chains.classify_intent import classify_intent_node, route_from_intent
from project.chains.clarification_chain import clarification_node
from project.chains.human_handoff_chain import human_handoff_node
from project.chains.memory_chain import memory_chain_node
from project.chains.rag_chain import rag_chain_node
from project.chains.simple_response_chain import simple_response_node
from project.llm import model
from project.state import GraphState

# NOTE: pip install langgraph


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

    # Every branch feeds into the 'Final answer' star
    graph.add_edge("simple_response", END)
    graph.add_edge("memory_chain", END)
    graph.add_edge("rag_chain", END)
    graph.add_edge("human_handoff", END)
    graph.add_edge("clarification", END)

    return graph.compile()


if __name__ == "__main__":
    if model is None:
        raise RuntimeError("Set `model` in project/llm.py to a real chat model instance.")

    app = build_graph()

    result = app.invoke({"query": "should i quit university?"})
    print("Intent:", result["intent"])
    print("Answer:", result["answer"])
