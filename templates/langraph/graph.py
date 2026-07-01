# ============================================================
# graph.py
# ============================================================

from langgraph.graph import StateGraph, START, END

from state import GraphState

from nodes import (
    greeting_node,
    classify_intent,
    intent_router,
    document_picker,
    retrieve_documents_node,
    rag_router,
    generate_answer_node,
    clarification_node,
    memory_node,
    human_review_node,
    final_node,
)

# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(GraphState)

# ------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------

builder.add_node(
    "intent_classifier",
    classify_intent
)

builder.add_node(
    "greeting",
    greeting_node
)

builder.add_node(
    "memory",
    memory_node
)

builder.add_node(
    "document_picker",
    document_picker
)

builder.add_node(
    "retrieve_documents",
    retrieve_documents_node
)

builder.add_node(
    "generate_answer",
    generate_answer_node
)

builder.add_node(
    "clarification",
    clarification_node
)

builder.add_node(
    "human_review",
    human_review_node
)

builder.add_node(
    "final_answer",
    final_node
)

# ============================================================
# Edges
# ============================================================

# START
builder.add_edge(
    START,
    "intent_classifier"
)

# ------------------------------------------------------------
# Intent Router
# ------------------------------------------------------------

builder.add_conditional_edges(
    "intent_classifier",
    intent_router,
    {
        "greeting": "greeting",

        "memory": "memory",

        "human": "human_review",

        "rag": "document_picker",
    },
)

# ------------------------------------------------------------
# Greeting
# ------------------------------------------------------------

builder.add_edge(
    "greeting",
    "final_answer"
)

# ------------------------------------------------------------
# Memory
# ------------------------------------------------------------

builder.add_edge(
    "memory",
    "final_answer"
)

# ------------------------------------------------------------
# RAG
# ------------------------------------------------------------

builder.add_edge(
    "document_picker",
    "retrieve_documents"
)

builder.add_conditional_edges(
    "retrieve_documents",
    rag_router,
    {
        "documents_found": "generate_answer",
        "no_documents": "clarification",
    },
)

builder.add_edge(
    "generate_answer",
    "human_review"
)

builder.add_edge(
    "clarification",
    "final_answer"
)

# ------------------------------------------------------------
# Human Review
# ------------------------------------------------------------

builder.add_edge(
    "human_review",
    "final_answer"
)

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

builder.add_edge(
    "final_answer",
    END
)

# ============================================================
# Compile Graph
# ============================================================

graph = builder.compile()

# ============================================================
# Optional Graph Visualization
# ============================================================

if __name__ == "__main__":

    print(graph.get_graph().draw_ascii())

    # If you have pygraphviz installed:
    #
    # png = graph.get_graph().draw_mermaid_png()
    #
    # with open("graph.png", "wb") as f:
    #     f.write(png)