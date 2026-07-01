#importations
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import getpass
import os
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState
from human_review import human_review


#API Key
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

#importing chatgroq
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class GraphState(TypedDict):
    question: str
    intent: str
    selected_documents: list[str]
    retrieved_documents: list[str]
    answer: str


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# GREETING NODE
# ============================================================

def greeting_node(state: GraphState):

    return {
        "answer": (
            "Hello! 👋\n\n"
            "I'm your University Assistant.\n"
            "How can I help you today?"
        )
    }


# ============================================================
# INTENT CLASSIFIER
# ============================================================

intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an intent classifier.

Classify the user's question into EXACTLY ONE category.

Available categories:

greeting
rag
memory
human
unknown

Definitions

greeting
- hello
- hi
- hey
- good morning
- good evening

rag
Questions about university policies,
calendar,
grading,
attendance,
hostel,
labs,
contacts,
or any factual information that should be answered from documents.

memory
Questions involving saving,
remembering,
or recalling information.

human
Questions that explicitly ask for a human,
administrator,
staff member,
or require manual review.

Return ONLY ONE WORD.

No punctuation.
No explanation.
"""
        ),
        (
            "human",
            "{question}"
        ),
    ]
)


def classify_intent(state: GraphState):

    question = state["question"]

    chain = intent_prompt | llm

    result = chain.invoke(
        {
            "question": question
        }
    )

    intent = result.content.strip().lower()

    valid = [
    "greeting",
    "rag",
    "memory",
    "human",
    "unknown"
]

    if intent not in valid:
        intent = "unknown"

    return {
        "intent": intent
    }


# ============================================================
# ROUTER
# ============================================================

def intent_router(state: GraphState):

    return state["intent"]


# ============================================================
# DOCUMENT PICKER
# ============================================================

document_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You choose which university documents are required.

Available documents:

academic_calendar.md
grading_policy.md
attendance_policy.md
hostel_rules.md
lab_guidelines.md
contact_directory.md

Rules

Return ONLY filenames.

One filename per line.

No bullets.

No numbering.

If multiple documents are needed,
return multiple filenames.

Example

attendance_policy.md
grading_policy.md
"""
        ),
        (
            "human",
            "{question}"
        ),
    ]
)


def document_picker(state: GraphState):

    question = state["question"]

    chain = document_prompt | llm

    result = chain.invoke(
        {
            "question": question
        }
    )

    docs = []

    for line in result.content.splitlines():

        line = line.strip()

        if line:

            docs.append(line)

    return {
        "selected_documents": docs
    }


# ============================================================
# MEMORY NODE
# ============================================================

def memory_node(state: GraphState):

    """
    Placeholder.

    Later this node can call your memory.py file.

    Example:

    from memory import save_memory

    save_memory(...)
    """

    return {
        "answer": "Memory operation completed."
    }


# These will be implemented later
from rag_pipeline import (
    retrieve_documents,
    generate_rag_answer,
)

# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents_node(state: GraphState):

    """
    Retrieves document chunks from the selected documents.

    Placeholder:
    Calls your future RAG retrieval pipeline.
    """

    question = state["question"]

    selected_docs = state["selected_documents"]

    retrieved_docs = retrieve_documents(
        question=question,
        selected_documents=selected_docs,
    )

    return {
        "retrieved_documents": retrieved_docs
    }


# ============================================================
# RAG ROUTER
# ============================================================

def rag_router(state: GraphState):

    docs = state.get("retrieved_documents", [])

    if docs:
        return "documents_found"

    return "no_documents"


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer_node(state: GraphState):

    """
    Calls your future RAG answer generator.
    """

    answer = generate_rag_answer(

        question=state["question"],

        retrieved_documents=state["retrieved_documents"]

    )

    return {
        "answer": answer
    }


# ============================================================
# CLARIFICATION
# ============================================================

def clarification_node(state: GraphState):

    return {
        "answer":
        (
            "I couldn't find enough information to answer your question.\n\n"
            "Could you rephrase it or provide a little more detail?"
        )
    }


# ============================================================
# HUMAN REVIEW
# ============================================================

def human_review_node(state: GraphState):

    """
    Sends the draft answer for human approval.
    """

    approved_answer = human_review(
        state["answer"]
    )

    return {
        "answer": approved_answer
    }


# ============================================================
# FINAL ANSWER
# ============================================================

def final_node(state: GraphState):

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print()

    print(state["answer"])

    print()
    print("=" * 60)

    return state