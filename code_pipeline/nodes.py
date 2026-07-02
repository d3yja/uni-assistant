#importations
from langchain_groq import ChatGroq
import os
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState
from human_review import human_review
from dotenv import load_dotenv
from memory import (
    save_user_memory,
    get_user_memory,
    get_all_memory,
)

load_dotenv()

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
Questions about remembering information.
Examples
Remember my favorite subject is AI.
Remember I live in Lahore.
Remember my name is Muhammad.
What do you remember about me?
What do you know about me?
Recall my saved information.

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
You are selecting documents for a retrieval system.

Choose EVERY document that might contain information needed to answer the question.

If answering requires combining information from multiple documents, return ALL relevant filenames.

Examples:

Question:
What is the fine for breaking a microscope?

Return:

lab_guidelines.md
equipment_prices.md

Question:
When is the final exam?

Return:

academic_calendar.md

Question:
What happens if I miss three labs?

Return:

attendance_policy.md
lab_guidelines.md

Return ONLY filenames.

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

    question = state["question"].lower()

    # -----------------------------
    # Save memory
    # -----------------------------
    if "remember" in question:

        try:

            info = state["question"].split("remember", 1)[1].strip()

            save_user_memory(info)

            return {
                "answer": f"I'll remember that: {info}"
            }

        except Exception:

            return {
                "answer": "I couldn't understand what to remember."
            }

    # -----------------------------
    # Recall memory
    # -----------------------------
    if (
        "what do you remember" in question
        or "what do you know about me" in question
    ):

        memory = get_all_memory()

        if "memories" not in memory or len(memory["memories"]) == 0:

            return {
                "answer": "I don't have anything stored yet."
            }

        text = "Here is what I remember:\n\n"

        for i, item in enumerate(memory["memories"], start=1):

            text += f"{i}. {item}\n"

        return {
            "answer": text
        }

    return {
        "answer": "Memory request completed."
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
# Human Review
# ============================================================

review_prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are deciding whether an answer requires human review.

Return ONLY:

yes

or

no

Review is required if:

- the retrieved answer has low confidence
- the user asks for permission
- the question is about personal academic decisions
- the answer could not be confidently answered
- the answer says information was not found
- retrieved_documents is empty
- the answer says "I couldn't find"

Return only yes or no.
"""
),
(
"human",
"""
Question:
{question}

Answer:
{answer}
"""
)
]
)


def review_decision_node(state: GraphState):

    chain = review_prompt | llm

    result = chain.invoke(
        {
            "question": state["question"],
            "answer": state["answer"]
        }
    )

    decision = result.content.strip().lower()

    return {
        "needs_human_review":
            decision == "yes"
    }


# ============================================================
# Human Review Router Node
# ============================================================

def review_router(state: GraphState):

    if state["needs_human_review"]:
        return "review"

    return "final"


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