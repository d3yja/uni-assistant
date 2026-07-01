import os
from typing import TypedDict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# NOTE: install deps if you haven't already:
#   pip install langgraph langchain langchain-community langchain-chroma langchain-text-splitters
#   pip install "unstructured[md]" fastembed pydantic
#   # plus your chat-model client, e.g. `pip install langchain-openai`

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DOCS_DIR = "./docs"
PERSIST_DIR = "./db_md/chroma_db"

# filename -> short description, used both for the picker prompt and validation
AVAILABLE_DOCS = {
    "academic_calendar.md": "Semester dates, exam schedules, holidays, and academic deadlines.",
    "grading_policy.md": "How grades are calculated, GPA rules, grade appeals, grading scales.",
    "attendance_policy.md": "Attendance requirements, absence rules, and consequences for missing classes.",
    "hostel_rules.md": "Hostel/dormitory rules, curfews, visitor policy, disciplinary procedures.",
    "contact_directory.md": "Contact info for faculty, staff, departments, and offices.",
}

# --------------------------------------------------------------------------
# Chat model
# --------------------------------------------------------------------------
# Plug in whichever chat model you're using. It must support `.with_structured_output`.
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = None  # <-- replace with a real chat model instance before running

# --------------------------------------------------------------------------
# Build (or load) the vector store, tagging each chunk with its source doc
# --------------------------------------------------------------------------
def build_vectorstore() -> Chroma:
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    all_chunks = []
    for filename in AVAILABLE_DOCS:
        path = os.path.join(DOCS_DIR, filename)
        loader = UnstructuredMarkdownLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename  # tag for later filtering
        all_chunks.extend(splitter.split_documents(docs))

    return Chroma.from_documents(all_chunks, embeddings, persist_directory=PERSIST_DIR)


db = build_vectorstore()

# --------------------------------------------------------------------------
# Structured output schema for the document picker
# --------------------------------------------------------------------------
class DocumentSelection(BaseModel):
    documents: List[str] = Field(
        description=(
            "Filenames of every document relevant to the user's question. "
            "Must only contain values from the available documents list. "
            "Include more than one if the question spans multiple topics."
        )
    )


picker_prompt = ChatPromptTemplate.from_template(
    """You are a routing assistant for a university document Q&A system.
Given the user's question and the list of available documents below, select
every document that could plausibly help answer it. Select as few as
possible, but do not miss a relevant one.

Available documents:
{doc_list}

User question: {query}
"""
)

picker_chain = picker_prompt | model.with_structured_output(DocumentSelection) if model else None

# --------------------------------------------------------------------------
# Answer generation prompt
# --------------------------------------------------------------------------
answer_prompt = ChatPromptTemplate.from_template(
    """
<|system|>>
You are a helpful assistant designed to help users navigate a complex set of documents. Answer the user's query based on the following context. Follow these rules:

Use only information from the provided context.

If the context doesn't adequately address the query, say: "Based on the available information, I cannot provide a complete answer to this question."

Give clear, concise, and accurate responses. Explain complex terms if needed.

If the context contains conflicting information, point this out without attempting to resolve the conflict.

Don't use phrases like "according to the context," "as the context states," etc.

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
)
answer_chain = answer_prompt | model | StrOutputParser() if model else None

# --------------------------------------------------------------------------
# LangGraph state + nodes
# --------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    selected_docs: List[str]
    context: str
    answer: str


def document_picker_node(state: GraphState) -> GraphState:
    doc_list_str = "\n".join(f"- {name}: {desc}" for name, desc in AVAILABLE_DOCS.items())
    result: DocumentSelection = picker_chain.invoke(
        {"doc_list": doc_list_str, "query": state["query"]}
    )

    # guard against the model hallucinating a filename that doesn't exist
    valid = [d for d in result.documents if d in AVAILABLE_DOCS]
    if not valid:
        valid = list(AVAILABLE_DOCS.keys())  # fallback: search everything

    return {**state, "selected_docs": valid}


def retrieve_docs_node(state: GraphState) -> GraphState:
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.2,
            "filter": {"source": {"$in": state["selected_docs"]}},
        },
    )
    relevant_docs = retriever.invoke(state["query"])
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    return {**state, "context": context}


def generate_answer_node(state: GraphState) -> GraphState:
    answer = answer_chain.invoke({"context": state["context"], "query": state["query"]})
    return {**state, "answer": answer}


# --------------------------------------------------------------------------
# Build the subgraph: document_picker -> retrieve_docs -> generate_answer
# --------------------------------------------------------------------------
def build_rag_subgraph():
    graph = StateGraph(GraphState)
    graph.add_node("document_picker", document_picker_node)
    graph.add_node("retrieve_docs", retrieve_docs_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("document_picker")
    graph.add_edge("document_picker", "retrieve_docs")
    graph.add_edge("retrieve_docs", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


if __name__ == "__main__":
    if model is None:
        raise RuntimeError("Set `model` to a real chat model instance before running.")

    app = build_rag_subgraph()
    result = app.invoke({"query": "When are the mid-semester exams?"})
    print("Selected docs:", result["selected_docs"])
    print("Answer:", result["answer"])