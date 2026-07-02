"""
RAG chain: everything from the yellow 'RAG query classification' diamond
through 'Document picker', 'Retrieve docs', and 'Generate answer'.

Collapsed into a single file since it's one chain in the diagram, even
though internally it does three steps: pick relevant docs -> retrieve ->
generate.
"""

import os
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from ..config import AVAILABLE_DOCS, DOCS_DIR, EMBEDDING_MODEL_NAME, PERSIST_DIR
from ..llm import model
from ..state import GraphState

# NOTE: install deps if you haven't already:
#   pip install langchain-chroma langchain-community langchain-text-splitters
#   pip install "unstructured[md]" fastembed pydantic

# --------------------------------------------------------------------------
# Vector store: built once at import time, tagged with source filenames
# --------------------------------------------------------------------------
def _build_vectorstore() -> Chroma:
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
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
            d.metadata["source"] = filename
        all_chunks.extend(splitter.split_documents(docs))

    return Chroma.from_documents(all_chunks, embeddings, persist_directory=PERSIST_DIR)


_db = _build_vectorstore()

# --------------------------------------------------------------------------
# Step 1: document picker (structured output)
# --------------------------------------------------------------------------
class _DocumentSelection(BaseModel):
    documents: List[str] = Field(
        description=(
            "Filenames of every document relevant to the user's question. "
            "Must only contain values from the available documents list. "
            "Include more than one if the question spans multiple topics."
        )
    )


_picker_prompt = ChatPromptTemplate.from_template(
    """You are a routing assistant for a university document Q&A system.
Given the user's question and the list of available documents below, select
every document that could plausibly help answer it. Select as few as
possible, but do not miss a relevant one.

Available documents:
{doc_list}

User question: {query}
"""
)

_picker_chain = _picker_prompt | model.with_structured_output(_DocumentSelection)


def _pick_documents(query: str) -> List[str]:
    doc_list_str = "\n".join(f"- {name}: {desc}" for name, desc in AVAILABLE_DOCS.items())
    result: _DocumentSelection = _picker_chain.invoke({"doc_list": doc_list_str, "query": query})
    valid = [d for d in result.documents if d in AVAILABLE_DOCS]
    return valid or list(AVAILABLE_DOCS.keys())  # fallback: search everything


# --------------------------------------------------------------------------
# Step 2: retrieve, filtered to the selected docs
# --------------------------------------------------------------------------
def _retrieve(query: str, selected_docs: List[str]) -> tuple[str, List[str]]:
    retriever = _db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.2,
            "filter": {"source": {"$in": selected_docs}},
        },
    )
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # dedup while preserving order, so citations list each source only once
    seen = []
    for doc in relevant_docs:
        source = doc.metadata.get("source")
        if source and source not in seen:
            seen.append(source)

    return context, seen


# --------------------------------------------------------------------------
# Step 3: generate the final answer from context
# --------------------------------------------------------------------------
_answer_prompt = ChatPromptTemplate.from_template(
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

_answer_chain = _answer_prompt | model | StrOutputParser()

_NOT_FOUND_MESSAGE = "I could not find this information in the university knowledge base."


# --------------------------------------------------------------------------
# The single node this chain exposes to the main graph
# --------------------------------------------------------------------------
def rag_chain_node(state: GraphState) -> GraphState:
    query = state["query"]

    selected_docs = _pick_documents(query)
    context, sources = _retrieve(query, selected_docs)

    if not context.strip():
        # Nothing relevant retrieved: don't let the LLM guess, and flag this
        # for human review (see main.py's conditional edge on rag_found).
        return {
            **state,
            "selected_docs": selected_docs,
            "context": context,
            "answer": _NOT_FOUND_MESSAGE,
            "rag_found": False,
        }

    answer = _answer_chain.invoke({"context": context, "query": query})
    citation = "This answer is based on " + ", ".join(sources) + "."
    answer = f"{answer}\n\n{citation}"

    return {
        **state,
        "selected_docs": selected_docs,
        "context": context,
        "answer": answer,
        "rag_found": True,
    }