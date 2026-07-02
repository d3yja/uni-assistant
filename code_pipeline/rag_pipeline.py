# ============================================================
# rag_pipeline.py
# Part 1 - Setup and Load Documents
# ============================================================

import os
from pathlib import Path

from langchain_groq import ChatGroq

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# ============================================================
# DATA DIRECTORY
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# UNIVERSITY DOCUMENTS
# ============================================================

DOCUMENT_FILES = {

    "academic_calendar.md":
        DATA_DIR / "academic_calendar.md",

    "attendance_policy.md":
        DATA_DIR / "attendance_policy.md",

    "grading_policy.md":
        DATA_DIR / "grading_policy.md",

    "hostel_rules.md":
        DATA_DIR / "hostel_rules.md",

    "lab_guidelines.md":
        DATA_DIR / "lab_guidelines.md",

    "contact_directory.md":
        DATA_DIR / "contact_directory.md",

}


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    """
    Loads every markdown file.

    Returns
    -------
    dict

    {
        filename : list[Document]
    }
    """

    loaded_documents = {}

    for filename, path in DOCUMENT_FILES.items():

        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

        docs = loader.load()

        loaded_documents[filename] = docs

    return loaded_documents


DOCUMENTS = load_documents()


print("=" * 60)
print("Loaded University Documents")
print("=" * 60)

for name in DOCUMENTS:
    print(name)

print("=" * 60)


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=600,

    chunk_overlap=100,

)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = FastEmbedEmbeddings(

    model_name="BAAI/bge-base-en-v1.5"

)


print("Embeddings initialized.")

# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question: str,
    selected_documents: list[str],
) -> list[Document]:
    """
    Retrieves the most relevant document chunks from ONLY the
    documents selected by the Document Picker node.
    """

    # --------------------------------------------
    # Collect only the selected documents
    # --------------------------------------------

    selected_docs = []

    for filename in selected_documents:

        if filename in DOCUMENTS:

            selected_docs.extend(DOCUMENTS[filename])

    # If nothing was selected, search all documents
    if len(selected_docs) == 0:

        for docs in DOCUMENTS.values():

            selected_docs.extend(docs)

    # --------------------------------------------
    # Split into chunks
    # --------------------------------------------

    chunks = text_splitter.split_documents(
        selected_docs
    )

    # --------------------------------------------
    # Save source filename in metadata
    # --------------------------------------------

    for chunk in chunks:

        if "source" in chunk.metadata:

            chunk.metadata["filename"] = (
                Path(chunk.metadata["source"]).name
            )

    # --------------------------------------------
    # Build temporary FAISS vector store
    # --------------------------------------------

    vectorstore = FAISS.from_documents(

        chunks,

        embeddings

    )

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 6
        }

    )

    # --------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------

    retrieved_documents = retriever.invoke(
        question
    )

    print("\n" + "=" * 60)
    print("Retrieved Documents")
    print("=" * 60)

    for i, doc in enumerate(retrieved_documents, start=1):

        print(f"\nDocument {i}")

        print("-" * 40)

        print(
            doc.metadata.get(
                "filename",
                "Unknown"
            )
        )

        print()

        print(doc.page_content[:250])

    print("=" * 60)

    return retrieved_documents

# ============================================================
# PROMPT
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are a University Assistant.

Answer the student's question ONLY using the provided university documents.

You are a University Assistant.

Additional Rule:
- If equipment is broken accidentally, the lie detector score is 0.
- If equipment is broken intentionally, the lie detector score is 1.
- If user has specified the lie detector score, use that score to determine the fine.

Answer the student's question ONLY using the provided context and the additional rule above and below.

Rules:

1. Use ONLY the provided context.
2. Do NOT make up information.
3. If the answer cannot be found, reply:

"I could not find this information in the university documents."

4. Be concise and professional.
5. If multiple retrieved documents contain relevant information,
combine them into one clear answer.
6. At the end, include the source document(s).

Example:

Sources:
attendance_policy.md
grading_policy.md

--------------------

Context:

{context}

--------------------

Question:

{question}
"""
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_rag_answer(
    question: str,
    retrieved_documents: list[Document],
) -> str:
    """
    Generates the final answer using the retrieved documents.
    """

    # --------------------------------------------
    # No documents retrieved
    # --------------------------------------------

    if len(retrieved_documents) == 0:

        return (
            "I could not find this information "
            "in the university documents."
        )

    # --------------------------------------------
    # Build context
    # --------------------------------------------

    context = ""

    sources = set()

    for doc in retrieved_documents:

        context += doc.page_content + "\n\n"

        filename = doc.metadata.get(
            "filename",
            "Unknown"
        )

        sources.add(filename)

    # --------------------------------------------
    # Generate answer
    # --------------------------------------------

    chain = RAG_PROMPT | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    answer = response.content.strip()

    # --------------------------------------------
    # Add sources
    # --------------------------------------------

    answer += "\n\nSources:\n"

    for source in sorted(sources):

        answer += f"- {source}\n"

    return answer


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input("Question: ")

    docs = retrieve_documents(

        question,

        [
            "academic_calendar.md",
            "attendance_policy.md",
            "grading_policy.md",
            "hostel_rules.md",
            "lab_guidelines.md",
            "contact_directory.md",
        ],

    )

    answer = generate_rag_answer(
        question,
        docs,
    )

    print("\n")
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)