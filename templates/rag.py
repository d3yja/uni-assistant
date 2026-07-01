from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader # Changed to PyPDFLoader
from langchain_chroma import Chroma
# from langchain_docling.loader import DoclingLoader # Removed DoclingLoader
# from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Ensure pypdf is installed for PyPDFLoader penis man

# Define the directory containing the PDF file and the persistent directory
current_dir = "/content/"
# !!! IMPORTANT: Replace 'your_document.pdf' with the actual name of your PDF file in /content/sample_data/ !!!
# For demonstration, I'm using a placeholder name. You might need to upload a PDF to /content/sample_data/ or specify the correct path.
file_path = "/content/sample_data/oddessy.pdf"
persistent_directory_pdf = os.path.join(current_dir, "db_pdf", "chroma_db") # Changed persistent directory

# Read the PDF content from the file
loader = PyPDFLoader(file_path) # Changed to PyPDFLoader and using file_path

# Load all documents
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=[" ", ",", "\n"])
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information (PDF) ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")


embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# embeddings = SentenceTransformerEmbeddings(model2)
print("\n--- Finished creating embeddings (PDF) ---")



# Create the vector store and persist it automatically
print("\n--- Creating PDF vector store ---")
db_pdf = Chroma.from_documents(
    docs, embeddings, persist_directory=persistent_directory_pdf) # Changed db variable name
retriever_pdf = db_pdf.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)
print("\n--- Finished creating PDF vector store and retriever ---")




# cell 2


query = "Who is wife?"

# Retrieve relevant documents based on the query
retriever_pdf_simple_query = db_pdf.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)
relevant_docs = retriever_pdf_simple_query.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents (PDF) ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# cell 3


from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """
<|system|>>
You are a helpful assistant designed to help users navigate a complex set of documents. Answer the user's query based on the following context. Follow these rules:

Use only information from the provided context.

If the context doesn't adequately address the query, say: "Based on the available information, I cannot provide a complete answer to this question."

Give clear, concise, and accurate responses. Explain complex terms if needed.

If the context contains conflicting information, point this out without attempting to resolve the conflict.

Don't use phrases like "according to the context," "as the context states," etc.

Remember, your purpose is to provide information based on the retrieved context, not to offer original advice.

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""


# cell 4

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

response = chain.invoke("Who is Odysseus")

print(response)
