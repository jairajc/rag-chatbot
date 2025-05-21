"""
Multi-Representation Indexing (Multi-Vector RAG) Example

WHAT are we doing?
------------------
We are building a RAG pipeline that indexes and retrieves documents using *multiple representations* (summaries and full docs).
This allows the retriever to match queries to both detailed and summarized content, improving recall and relevance.

WHY are we doing this?
----------------------
- Sometimes, a user's query matches a summary better than a chunk of the original document.
- Summaries can capture high-level meaning, while chunks capture details.
- By indexing both, we can retrieve more relevant information for a wider range of queries.

WHAT is achieved?
-----------------
- Improved retrieval for both broad and specific queries.
- More robust and flexible RAG pipelines.
- Foundation for multi-modal or semi-structured retrieval.

HOW is this helpful?
--------------------
- Useful for long documents, reports, or web pages where both overview and detail matter.
- Enables advanced RAG applications (e.g., research assistants, knowledge workers).

References:
- https://blog.langchain.dev/semi-structured-multi-modal-rag/
- https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector
- https://arxiv.org/abs/2312.06648
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

# --- ENVIRONMENT ---
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# --- LOAD SMALLER DOCUMENT ---
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Task_(project_management)")
docs = loader.load()

# --- SUMMARIZE EACH DOCUMENT ---
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.1,
    max_new_tokens=512
)
summary_chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)
summaries = summary_chain.batch(docs, {"max_concurrency": 5})

# --- MULTI-VECTOR INDEXING ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="summaries", embedding_function=embedding)
store = InMemoryByteStore()
id_key = "doc_id"
import uuid
doc_ids = [str(uuid.uuid4()) for _ in docs]
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# --- RETRIEVAL ---
query = "What is a task in project management?"
retrieved_docs = retriever.get_relevant_documents(query, n_results=2)
for i, doc in enumerate(retrieved_docs):
    print(f"\nRetrieved Doc {i+1}:\n", doc.page_content[:500])

# --- RAG GENERATION ---
rag_prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on this context:

{context}

Question: {question}
"""
)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
rag_chain = (
    {"context": lambda x: context, "question": lambda x: x["question"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)
response = rag_chain.invoke({"question": query})
print("\nRAG Response:\n", response)