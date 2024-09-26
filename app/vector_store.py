import os, glob
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain_cohere import CohereEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

_ = load_dotenv(find_dotenv())
cohere_api_key = os.getenv("COHERE_API_KEY")

def initialize() -> FAISS:
    """Initialize Vector Stoare(FAISS)."""
    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key, model="embed-multilingual-v3.0"
    )
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    files = glob.glob("./docs/*.txt")
    documents = []

    for file in files:
        loader = TextLoader(file_path=file)
        document = loader.load()
        documents.extend(document)
    vector_store.add_documents(documents=documents)
    return vector_store
