import os, glob, logging, uuid
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langfuse import Langfuse

logger = logging.getLogger(name=__name__)

_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
endpoint = os.getenv("ENDPOINT")
public_key = os.getenv("PUBLIC_KEY")
secret_key = os.getenv("SECRET_KEY")

langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=endpoint)


def initialize(model_name: str, session_id: str) -> FAISS:
    """Initialize Vector Stoare(FAISS)."""
    # Low-level SDKの使用例として、VectorStore(FAISS)の初期化処理をトレーシングしてみる
    trace = langfuse.trace(
        id=str(uuid.uuid4()),
        session_id=session_id,
        name="Initialize Vector Store",
        input=model_name,
    )
    if model_name == "gpt-4o-mini":
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    elif model_name == "command-r-plus":
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key, model="embed-multilingual-v3.0"
        )
    else:
        logger.error("Unsetted model name")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    files = glob.glob("./docs/*.txt")
    documents = []

    # ./docs/*.txtに存在するレシピ一覧を読み込み、LangChainのDocumentに変換する
    document_load_span = trace.span(name="DocumentLoad", input=files)
    for file in files:
        loader = TextLoader(file_path=file)
        document = loader.load()
        documents.extend(document)
    document_load_span.end(name="DocumentLoad", output=documents)
    
    # 読み込んだDocumentをVectorStore(FAISS)に格納する
    document_insert_span = trace.span(name="DocumentSave", input=documents)
    result = vector_store.add_documents(documents=documents)
    document_insert_span.end(name="DocumentSave", output=result)
    
    trace.update(output=vector_store.__dict__)
    return vector_store
