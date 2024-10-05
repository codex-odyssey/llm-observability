import os, uuid, logging
from dotenv import load_dotenv, find_dotenv
import vector_store as vs

import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.rerank import CohereRerank
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

logger = logging.getLogger(name=__name__)

_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
endpoint = os.getenv("LANGFUSE_ENDPOINT")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
fallback_prompt = """
以下の質問に答えてください。

## 質問
{{question}}
"""

# 簡易的なセッション管理
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_id = st.session_state["session_id"]

# Langfuse関連
if "langfuse" not in st.session_state:
    st.session_state["langfuse"] = Langfuse(
        public_key=public_key, secret_key=secret_key, host=endpoint
    )
langfuse = st.session_state["langfuse"]
if "langchain_callback" not in st.session_state:
    st.session_state["langchain_callback"] = CallbackHandler(
        host=endpoint,
        public_key=public_key,
        secret_key=secret_key,
        session_id=session_id,
        trace_name="Ask the BigBaBy",
        tags=["app"],
    )
langchain_callback = st.session_state["langchain_callback"]

st.title("🍖 Ask the BigBaBy 🍖")
st.caption(
    """
技術書典#17 - 俺たちのLLM Observability本のハンズオンで使用するサンプルアプリケーションです。  
今日の晩御飯の献立に困った時にご活用ください。  
OpenAI GPT-4o, Cohere Command R+のどちらかを選んで利用可能です。
"""
)

# サイドバー関連
with st.sidebar.container():
    with st.sidebar:
        st.sidebar.markdown("### LLM関連パラメータ")
        model_name = st.sidebar.selectbox(
            label="Model Name",
            options=["command-r-plus", "gpt-4o-mini"],
        )
        max_tokens = st.sidebar.slider(
            label="Max Tokens",
            min_value=128,
            max_value=2048,
            value=1024,
            step=128,
            help="LLMが出力する最大のトークン長",
        )
        temperature = st.sidebar.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="モデルの出力のランダム性",
        )
        st.sidebar.markdown("### 検索関連パラメータ")
        top_k = st.sidebar.slider(
            label="Top K",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="関連情報の取得数",
        )
        use_reranker = st.sidebar.radio(
            label="Reranker",
            options=[True, False],
            horizontal=True,
            help="Rerankerを使用するか（※使用には、CohereのAPI Keyが必要です）",
        )
        top_n = st.sidebar.slider(
            label="Top N",
            min_value=1,
            max_value=10,
            value=3,
            help="Rerankerで取得した情報を何件に絞り込むか",
        )


def generate_response(query: str):
    """Generate LLM response via streaming output."""
    if model_name == "gpt-4o-mini":
        chat_model = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif model_name == "command-r-plus":
        chat_model = ChatCohere(
            cohere_api_key=cohere_api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        logger.error("Unsetted model name")

    # 軽微な処理なので、アプリケーションの実行ごとに初期化する
    vector_store = vs.initialize(model_name=model_name, session_id=session_id)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Rerankerの使用フラグが有効（デフォルト）の場合は、CohereのRerankerを用いて、
    # 取得した情報を関連度順に並び替えた後に、指定件数分のみ採用する
    if use_reranker == True:
        compressor = CohereRerank(
            cohere_api_key=cohere_api_key, model="rerank-multilingual-v3.0", top_n=top_n
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    # bbql-app-promptをLangfuse上で作成すると、回答が変わることを確認する
    prompt = langfuse.get_prompt(
        name="bbql-app-prompt", fallback=fallback_prompt
    ).get_langchain_prompt()

    chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | PromptTemplate.from_template(prompt)
        | chat_model
        | StrOutputParser()
    )
    response = chain.stream(input=query, config={"callbacks": [langchain_callback]})
    for chunk in response:
        yield chunk


if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("今日は何が食べたい気分ですか？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        stream = generate_response(query=prompt)
        response = st.write_stream(stream=stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
