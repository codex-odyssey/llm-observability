import os, uuid
from dotenv import load_dotenv, find_dotenv
import vector_store as vs

import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_cohere.chat_models import ChatCohere
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
endpoint = os.getenv("ENDPOINT")
public_key = os.getenv("PUBLIC_KEY")
secret_key = os.getenv("SECRET_KEY")
fallback_prompt = """
以下の質問に答えてください。

## 質問
{{question}}
"""

# 簡易的なセッション管理
session_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_id = st.session_state["session_id"]

# Langfuse関連
langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=endpoint)
langchain_callback = CallbackHandler(
    host=endpoint, public_key=public_key, secret_key=secret_key, session_id=session_id
)

# Vector Storeの初期化
vector_store = vs.initialize()

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
        model_name = st.sidebar.selectbox(
            label="Model Name", options=["command-r-plus", "gpt-4o-mini"]
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
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    prompt = langfuse.get_prompt(
        name="bbql-app-prompt", fallback=fallback_prompt
    ).get_langchain_prompt()
    chain = (
        {"question": RunnablePassthrough(), "context": vector_store.as_retriever()}
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
