# LLM Observability

この度は『俺たちと探究する LLM オブザーバビリティ』をお読みいただき、誠にありがとうございます。   
（また、本リポジトリに Star をつけていただけると、大変嬉しいです。）

## 前提

- Python 3.12.x

## 環境構築

### ハンズオン環境

`.env.sample` をコピーし、`.env` を作成します。

```sh
cp .env.sample .env
```

必要な値を設定します。

```sh
# LLM
OPENAI_API_KEY="sk-proj-..."
COHERE_API_KEY="RTel..."

# Langfuse
ENDPOINT="http://langfuse-server:3000"
PUBLIC_KEY="pk-lf-..."
SECRET_KEY="sk-lf-..."
```

アプリケーションを起動します。

```sh
docker compose up --build -d
```

以下のようなステータスであれば、起動に成功しています。

```sh
docker compose ps
NAME         IMAGE                   COMMAND                  SERVICE           CREATED          STATUS                   PORTS
app          llm-observability-app   "streamlit run main.…"   app               18 seconds ago   Up 7 seconds             0.0.0.0:8501->8501/tcp, :::8501->8501/tcp
langfuse     langfuse/langfuse:2     "dumb-init -- ./web/…"   langfuse-server   18 seconds ago   Up 3 seconds             0.0.0.0:3000->3000/tcp, :::3000->3000/tcp
postgresql   postgres                "docker-entrypoint.s…"   db                18 seconds ago   Up 7 seconds (healthy)   0.0.0.0:5432->5432/tcp, :::5432->5432/tcp
```

以下の、URLでそれぞれのコンポーネントにアクセスが可能です。

- アプリケーション: [http://localhost:8501](http://localhost:8501)
- Langfuse: [http://localhost:3000](http://localhost:3000)

## 諸注意

このリポジトリは以下の目的で作成しています。

- 目的 1:『俺たちと探究する LLM オブザーバビリティ』の読者に対して、サンプルアプリケーションの実行環境を提供すること
- 目的 2: 著者自身に対して、執筆活動の合間の息抜きとなる遊び場を提供すること
  - そのため、実装したアプリケーションや各 OSS の設定などは、推奨される設定と異なる場合があります。ご注意のうえ、ご参照ください。
