FROM python:3.12.5-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && \
    apt-get -y upgrade && \
    pip install -r requirements.txt
EXPOSE 8501
COPY . /app
ENV PATH=“${PATH}:/root/.local/bin”
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
