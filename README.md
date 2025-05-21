# RAG-Chatbot

A collection of advanced Retrieval-Augmented Generation (RAG) pipelines using [LangChain](https://python.langchain.com/) and HuggingFace models. This project demonstrates multiple RAG strategies, including multi-query retrieval, multi-representation (multi-vector) indexing, and RAG fusion with reranking, all applied to real-world web documents.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Pipeline Descriptions](#pipeline-descriptions)
  - [1. Basic RAG Pipeline](#1-basic-rag-pipeline)
  - [2. Multi-Query RAG](#2-multi-query-rag)
  - [3. Multi-Representation (Multi-Vector) RAG](#3-multi-representation-multi-vector-rag)
  - [4. RAG Fusion with Reranking](#4-rag-fusion-with-reranking)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [References](#references)

---

## Overview

This repository explores advanced RAG techniques for improving the relevance and robustness of LLM-based question answering over external knowledge. The pipelines leverage:

- **Web document loading**
- **Text chunking and summarization**
- **Dense vector embedding and storage**
- **Multi-perspective and multi-representation retrieval**
- **LLM-based answer generation**

---

## Features

- **Web Scraping**: Load and parse web articles using `WebBaseLoader` and BeautifulSoup.
- **Chunking**: Split documents into overlapping chunks for better retrieval granularity.
- **Embeddings**: Use HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for semantic search.
- **Multi-Query Retrieval**: Generate multiple query variants to improve recall.
- **Multi-Representation Indexing**: Index both summaries and full documents for flexible retrieval.
- **RAG Fusion & Reranking**: Combine and deduplicate results from multiple queries.
- **LLM Integration**: Use HuggingFace's Zephyr-7b-beta for summarization and answer generation.
- **Environment Management**: Secure API keys and configuration via `.env` files.

---

## Pipeline Descriptions

### 1. Basic RAG Pipeline ([overview.py](overview.py))

- Loads a web article on LLM agents.
- Splits it into chunks and embeds them.
- Retrieves relevant chunks for a user query.
- Generates an answer using a prompt and LLM.

### 2. Multi-Query RAG ([multi_query_rag.py](multi_query_rag.py))

- For a given question, generates multiple alternative queries using an LLM.
- Retrieves documents for each query variant.
- Deduplicates and merges results for improved coverage.
- Answers the question using the aggregated context.

### 3. Multi-Representation (Multi-Vector) RAG ([multi_representation_rag.py](multi_representation_rag.py))

- Loads a document and generates a summary for each chunk.
- Indexes both summaries and full documents as separate vectors.
- Enables retrieval based on either detailed or high-level content.
- Useful for long or complex documents.

### 4. RAG Fusion with Reranking ([rag_fusion_rerank.py](rag_fusion_rerank.py))

- Combines multi-query retrieval with fusion and deduplication.
- Retrieves and merges results from multiple query perspectives.
- Reranks and answers using the most relevant context.

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a virtual environment and activate it:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   *(Create a `requirements.txt` with the following if not present:*
   ```
   langchain
   langchain-community
   langchain-huggingface
   python-dotenv
   chromadb
   beautifulsoup4
   ```
   *)

4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys:
     ```sh
     cp .env.example .env
     ```
   - Edit `.env` and provide your [LangSmith](https://smith.langchain.com/) and [HuggingFace](https://huggingface.co/) API keys.

---

## Environment Variables

See [.env.example](.env.example):

- `LANGCHAIN_API_KEY` - Your LangSmith API key
- `LANGCHAIN_TRACING_V2` - Enable LangChain tracing (true/false)
- `LANGCHAIN_ENDPOINT` - LangChain endpoint URL
- `HUGGINGFACEHUB_API_TOKEN` - Your HuggingFace API token
- `TOKENIZERS_PARALLELISM` - (Optional) Set to `false` to avoid tokenizer warnings

---

## Usage

Run any of the example pipelines:

```sh
python overview.py
python multi_query_rag.py
python multi_representation_rag.py
python rag_fusion_rerank.py
```

Each script will:
- Load and process documents
- Retrieve relevant context for a sample question
- Print the generated answer

You can modify the question or source URLs in each script as needed.

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Multi-Vector Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
- [LangChain Blog: Semi-Structured Multi-Modal RAG](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
- [HuggingFace Transformers](https://huggingface.co/)
- [Chroma Vector Database](https://www.trychroma.com/)
- [Original LLM Agents Blog Post](https://lilianweng.github.io/posts/2023-06-23-agent/)

---

## License

MIT License

---

## Acknowledgements

- [LangChain Team](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [ChromaDB](https://www.trychroma.com/)