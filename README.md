# Agentic PDF RAG

A small Streamlit chat app that indexes a PDF into local ChromaDB and uses a LangChain ReAct agent to answer questions from retrieved passages.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GE_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` for your OpenAI-compatible chat endpoint.
4. Place a PDF you are authorized to process at `data/Mind Management, Not Time Management.pdf`, or pass another PDF path to the ingest command.
5. Index it and start the chat:

   ```powershell
   python ingest.py --preview
   python ingest.py
   python -m streamlit run app.py
   ```

The source book PDF, local Chroma database, environment files, and virtual environments are intentionally excluded from Git. Add only documents you are authorized to use; place the PDF locally before indexing.

## Design

- `chunking.py` extracts heading-aware sections and splits them into overlapping chunks with page metadata.
- `vector_store.py` stores and retrieves Chroma records.
- `reranking.py` combines semantic distance with lexical and heading matches.
- `retrieval.py` applies scoped search and the no-source relevance gate.
- `agentic_rag.py` defines the LangChain agent tools: semantic book search, section listing/reading, and a safe arithmetic calculator.
- `app.py` provides the Streamlit chat and visible tool activity.
- `ingest.py` indexes a PDF; `query.py` provides a terminal chat loop.

The agent can target an exact chapter or section by passing `scope` to `search_book`. Calculator expressions are parsed with an allowlist of arithmetic operators; Python `eval` is not used.
