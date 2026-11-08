# Agentic PDF RAG

A Streamlit ebook assistant that indexes PDFs into separate local ChromaDB knowledge bases and uses a LangChain agent to answer with retrieved evidence and citations.

## Project Layout

```text
simple-rag/
|-- agent/
|   |-- bookagent.py
|   `-- prompts.py
|-- data1/
|   `-- Mind Management, Not Time Management.pdf
|-- data2/
|   `-- Deep Work.pdf
|-- retrieval/
|   |-- reranking.py
|   |-- retrieval.py
|   `-- vector_store.py
|-- tools/
|   |-- calculator.py
|   |-- chapter_search.py
|   |-- list_books.py
|   |-- list_book_structure.py
|   |-- page_search.py
|   |-- section_search.py
|   `-- semantic_search.py
|-- agentic_rag.py
|-- app.py
|-- chunking.py
|-- ingest.py
|-- query.py
`-- settings.py
```

`data1` is indexed as `kb1` (Mind Management), and `data2` is indexed as `kb2` (Deep Work). Each knowledge base has its own Chroma collection; chunk metadata also stores the readable book title, chapter, section, and PDF page.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GE_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` for your OpenAI-compatible chat endpoint.
4. Put each authorized PDF in its matching source folder and index it into the corresponding knowledge base:

   ```powershell
   python ingest.py "data1/Mind Management, Not Time Management.pdf" --book "Mind Management" --kb kb1 --preview
   python ingest.py "data1/Mind Management, Not Time Management.pdf" --book "Mind Management" --kb kb1
   python ingest.py "data2/Deep Work.pdf" --book "Deep Work" --kb kb2 --preview
   python ingest.py "data2/Deep Work.pdf" --book "Deep Work" --kb kb2
   python -m streamlit run app.py
   ```

Run ingestion again after changing a PDF or its chunking settings. Keep the `--book` title and `--kb` assignment consistent when re-indexing.

The source book PDF, local Chroma database, environment files, and virtual environments are intentionally excluded from Git. Add only documents you are authorized to use; place the PDF locally before indexing.

## Design

- `chunking.py` extracts heading-aware sections and splits them into page-aware chunks.
- `retrieval/vector_store.py` stores chunks in the selected Chroma knowledge base.
- `retrieval/retrieval.py` applies book/chapter/section/page filters and the no-source relevance gate.
- `retrieval/reranking.py` combines semantic distance with lexical and heading matches.
- `tools/` contains the agent's book discovery, scoped search, page search, and safe calculator tools.
- `agent/` constructs the LangChain agent and defines its operating instructions.
- `agentic_rag.py` orchestrates the agent and validates cited chunk IDs against retrieved evidence.
- `app.py` provides the Streamlit chat and visible tool activity.
- `ingest.py` indexes a PDF into a selected knowledge base; `query.py` provides a terminal chat loop.

## Useful Ebook Tools

The current tools cover source selection, general retrieval, chapter/section/page-scoped retrieval, book-structure discovery, and arithmetic. Good next additions are:

- `find_in_book(query, book=None)`: locate topics, names, or phrases and return cited page ranges; keep it as a retrieval mode rather than a duplicate search implementation.
- `get_passage_context(chunk_id, before=1, after=1)`: expand a cited chunk with adjacent chunks for context, constrained to the same book and chapter.
- `compare_books(question, books)`: orchestrate a search in each named book and return separate evidence sets before the LLM compares them.
- `chapter_overview(book, chapter)`: return the chapter's section headings and page spans; use the PDF metadata, not a generated summary, as navigation data.

Build evaluation tests before adding more tools: single-book questions, explicit book selection, chapter/page scope, cross-book comparisons, and off-topic questions that should return no sources. Calculator expressions use an allowlist of arithmetic operators; Python `eval` is not used.


> Select-String -SimpleMatch "tool_result("

.venv\Lib\site-packages\langchain\mcp\tools.py:168:def _convert_call_tool_result(
.venv\Lib\site-packages\langchain\mcp\tools.py:294:        return _convert_call_tool_result(result)
.venv\Lib\site-packages\langchain_community\chat_models\oci_generative_ai.py:286:                            tool_result = self.oci_tool_result()
.venv\Lib\site-packages\langsmith\integrations\pipecat\processor.py:317:            return self._handle_tool_result(tspan, trace_id)
.venv\Lib\site-packages\langsmith\integrations\pipecat\processor.py:441:    def _handle_tool_result(self, tspan: TranslatedSpan, trace_id: int) -> bool:
tools\calculator.py:60:        return tool_result("calculator", expression, f"Error: {error}")
tools\calculator.py:61:    return tool_result("calculator", expression, f"Result: {result}")
tools\chapter_search.py:41:        return tool_result(
tools\chapter_search.py:57:        return tool_result(
tools\chapter_search.py:63:    return tool_result(
tools\list_books.py:17:    return tool_result(
tools\list_book_structure.py:23:            return tool_result(
tools\list_book_structure.py:42:    return tool_result(
tools\page_search.py:13:        return tool_result("page_search", str(page), "Page number must be positive")
tools\page_search.py:24:        return tool_result(
tools\page_search.py:30:    return tool_result(
tools\results.py:4:def tool_result(
tools\section_search.py:29:        return tool_result(
tools\section_search.py:35:    return tool_result(
tools\semantic_search.py:42:        return tool_result(
tools\semantic_search.py:49:    return tool_result(
venv\Lib\site-packages\langchain\mcp\tools.py:168:def _convert_call_tool_result(
venv\Lib\site-packages\langchain\mcp\tools.py:294:        return _convert_call_tool_result(result)
venv\Lib\site-packages\langchain_community\chat_models\oci_generative_ai.py:286:                            tool_result = self.oci_tool_result()
venv\Lib\site-packages\langsmith\integrations\pipecat\processor.py:317:            return self._handle_tool_result(tspan, trace_id)
venv\Lib\site-packages\langsmith\integrations\pipecat\processor.py:441:    def _handle_tool_result(self, tspan: TranslatedSpan, trace_id: int) -> bool:


(.venv) PS C:\Users\shrsamal\OneDrive - Capgemini\Desktop\simple-rag> 
