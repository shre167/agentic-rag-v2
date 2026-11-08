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


import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent.bookagent import build_agent
from retrieval.retrieval import NO_SOURCE_ANSWER
from settings import ROOT


load_dotenv(ROOT / ".env")


def validate_citations(
    raw_response: str, retrieved_ids: set[str]
) -> tuple[str, list[str], bool]:
    """Accept only JSON answers whose citations were retrieved this turn."""
    content = raw_response.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            return NO_SOURCE_ANSWER, [], False
        try:
            payload = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return NO_SOURCE_ANSWER, [], False

    if not isinstance(payload, dict):
        return NO_SOURCE_ANSWER, [], False
    answer = payload.get("answer")
    source_ids = payload.get("sources")
    if (
        not isinstance(answer, str)
        or not isinstance(source_ids, list)
        or any(not isinstance(source_id, str) for source_id in source_ids)
    ):
        return NO_SOURCE_ANSWER, [], False

    source_ids = list(dict.fromkeys(source_ids))
    inline_ids = re.findall(r"\[([^\[\]]+)\]", answer)
    if (
        not set(source_ids).issubset(retrieved_ids)
        or not set(inline_ids).issubset(retrieved_ids)
        or not set(inline_ids).issubset(source_ids)
    ):
        return NO_SOURCE_ANSWER, [], False
    if answer.strip() != NO_SOURCE_ANSWER and retrieved_ids and not source_ids:
        return NO_SOURCE_ANSWER, [], False
    return answer.strip(), source_ids, True


def _tool_payload(message) -> dict | None:
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) and "tool" in payload else None


def answer_with_agent(
    question: str,
    history: list[dict] | None = None,
    k: int = 5,
    kb_name: str = "kb1",
) -> tuple[str, list[dict], list[dict]]:
    """Run the modular LangChain agent and return a validated answer and sources."""
    api_key = os.getenv("GE_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")
    if not all([api_key, base_url, model_name]):
        raise RuntimeError("Set GE_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env.")

    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    agent = build_agent(llm)
    messages = [
        {
            "role": "system",
            "content": (
                f"The user selected knowledge base '{kb_name}'. Use this exact kb_name "
                "argument for every knowledge-base search tool call. Do not search other "
                "collections unless the application explicitly selects them."
            ),
        },
        *list(history or [])[-6:],
        {"role": "user", "content": question},
    ]
    result = agent.invoke({"messages": messages})

    activity = []
    passages_by_id = {}
    calculator_used = False
    for message in result["messages"]:
        if getattr(message, "type", None) != "tool":
            continue
        payload = _tool_payload(message)
        if not payload:
            continue
        activity.append(
            {
                "tool": payload.get("tool", "tool"),
                "input": payload.get("input", ""),
                "result": payload.get("result", ""),
                "matches": len(payload.get("passages", [])),
            }
        )
        if payload.get("tool") == "calculator":
            calculator_used = True
        for passage in payload.get("passages", []):
            if isinstance(passage, dict) and isinstance(passage.get("id"), str):
                passages_by_id[passage["id"]] = passage

    final_message = next(
        (
            message
            for message in reversed(result["messages"])
            if getattr(message, "type", None) == "ai"
            and not getattr(message, "tool_calls", None)
        ),
        None,
    )
    if final_message is None:
        activity.append(
            {
                "action": "Generate grounded answer",
                "result": "No final answer was produced",
                "citation_status": "rejected",
            }
        )
        return NO_SOURCE_ANSWER, [], activity

    raw_answer = final_message.content
    if isinstance(raw_answer, list):
        raw_answer = "".join(
            part.get("text", "") for part in raw_answer if isinstance(part, dict)
        )
    answer, cited_ids, valid_response = validate_citations(
        str(raw_answer), set(passages_by_id)
    )

    if not passages_by_id and not calculator_used:
        activity.append(
            {
                "action": "Generate grounded answer",
                "result": NO_SOURCE_ANSWER,
                "citation_status": "no_sources",
            }
        )
        return NO_SOURCE_ANSWER, [], activity
    if not valid_response:
        activity.append(
            {
                "action": "Generate grounded answer",
                "result": "Answer rejected because citations were invalid or malformed",
                "citation_status": "rejected",
            }
        )
        return NO_SOURCE_ANSWER, [], activity

    sources = [passages_by_id[source_id] for source_id in cited_ids]
    citation_status = "validated" if cited_ids else "not_required"
    activity.append(
        {
            "action": "Generate grounded answer",
            "result": "Answer generated with validated sources",
            "citation_status": citation_status,
            "validated_citations": len(cited_ids),
        }
    )
    return answer or NO_SOURCE_ANSWER, sources, activityI found the issue.

Your file itself is syntactically fine. The problem is this line:

inline_ids = re.findall(r"<strong class="___1qwroh2 fl43uef f19n0e5" data-lexical-text="true">\[</strong>([^<strong class="___1qwroh2 fl43uef f19n0e5" data-lexical-text="true">\[\]</strong>]+)<strong class="___1qwroh2 fl43uef f19n0e5" data-lexical-text="true">\]</strong>", answer)


The nested double quotes break Python parsing.

Replace the entire line with:

inline_ids = [
    citation.strip()
    for citation in re.findall(
        r"\[([^\[\]]+)\]",
        answer,
    )
]


So your validate_citations() section becomes:

source_ids = list(dict.fromkeys(source_ids))

inline_ids = [
    citation.strip()
    for citation in re.findall(
        r"\[([^\[\]]+)\]",
        answer,
    )
]

if (
    not set(source_ids).issubset(retrieved_ids)
    or not set(inline_ids).issubset(retrieved_ids)
    or not set(inline_ids).issubset(source_ids)
):
    return NO_SOURCE_ANSWER, [], False

if answer.strip() != NO_SOURCE_ANSWER and retrieved_ids and not source_ids:
    return NO_SOURCE_ANSWER, [], False

return answer.strip(), source_ids, True


Also add this debug block before validation:

print("\n" + "=" * 80)
print("RAW MODEL RESPONSE")
print("=" * 80)
print(raw_answer)
print("=" * 80 + "\n")


right before:

answer, cited_ids, valid_response = validate_citations(
    str(raw_answer), set(passages_by_id)
)


Then save and run:

python query.py


or

streamlit run app.py


Ask:

what is the creative sweet spot


and paste the RAW MODEL RESPONSE output. That's the next thing we need to inspect.
