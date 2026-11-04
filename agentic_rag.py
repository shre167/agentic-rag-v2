import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from retrieval import NO_SOURCE_ANSWER, indexed_collection, retrieve_chunks
from safe_calculator import calculate
from settings import ROOT


load_dotenv(ROOT / ".env")
MAX_SEARCH_CALLS = 3
MAX_TOOL_CALLS = 6


def answer_with_agent(
    question: str, history: list[dict] | None = None, k: int = 5
) -> tuple[str, list[dict], list[dict]]:
    """Let a LangChain agent decide when and how to search the indexed book."""
    api_key = os.getenv("GE_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")
    if not all([api_key, base_url, model_name]):
        raise RuntimeError("Set GE_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env.")

    sources_by_id: dict[str, dict] = {}
    activity = []
    calculations = []

    def action_limit_reached() -> bool:
        return len(activity) >= MAX_TOOL_CALLS

    def record_action(tool_name: str, query: str, result: str, matches: int):
        activity.append(
            {
                "tool": tool_name,
                "input": query,
                "result": result,
                "matches": matches,
            }
        )

    @tool
    def search_book(query: str, scope: str = "") -> str:
        """Semantic-search the book; optionally limit to an exact chapter or section name."""
        if action_limit_reached():
            return "Tool-call limit reached. Use the evidence gathered so far."
        if sum(step["tool"] == "search_book" for step in activity) >= MAX_SEARCH_CALLS:
            return "Search limit reached. Use the evidence gathered so far."

        passages = retrieve_chunks(query, k=k, scope=scope or None)
        action_input = f"{query} [scope: {scope}]" if scope else query
        record_action(
            "search_book",
            action_input,
            f"Found {len(passages)} relevant passages" if passages else "No relevant passages",
            len(passages),
        )
        if not passages:
            return "NO_RELEVANT_PASSAGES: the indexed book has no relevant source for this query."

        formatted = []
        for passage in passages:
            source_id = passage["id"]
            sources_by_id[source_id] = passage
            formatted.append(
                f"[{source_id}] Chapter {passage.get('chapter', 'Unknown')}, "
                f"{passage.get('section', 'Unknown')}, "
                f"pages {passage.get('page_start', '?')}-{passage.get('page_end', '?')}\n"
                f"{passage['text']}"
            )
        return "\n\n".join(formatted)

    @tool
    def calculator(expression: str) -> str:
        """Calculate a basic arithmetic expression using numbers and arithmetic operators."""
        if action_limit_reached():
            return "Tool-call limit reached."
        try:
            result = calculate(expression)
        except ValueError as error:
            record_action("calculator", expression, "Invalid expression", 0)
            return str(error)
        calculations.append({"expression": expression, "result": result})
        record_action("calculator", expression, f"Result: {result}", 0)
        return str(result)

    @tool
    def list_book_sections() -> str:
        """List chapter and section headings available in the indexed book."""
        if action_limit_reached():
            return "Tool-call limit reached. Use the evidence gathered so far."
        stored = indexed_collection().get(include=["metadatas"])
        headings = sorted(
            {
                f"{metadata.get('chapter', 'Unknown chapter')}: "
                f"{metadata.get('section', 'Unknown section')}"
                for metadata in stored["metadatas"]
            }
        )
        record_action("list_book_sections", "", f"Found {len(headings)} headings", len(headings))
        return "\n".join(headings) if headings else "The book index has no section headings."

    @tool
    def get_section_passages(section: str) -> str:
        """Read indexed passages from a named chapter or section heading."""
        if action_limit_reached():
            return "Tool-call limit reached. Use the evidence gathered so far."
        stored = indexed_collection().get(include=["documents", "metadatas"])
        requested = section.casefold().strip()
        matches = []
        for chunk_id, text, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"]
        ):
            heading = metadata.get("section", "")
            chapter = metadata.get("chapter", "")
            heading_normalized = heading.casefold()
            chapter_normalized = chapter.casefold()
            if (
                requested in heading_normalized
                or heading_normalized in requested
                or requested in chapter_normalized
                or chapter_normalized in requested
            ):
                passage = {
                    "id": chunk_id,
                    "text": text,
                    **metadata,
                }
                matches.append(passage)
                sources_by_id[chunk_id] = passage
                if len(matches) >= k:
                    break

        record_action(
            "get_section_passages",
            section,
            f"Found {len(matches)} passages" if matches else "No matching section",
            len(matches),
        )
        if not matches:
            return "NO_RELEVANT_PASSAGES: no indexed heading matches that section."
        return "\n\n".join(
            f"[{passage['id']}] Chapter {passage.get('chapter', 'Unknown')}, "
            f"{passage.get('section', 'Unknown')}, pages "
            f"{passage.get('page_start', '?')}-{passage.get('page_end', '?')}\n"
            f"{passage['text']}"
            for passage in matches
        )

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    agent = create_agent(
        model=llm,
        tools=[search_book, list_book_sections, get_section_passages, calculator],
        system_prompt=(
            "You are a book-grounded ReAct assistant. For arithmetic-only questions, use "
            "calculator first and do not search the book; translate natural-language math "
            "into a basic arithmetic expression (for example, '18 percent of 250' becomes "
            "'(18 / 100) * 250'). For factual questions about the book, first use search_book. "
            "You may use list_book_sections to discover headings and "
            "get_section_passages to read a known heading. Refine or repeat semantic searches "
            "or pass an exact chapter/section as search_book's scope to target a part of the book. "
            "Use calculator for arithmetic instead of mental math. "
            "Repeat or refine a search when evidence is weak or incomplete. Use no more than "
            f"{MAX_SEARCH_CALLS} semantic searches and {MAX_TOOL_CALLS} total tool calls. "
            "Answer only from passage text returned by search_book or get_section_passages; "
            "section listings are navigation only, not evidence. "
            "If no relevant passage is returned, say exactly: "
            f"{NO_SOURCE_ANSWER} Do not use outside knowledge. Cite each claim with the "
            "exact chunk ID in brackets, for example [19a1c91f09bd-00068]. Never invent "
            "a citation or cite a chunk that the tool did not return."
        ),
    )

    messages = list(history or [])[-6:]
    messages.append({"role": "user", "content": question})
    result = agent.invoke({"messages": messages})
    final_message = result["messages"][-1]
    answer = final_message.content
    if isinstance(answer, list):
        answer = "".join(
            part.get("text", "") for part in answer if isinstance(part, dict)
        )
    answer = str(answer).strip()

    if not sources_by_id and not calculations:
        answer = NO_SOURCE_ANSWER
    elif answer.strip().upper() in {"[NO_SOURCES]", "NO_SOURCES"}:
        answer = NO_SOURCE_ANSWER
        sources_by_id.clear()

    return answer or NO_SOURCE_ANSWER, list(sources_by_id.values()), activity
