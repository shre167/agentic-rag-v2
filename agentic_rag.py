import json
import os
import re
import concurrent.futures

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent.bookagent import build_agent
from retrieval.retrieval import NO_SOURCE_ANSWER
from retrieval.vector_store import list_knowledge_bases, get_collection
from settings import ROOT


load_dotenv(ROOT / ".env")


class AgentResult(tuple):
    """3-tuple subclass for backwards compatibility. Carries answer, sources, activity plus reasoning and answer_type."""
    answer: str
    sources: list[dict]
    activity: list[dict]
    reasoning: str | None
    answer_type: str  # "book" | "web" | "general" | "mixed"

    def __new__(
        cls,
        answer: str,
        sources: list[dict],
        activity: list[dict],
        reasoning: str | None = None,
        answer_type: str = "book",
    ):
        instance = super().__new__(cls, (answer, sources, activity))
        instance.answer = answer
        instance.sources = sources
        instance.activity = activity
        instance.reasoning = reasoning
        instance.answer_type = answer_type
        return instance


def validate_citations(
    raw_response: str,
    retrieved_ids: set[str],
) -> tuple[str, list[str], bool, str | None, str]:
    """
    Validate a JSON agent response.

    Returns (answer, source_ids, valid, reasoning, answer_type).
    answer_type is one of: "book", "web", "general", "mixed".

    Web answers ([web-XX] citations) are validated against the web passage IDs
    returned by web_search, not against book chunk IDs.
    General answers (answer_type="general") require no citations.
    """

    content = raw_response.strip()

    # Remove Markdown JSON code fences if present.
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # Parse JSON response.
    try:
        payload = json.loads(content)

    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start < 0 or end <= start:
            return NO_SOURCE_ANSWER, [], False, None, "general"

        try:
            payload = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return NO_SOURCE_ANSWER, [], False, None, "general"

    if not isinstance(payload, dict):
        return NO_SOURCE_ANSWER, [], False, None, "general"

    answer = payload.get("answer")
    source_ids = payload.get("sources")
    reasoning = payload.get("reasoning")
    answer_type = payload.get("answer_type", "book")

    if not isinstance(reasoning, str):
        reasoning = None
    if answer_type not in {"book", "web", "general", "mixed"}:
        answer_type = "book"

    if (
        not isinstance(answer, str)
        or not isinstance(source_ids, list)
        or any(not isinstance(source_id, str) for source_id in source_ids)
    ):
        return NO_SOURCE_ANSWER, [], False, reasoning, answer_type

    # Remove duplicate source IDs.
    source_ids = list(dict.fromkeys(source_ids))

    # Extract inline citations such as [chunk_12], [web-01], or [id1, id2].
    raw_inline_ids = re.findall(r"\[([^\[\]]+)\]", answer)
    inline_ids = []
    for item in raw_inline_ids:
        for part in re.split(r"[,;]\s*", item.strip()):
            part = part.strip()
            if part:
                inline_ids.append(part)

    # General answers need no citation validation.
    if answer_type == "general":
        return answer.strip(), source_ids, True, reasoning, answer_type

    # Validate all cited IDs exist in what tools returned this session.
    if (
        not set(source_ids).issubset(retrieved_ids)
        or not set(inline_ids).issubset(retrieved_ids)
        or not set(inline_ids).issubset(source_ids)
    ):
        return NO_SOURCE_ANSWER, [], False, reasoning, answer_type

    # A grounded answer must contain source IDs when sources were retrieved.
    if answer.strip() != NO_SOURCE_ANSWER and retrieved_ids and not source_ids:
        return NO_SOURCE_ANSWER, [], False, reasoning, answer_type

    return answer.strip(), source_ids, True, reasoning, answer_type


def _tool_payload(message) -> dict | None:
    """Extract structured tool output from a tool message."""

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
    kb_name: str | None = None,
) -> AgentResult:
    """Run the modular LangChain agent and return a validated answer, sources, activity, reasoning, and answer_type."""

    # Load LLM configuration.
    api_key = os.getenv("GE_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")

    if not all([api_key, base_url, model_name]):
        raise RuntimeError(
            "Set GE_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env."
        )

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    # Build the agent (list_books + semantic_search + web_search).
    agent = build_agent(llm)

    # Optionally inject a soft context hint (never constrains routing).
    context_note = ""
    if kb_name:
        context_note = (
            f"Context hint: the user currently has '{kb_name}' open in the viewer, "
            "but you are free to query any or all collections as needed."
        )

    messages = [
        *list(history or [])[-6:],
        {
            "role": "user",
            "content": f"{context_note}\n\n{question}".strip() if context_note else question,
        },
    ]

    # Execute agent.
    result = agent.invoke({"messages": messages})

    activity = []
    passages_by_id = {}
    calculator_used = False
    web_search_used = False

    # Walk the full message sequence in order.
    # AI messages with tool_calls hold the LLM's thought for WHY it chose each tool.
    # We pair each AI thought to its tool message via tool_call_id.
    pending_tool_thoughts: dict[str, str] = {}  # tool_call_id -> LLM thought text
    tool_counts: dict[str, int] = {}
    previous_calls: list[dict] = []

    for message in result["messages"]:
        msg_type = getattr(message, "type", None)

        # Capture LLM thoughts before tool calls.
        if msg_type == "ai" and getattr(message, "tool_calls", None):
            thought = ""
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                thought = content.strip()
            elif isinstance(content, list):
                thought = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip()

            for tc in message.tool_calls:
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                pending_tool_thoughts[tc_id] = thought
            continue

        if msg_type != "tool":
            continue

        payload = _tool_payload(message)

        if not payload:
            continue

        tool_name = payload.get("tool", "tool")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        call_num = tool_counts[tool_name]

        # Retrieve the LLM thought that triggered this tool call.
        tool_call_id = getattr(message, "tool_call_id", "") or ""
        thought_for_this_call = pending_tool_thoughts.pop(tool_call_id, "")

        # Extract input info.
        raw_input = payload.get("input", "")
        kb_param = None
        if isinstance(raw_input, str):
            kb_match = re.search(r"kb:\s*([a-zA-Z0-9_-]+)", raw_input)
            if kb_match:
                kb_param = kb_match.group(1)

        # Synthesize clear 'why_called' reasoning if LLM thought was empty.
        why_called = thought_for_this_call
        why_repeated = None
        why_web_instead_of_semantic = None

        if tool_name == "list_books":
            if not why_called:
                why_called = "Discover all indexed book collections and build the routing map to match user question to knowledge bases."

        elif tool_name == "semantic_search":
            passages = payload.get("passages", [])
            if not why_called:
                if kb_param:
                    why_called = f"Query collection '{kb_param}' for book passages matching the user's question."
                else:
                    why_called = "Search indexed book content using semantic search and metadata filters."

            # Explain repeated tool calls.
            if call_num > 1:
                prev_semantic = [c for c in previous_calls if c["tool"] == "semantic_search"]
                if prev_semantic:
                    last_call = prev_semantic[-1]
                    last_kb = last_call.get("kb_param")
                    if last_kb and kb_param and last_kb != kb_param:
                        why_repeated = (
                            f"Multi-source retrieval: Step previously searched '{last_kb}'. "
                            f"This repeated call queries '{kb_param}' so the agent can gather cross-book "
                            f"evidence and compare perspectives across multiple books."
                        )
                    else:
                        why_repeated = (
                            f"Follow-up query: Search #{call_num} in this collection with adjusted filters/terms "
                            f"to ensure complete coverage."
                        )

        elif tool_name == "web_search":
            web_search_used = True
            if not why_called:
                why_called = f"Search live web for external topic '{raw_input}'."
            why_web_instead_of_semantic = (
                "The topic is outside the scope of the indexed books ('Mind Management' & 'Deep Work'). "
                "Web search was chosen instead of book semantic search to retrieve accurate live facts "
                "and prevent hallucinating non-existent book passages."
            )

        # Build detailed 'what_found' summary.
        what_found_items = []
        passages = payload.get("passages", [])
        web_results = payload.get("web_results", [])
        routing_map = payload.get("routing_map")

        if tool_name == "list_books" and routing_map:
            what_found_items = [f"{kb} -> {title}" for kb, title in routing_map.items()]
            found_summary = f"Discovered {len(routing_map)} collection(s): " + ", ".join(what_found_items)
        elif tool_name == "web_search":
            if web_results:
                what_found_items = [f"[{r['rank']}] {r['title']} ({r.get('url','')})" for r in web_results]
                found_summary = f"Found {len(web_results)} web article(s) via {payload.get('engine', 'web')}"
            else:
                found_summary = "No web results found."
        elif tool_name == "semantic_search":
            if passages:
                books_found = sorted({p.get("book", "?") for p in passages})
                sections_found = [p.get("section", "") for p in passages[:3] if p.get("section")]
                sections_str = f" in '{', '.join(sections_found)}'" if sections_found else ""
                found_summary = f"Retrieved {len(passages)} chunk(s) from {', '.join(books_found)}{sections_str}"
                for p in passages:
                    lines_str = f" ll.{p['line_start']}-{p['line_end']}" if "line_start" in p else ""
                    what_found_items.append(
                        f"[{p['id']}] {p.get('book','Book')} | Chapter {p.get('chapter','?')} | "
                        f"pp.{p.get('page_start','?')}-{p.get('page_end','?')}{lines_str} — {p.get('section','')}"
                    )
            else:
                found_summary = "No matching chunks found for this query in the selected collection."
        else:
            found_summary = payload.get("result", "Completed")

        activity_step = {
            "step": len(activity) + 1,
            "tool": tool_name,
            "call_number": call_num,
            "input": raw_input,
            "kb_param": kb_param,
            "result": payload.get("result", ""),
            "found_summary": found_summary,
            "what_found_items": what_found_items,
            "matches": len(passages),
            "routing_map": routing_map,
            "web_results": web_results if tool_name == "web_search" else None,
            "tool_reasoning": why_called,
            "why_repeated": why_repeated,
            "why_web_instead_of_semantic": why_web_instead_of_semantic,
        }

        previous_calls.append(activity_step)
        activity.append(activity_step)

        if tool_name == "calculator":
            calculator_used = True

        for passage in passages:
            if isinstance(passage, dict) and isinstance(passage.get("id"), str):
                passages_by_id[passage["id"]] = passage

    # Find the final AI response.
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
        return AgentResult(NO_SOURCE_ANSWER, [], activity, None, "general")

    raw_answer = final_message.content

    # Handle multimodal/list-based response content.
    if isinstance(raw_answer, list):
        raw_answer = "".join(
            part.get("text", "")
            for part in raw_answer
            if isinstance(part, dict)
            and isinstance(part.get("text", ""), str)
        )

    # Validate answer, citations, extract reasoning and answer_type.
    answer, cited_ids, valid_response, reasoning, answer_type = validate_citations(
        str(raw_answer),
        set(passages_by_id),
    )

    if reasoning:
        activity.insert(0, {
            "action": "Agent Dynamic Reasoning & Strategy",
            "result": reasoning,
            "reasoning": reasoning,
            "answer_type": answer_type,
        })

    # Allow general answers (no sources required) and web answers.
    if not passages_by_id and not calculator_used and answer_type == "book":
        activity.append(
            {
                "action": "Generate grounded answer",
                "result": NO_SOURCE_ANSWER,
                "citation_status": "no_sources",
            }
        )
        return AgentResult(NO_SOURCE_ANSWER, [], activity, reasoning, "general")

    # Reject malformed responses or invalid citations (except general answers).
    if not valid_response and answer_type != "general":
        activity.append(
            {
                "action": "Generate grounded answer",
                "result": "Answer rejected: citations invalid or malformed",
                "citation_status": "rejected",
            }
        )
        return AgentResult(NO_SOURCE_ANSWER, [], activity, reasoning, answer_type)

    # Return only validated source passages.
    sources = [
        passages_by_id[source_id]
        for source_id in cited_ids
        if source_id in passages_by_id
    ]

    citation_status = "validated" if cited_ids else "not_required"

    activity.append(
        {
            "action": "Generate grounded answer",
            "result": f"Answer generated ({answer_type}) with validated sources",
            "citation_status": citation_status,
            "validated_citations": len(cited_ids),
            "answer_type": answer_type,
        }
    )

    return AgentResult(answer or NO_SOURCE_ANSWER, sources, activity, reasoning, answer_type)
