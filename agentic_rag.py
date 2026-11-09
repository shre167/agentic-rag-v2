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
    raw_response: str,
    retrieved_ids: set[str],
) -> tuple[str, list[str], bool]:
    """Accept only JSON answers whose citations were retrieved this turn."""

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

    # Remove duplicate source IDs.
    source_ids = list(dict.fromkeys(source_ids))

    # Extract inline citations such as [chunk_12] or [id1, id2].
    raw_inline_ids = re.findall(r"\[([^\[\]]+)\]", answer)
    inline_ids = []
    for item in raw_inline_ids:
        for part in re.split(r"[,;]\s*", item.strip()):
            part = part.strip()
            if part:
                inline_ids.append(part)

    # Validate all citations against retrieved passages.
    if (
        not set(source_ids).issubset(retrieved_ids)
        or not set(inline_ids).issubset(retrieved_ids)
        or not set(inline_ids).issubset(source_ids)
    ):
        return NO_SOURCE_ANSWER, [], False

    # A grounded answer must contain source IDs when sources were retrieved.
    if answer.strip() != NO_SOURCE_ANSWER and retrieved_ids and not source_ids:
        return NO_SOURCE_ANSWER, [], False

    return answer.strip(), source_ids, True


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
    kb_name: str = "kb1",
) -> tuple[str, list[dict], list[dict]]:
    """Run the modular LangChain agent and return a validated answer and sources."""

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

    # Build the existing agent.
    agent = build_agent(llm)

    messages = [
        {
            "role": "system",
            "content": (
                f"The user selected knowledge base '{kb_name}'. "
                "Use this exact kb_name argument for every knowledge-base "
                "search tool call. Do not search other collections unless "
                "the application explicitly selects them."
            ),
        },
        *list(history or [])[-6:],
        {
            "role": "user",
            "content": question,
        },
    ]

    # Execute agent.
    result = agent.invoke({"messages": messages})

    activity = []
    passages_by_id = {}
    calculator_used = False

    # Collect tool activity and retrieved passages.
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

            if (
                isinstance(passage, dict)
                and isinstance(passage.get("id"), str)
            ):
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

        return NO_SOURCE_ANSWER, [], activity

    raw_answer = final_message.content

    # Handle multimodal/list-based response content.
    if isinstance(raw_answer, list):
        raw_answer = "".join(
            part.get("text", "")
            for part in raw_answer
            if isinstance(part, dict)
            and isinstance(part.get("text", ""), str)
        )

    # DEBUG: Inspect the exact LLM response before validation.
    print("\n" + "=" * 80)
    print("RAW MODEL RESPONSE")
    print("=" * 80)
    print(raw_answer)
    print("=" * 80 + "\n")

    # Validate answer and citations.
    answer, cited_ids, valid_response = validate_citations(
        str(raw_answer),
        set(passages_by_id),
    )

    # Reject unsupported answers unless calculator was used.
    if not passages_by_id and not calculator_used:

        activity.append(
            {
                "action": "Generate grounded answer",
                "result": NO_SOURCE_ANSWER,
                "citation_status": "no_sources",
            }
        )

        return NO_SOURCE_ANSWER, [], activity

    # Reject malformed responses or invalid citations.
    if not valid_response:

        activity.append(
            {
                "action": "Generate grounded answer",
                "result": (
                    "Answer rejected because citations were "
                    "invalid or malformed"
                ),
                "citation_status": "rejected",
            }
        )

        return NO_SOURCE_ANSWER, [], activity

    # Return only validated source passages.
    sources = [
        passages_by_id[source_id]
        for source_id in cited_ids
    ]

    citation_status = "validated" if cited_ids else "not_required"

    activity.append(
        {
            "action": "Generate grounded answer",
            "result": "Answer generated with validated sources",
            "citation_status": citation_status,
            "validated_citations": len(cited_ids),
        }
    )

    return answer or NO_SOURCE_ANSWER, sources, activity
