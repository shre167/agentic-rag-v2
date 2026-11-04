import os

from dotenv import load_dotenv
from openai import OpenAI

from reranking import rerank_passages
from settings import ROOT
from vector_store import get_collection


load_dotenv(ROOT / ".env")

MAX_DISTANCE = 0.65
NO_SOURCE_ANSWER = "I couldn't find relevant information about that in the book."


def retrieve_chunks(
    question: str,
    k: int = 5,
    max_distance: float = MAX_DISTANCE,
    scope: str | None = None,
) -> list[dict]:
    """Return relevant passages, optionally limited to a chapter or exact section."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    where = None
    if scope:
        stored = collection.get(include=["metadatas"])
        requested = scope.strip().casefold()
        chapters = {
            metadata.get("chapter", "")
            for metadata in stored["metadatas"]
            if metadata.get("chapter", "").casefold() == requested
        }
        sections = {
            metadata.get("section", "")
            for metadata in stored["metadatas"]
            if metadata.get("section", "").casefold() == requested
        }
        conditions = ([{"chapter": {"$eq": value}} for value in chapters]
                      + [{"section": {"$eq": value}} for value in sections])
        if not conditions:
            return []
        where = conditions[0] if len(conditions) == 1 else {"$or": conditions}

    candidate_count = min(count, max(k * 4, 20))
    results = collection.query(
        query_texts=[question],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    passages = [
        {"id": chunk_id, "text": text, **metadata, "distance": distance}
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    relevant = [passage for passage in passages if passage["distance"] <= max_distance]
    return rerank_passages(question, relevant, limit=k)


def answer_question(
    question: str, history: list[dict] | None = None, k: int = 5
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and answer using the configured chat model."""
    passages = retrieve_chunks(question, k=k)
    if not passages:
        return NO_SOURCE_ANSWER, []

    api_key = os.getenv("GE_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if not all([api_key, base_url, model]):
        raise RuntimeError("Set GE_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env.")

    context = "\n\n".join(
        f"[Source {rank}: Chapter {source.get('chapter', 'Unknown')}, "
        f"{source.get('section', 'Unknown')}, "
        f"pages {source.get('page_start', '?')}-{source.get('page_end', '?')}]\n"
        f"{source['text']}"
        for rank, source in enumerate(passages, start=1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only the supplied PDF passages. Use conversation history "
                "to understand follow-up questions, but ground factual claims in the current "
                "passages. If the passages do not contain information that answers the question, "
                "reply with exactly [NO_SOURCES] and nothing else. Cite supporting passages as "
                "[Source N]."
            ),
        }
    ]
    messages.extend((history or [])[-6:])
    messages.append(
        {
            "role": "user",
            "content": f"Question: {question}\n\nPDF passages:\n{context}",
        }
    )

    response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
        model=model,
        messages=messages,
    )
    answer = response.choices[0].message.content or "No answer was returned."
    if answer.strip().upper() == "[NO_SOURCES]":
        return NO_SOURCE_ANSWER, []
    return answer, passages


def indexed_collection():
    """Get the collection for UI inspection without exposing Chroma setup details."""
    return get_collection()
