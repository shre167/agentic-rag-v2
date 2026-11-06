from langchain.tools import tool

from retrieval.retrieval import retrieve_chunks


@tool
def search_knowledge_base(
    kb_name: str,
    query: str,
    k: int = 5,
) -> str:
    """
    Search a specific knowledge base.

    Available knowledge bases:

    kb1 -> Mind Management
    kb2 -> Deep Work
    """

    passages = retrieve_chunks(
        kb_name=kb_name,
        query=query,
        k=k,
    )

    if not passages:
        return (
            f"NO_RELEVANT_PASSAGES: "
            f"no evidence found in {kb_name}"
        )

    results = []

    for passage in passages:
        results.append(
            f"[{passage['id']}]\n"
            f"{passage['text']}"
        )

    return "\n\n".join(results)