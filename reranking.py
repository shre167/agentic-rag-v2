import math
import re


STOP_WORDS = {
    "a", "about", "above", "after", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "of", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def _stem(word: str) -> str:
    if word in {"manage", "manages", "managed", "managing", "management"}:
        return "manag"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in (
        "ingly", "edly", "ivity", "ing", "ed", "ment", "tion", "ive", "ity", "es", "s"
    ):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[: -len(suffix)]
            break
    return word


def _tokenize(text: str) -> list[str]:
    return [
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


def rerank_passages(question: str, passages: list[dict], limit: int = 5) -> list[dict]:
    """Rerank vector candidates with BM25 term relevance and semantic similarity."""
    if not passages:
        return []

    query_terms = _tokenize(question)
    if not query_terms:
        return passages[:limit]

    documents = [
        _tokenize(f"{passage.get('section', '')} {passage.get('text', '')}")
        for passage in passages
    ]
    document_count = len(documents)
    average_length = sum(map(len, documents)) / document_count or 1
    document_frequency = {
        term: sum(term in set(document) for document in documents)
        for term in set(query_terms)
    }

    scored = []
    for passage, document in zip(passages, documents):
        frequencies = {term: document.count(term) for term in set(query_terms)}
        bm25 = 0.0
        for term in set(query_terms):
            term_frequency = frequencies[term]
            if not term_frequency:
                continue
            inverse_frequency = math.log1p(
                (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            length_norm = 1.2 * (0.25 + 0.75 * len(document) / average_length)
            bm25 += inverse_frequency * (term_frequency * 2.2) / (term_frequency + length_norm)

        heading_terms = set(_tokenize(passage.get("section", "")))
        heading_match = len(set(query_terms) & heading_terms) / len(set(query_terms))
        distance = max(0.0, min(2.0, float(passage.get("distance", 2.0))))
        semantic_score = 1.0 - distance / 2.0
        scored.append((passage, bm25, heading_match, semantic_score))

    max_bm25 = max(score[1] for score in scored) or 1.0
    ranked = []
    for passage, bm25, heading_match, semantic_score in scored:
        score = 0.65 * (bm25 / max_bm25) + 0.25 * semantic_score + 0.10 * heading_match
        ranked.append(({**passage, "rerank_score": score}, score))

    ranked.sort(key=lambda result: result[1], reverse=True)
    return [passage for passage, _ in ranked[:limit]]
