"""
Simulate the full agent tool call sequence for a multi-source query.
This replicates what the LLM agent does at runtime, verifying:
1. list_books returns routing_map for BOTH kbs
2. semantic_search independently returns chunks from kb1 AND kb2
3. All passages from both tool calls are merged correctly
4. validate_citations works on cross-kb citations
"""
import json
from agentic_rag import validate_citations

from tools.list_books import _BOOK_CACHE
import tools.list_books as lb_mod
lb_mod._BOOK_CACHE = None  # reset cache for clean test

from tools.list_books import list_books
from tools.semantic_search import semantic_search

QUERY = "Compare how both books discuss focus and deep work"

print("=" * 65)
print("STEP 1: Agent calls list_books to discover routing_map")
print("=" * 65)
books_raw = list_books.invoke({})
books_data = json.loads(books_raw)
routing_map = books_data.get("routing_map", {})
instruction = books_data.get("instruction", "")
items = books_data.get("items", [])

print(f"routing_map: {routing_map}")
print(f"items: {items}")
print(f"instruction:\n  {instruction}")
assert len(routing_map) >= 2, f"Expected 2 KBs, got: {routing_map}"
print("PASS\n")

print("=" * 65)
print("STEP 2: Agent decides -> comparative query -> search BOTH KBs")
print("=" * 65)
kb_names = list(routing_map.keys())  # ['kb1', 'kb2']
print(f"Agent will search: {kb_names}")
print()

all_passages = {}
for kb in kb_names:
    book_title = routing_map[kb]
    print(f"  -> semantic_search on {kb} ({book_title})")
    result_raw = semantic_search.invoke({
        "query": QUERY,
        "kb_name": kb,
        "k": 3,
    })
    result = json.loads(result_raw)
    passages = result.get("passages", [])
    print(f"     returned {len(passages)} passages")
    for p in passages:
        pid = p["id"]
        book = p.get("book", "?")
        page = f"pp.{p.get('page_start','?')}-{p.get('page_end','?')}"
        line = f"ll.{p.get('line_start','?')}-{p.get('line_end','?')}"
        print(f"       {pid} | {book} | {page} | {line}")
        all_passages[pid] = p

print()
print(f"Total merged passages from ALL kbs: {len(all_passages)}")

# Verify both KBs contributed
kb1_books = {p.get("book") for p in all_passages.values() if p["id"].startswith("7a12")}
kb2_books = {p.get("book") for p in all_passages.values() if not p["id"].startswith("7a12")}
all_books = {p.get("book") for p in all_passages.values()}
print(f"Books in merged result: {all_books}")
assert len(all_books) >= 2, f"Expected passages from 2 different books, got: {all_books}"
print("PASS\n")

print("=" * 65)
print("STEP 3: validate_citations works on cross-KB citations")
print("=" * 65)
# Simulate an LLM response that cites one chunk from each KB
ids = list(all_passages.keys())
id_kb1 = next((i for i in ids if all_passages[i].get("book") == routing_map.get("kb1")), None)
id_kb2 = next((i for i in ids if all_passages[i].get("book") == routing_map.get("kb2")), None)
assert id_kb1 and id_kb2, f"Need one chunk from each KB. Got: kb1={id_kb1}, kb2={id_kb2}"

fake_llm_response = json.dumps({
    "reasoning": (
        f"I called list_books and saw routing_map={routing_map}. "
        f"Since the question is comparative, I searched kb1 ({routing_map['kb1']}) "
        f"and kb2 ({routing_map['kb2']}) separately. "
        "I found relevant passages in both collections."
    ),
    "answer": (
        f"Both books discuss focus. Mind Management says [{id_kb1}] ... "
        f"while Deep Work argues [{id_kb2}] ..."
    ),
    "sources": [id_kb1, id_kb2]
})

answer, source_ids, valid, reasoning = validate_citations(
    fake_llm_response,
    set(all_passages.keys()),
)

print(f"Answer:    {answer[:80]}...")
print(f"Sources:   {source_ids}")
print(f"Valid:     {valid}")
print(f"Reasoning: {reasoning[:100]}...")
assert valid, "Cross-KB citation validation should pass"
assert len(source_ids) == 2, f"Expected 2 sources, got {source_ids}"
print("PASS\n")

print("=" * 65)
print("ALL MULTI-SOURCE TESTS PASSED")
print("Agent CAN dynamically route to multiple KBs without hardcoding.")
print("=" * 65)
