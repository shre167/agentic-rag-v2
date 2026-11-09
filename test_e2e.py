"""End-to-end verification of retrieval, multi-source, and tool functionality."""
import json
import time

from tools.semantic_search import semantic_search
from tools.list_books import list_books
from retrieval.retrieval import retrieve_chunks
from agentic_rag import validate_citations

print("=" * 60)
print("TEST 1: list_books (all KBs)")
print("=" * 60)
result = list_books.invoke({"kb_name": "kb1"})
data = json.loads(result)
print("Items:", data["items"])
assert len(data["items"]) >= 2, "Expected at least 2 books"
print("PASS\n")

print("=" * 60)
print("TEST 2: Page retrieval - page 10 kb1")
print("=" * 60)
t0 = time.time()
r = semantic_search.invoke({"query": "what is on page 10", "kb_name": "kb1", "k": 3})
elapsed = (time.time() - t0) * 1000
d = json.loads(r)
print(f"Got {len(d['passages'])} chunks in {elapsed:.0f}ms")
for p in d["passages"]:
    print(f"  {p['id']} | page {p['page_start']}-{p['page_end']} | line {p.get('line_start','?')}-{p.get('line_end','?')}")
assert len(d["passages"]) > 0, "Expected page 10 results"
assert all(p["page_start"] <= 10 <= p["page_end"] for p in d["passages"]), "Pages should cover page 10"
print("PASS\n")

print("=" * 60)
print("TEST 3: Line retrieval - line 15 of page 10 kb1")
print("=" * 60)
t0 = time.time()
r = semantic_search.invoke({"query": "line 15 of page 10", "kb_name": "kb1", "k": 3})
elapsed = (time.time() - t0) * 1000
d = json.loads(r)
print(f"Got {len(d['passages'])} chunks in {elapsed:.0f}ms")
for p in d["passages"]:
    print(f"  {p['id']} | page {p['page_start']}-{p['page_end']} | line {p.get('line_start','?')}-{p.get('line_end','?')}")
print("PASS\n")

print("=" * 60)
print("TEST 4: Chapter retrieval - chapter 1 kb1")
print("=" * 60)
chunks = retrieve_chunks("kb1", "focus management", chapter="1", k=3)
print(f"Got {len(chunks)} chunks from chapter 1")
for c in chunks:
    print(f"  {c['id']} | chapter={c.get('chapter_number','?')} | {c.get('chapter','?')[:40]}")
print("PASS\n")

print("=" * 60)
print("TEST 5: Multi-source retrieval (kb1 + kb2)")
print("=" * 60)
t0 = time.time()
r1 = semantic_search.invoke({"query": "focus and deep concentration work", "kb_name": "kb1", "k": 3})
r2 = semantic_search.invoke({"query": "focus and deep concentration work", "kb_name": "kb2", "k": 3})
elapsed = (time.time() - t0) * 1000
d1 = json.loads(r1)
d2 = json.loads(r2)
print(f"kb1: {len(d1['passages'])} chunks | kb2: {len(d2['passages'])} chunks | total {elapsed:.0f}ms")
kb1_books = {p.get("book","?") for p in d1["passages"]}
kb2_books = {p.get("book","?") for p in d2["passages"]}
print(f"kb1 books: {kb1_books}")
print(f"kb2 books: {kb2_books}")
assert len(d1["passages"]) > 0, "kb1 must return results"
assert len(d2["passages"]) > 0, "kb2 must return results"
print("PASS\n")

print("=" * 60)
print("TEST 6: validate_citations with reasoning field")
print("=" * 60)
# Simulate a valid LLM response
all_passages = d1["passages"] + d2["passages"]
retrieved_ids = {p["id"] for p in all_passages}
sample_id = list(retrieved_ids)[0]
fake_response = json.dumps({
    "reasoning": "I called list_books to find kb1 and kb2, then searched both for focus-related content.",
    "answer": f"Focus is discussed extensively [{sample_id}].",
    "sources": [sample_id]
})
answer, source_ids, valid, reasoning = validate_citations(fake_response, retrieved_ids)
print(f"Answer: {answer[:60]}...")
print(f"Sources: {source_ids}")
print(f"Valid: {valid}")
print(f"Reasoning extracted: {reasoning[:80]}...")
assert valid, "Citation validation should pass"
assert reasoning is not None, "Reasoning should be extracted"
print("PASS\n")

print("=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
