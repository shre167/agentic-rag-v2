"""Profile every stage of a multi-source agent response."""
import time

stages = {}

# --- Stage 1: imports / cold start ---
t0 = time.time()
from tools.list_books import list_books
import tools.list_books as lb_mod
lb_mod._BOOK_CACHE = None
from tools.semantic_search import semantic_search
import json
stages["imports"] = time.time() - t0

# --- Stage 2: list_books (1st LLM tool call) ---
t0 = time.time()
books_raw = list_books.invoke({})
stages["list_books"] = time.time() - t0

# --- Stage 3: semantic_search kb1 (embedding + chroma query) ---
t0 = time.time()
r1 = semantic_search.invoke({"query": "focus and deep work", "kb_name": "kb1", "k": 3})
stages["search_kb1_cold"] = time.time() - t0

# --- Stage 4: semantic_search kb2 (embedding cached, chroma query) ---
t0 = time.time()
r2 = semantic_search.invoke({"query": "focus and deep work", "kb_name": "kb2", "k": 3})
stages["search_kb2_warm"] = time.time() - t0

# --- Stage 5: second search kb1 (fully warm) ---
t0 = time.time()
r3 = semantic_search.invoke({"query": "time management strategies", "kb_name": "kb1", "k": 3})
stages["search_kb1_warm"] = time.time() - t0

# --- Stage 6: parallel simulation ---
import concurrent.futures
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    f1 = ex.submit(semantic_search.invoke, {"query": "distraction", "kb_name": "kb1", "k": 3})
    f2 = ex.submit(semantic_search.invoke, {"query": "distraction", "kb_name": "kb2", "k": 3})
    concurrent.futures.wait([f1, f2])
stages["parallel_both_kbs"] = time.time() - t0

print("\n=== LATENCY PROFILE ===")
for k, v in stages.items():
    bar = "#" * int(v * 10)
    print(f"  {k:<25} {v*1000:>7.0f}ms  {bar}")

d1 = json.loads(r1)
d2 = json.loads(r2)
print(f"\nkb1 chunks: {len(d1['passages'])}  kb2 chunks: {len(d2['passages'])}")
total_sequential = stages["list_books"] + stages["search_kb1_cold"] + stages["search_kb2_warm"]
print(f"\nSequential (list+kb1+kb2): {total_sequential*1000:.0f}ms")
print(f"Parallel   (kb1 || kb2):   {stages['parallel_both_kbs']*1000:.0f}ms")
print(f"Saving with parallelism:   {(stages['search_kb1_cold']+stages['search_kb2_warm'] - stages['parallel_both_kbs'])*1000:.0f}ms")
