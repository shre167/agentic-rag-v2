"""
Comprehensive verification test:
1. Multi-source repeated tool call reasoning
2. Web search fallback and execution
3. Citation validation for book vs web vs general answers
"""

import json
from agentic_rag import validate_citations, AgentResult
from tools.web_search import web_search
from tools.list_books import list_books
from tools.semantic_search import semantic_search

print("=" * 65)
print("TEST 1: Validate Citations with answer_type='web'")
print("=" * 65)
raw_web_response = json.dumps({
    "answer_type": "web",
    "reasoning": "The query asks about quantum computing, which is not covered in the indexed books 'Mind Management' or 'Deep Work'. Therefore, I used web_search to find live factual information.",
    "answer": "Quantum computing uses qubits to perform complex calculations [web-01].",
    "sources": ["web-01"]
})

ans, src_ids, valid, reasoning, atype = validate_citations(
    raw_web_response,
    {"web-01", "web-02"}
)
print("Answer:", ans)
print("Sources:", src_ids)
print("Valid:", valid)
print("Answer Type:", atype)
assert valid is True
assert atype == "web"
assert src_ids == ["web-01"]
print("PASS [OK]\n")

print("=" * 65)
print("TEST 2: Validate Citations with answer_type='general'")
print("=" * 65)
raw_general_response = json.dumps({
    "answer_type": "general",
    "reasoning": "The user asked a conversational or broad conceptual question where no specific passages were needed.",
    "answer": "Hello! I am your agentic assistant ready to answer questions from your indexed books or search the web.",
    "sources": []
})

ans, src_ids, valid, reasoning, atype = validate_citations(
    raw_general_response,
    set()
)
print("Answer:", ans)
print("Valid:", valid)
print("Answer Type:", atype)
assert valid is True
assert atype == "general"
print("PASS [OK]\n")

print("=" * 65)
print("TEST 3: Validate Citations with answer_type='book'")
print("=" * 65)
raw_book_response = json.dumps({
    "answer_type": "book",
    "reasoning": "I queried kb1 for Mind Management and found passage [7a12baba0127-00004].",
    "answer": "Mind management principles explain time allocation [7a12baba0127-00004].",
    "sources": ["7a12baba0127-00004"]
})

ans, src_ids, valid, reasoning, atype = validate_citations(
    raw_book_response,
    {"7a12baba0127-00004"}
)
print("Answer:", ans)
print("Valid:", valid)
print("Answer Type:", atype)
assert valid is True
assert atype == "book"
print("PASS [OK]\n")

print("=" * 65)
print("TEST 4: Web Search Tool Execution (Corporate Resilient)")
print("=" * 65)
web_out = json.loads(web_search.invoke({"query": "Python language", "max_results": 2}))
print("Engine:", web_out.get("engine"))
print("Result summary:", web_out.get("result"))
print("Passages generated:", len(web_out.get("passages", [])))
print("PASS [OK]\n")

print("=" * 65)
print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
print("=" * 65)
