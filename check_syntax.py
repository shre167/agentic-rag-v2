import ast

for fname in ["app.py", "agentic_rag.py", "tools/web_search.py", "agent/prompts.py", "agent/bookagent.py"]:
    with open(fname, encoding="utf-8") as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"OK: {fname}")
    except SyntaxError as e:
        print(f"FAIL: {fname} -> {e}")
