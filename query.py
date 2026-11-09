"""
CLI Query Runner
================
Interactive command-line testing tool for the agentic RAG system.
Demonstrates:
- Dynamic collection routing
- Multi-source repeated tool call reasoning
- Web search for external/irrelevant queries
- Answer type classification (book vs web vs general vs mixed)
"""

from agentic_rag import answer_with_agent

print("=" * 70)
print("📚 Autonomous Agentic Book & Web Assistant (CLI)")
print("Type any question across indexed books or external topics (or 'exit' to quit)")
print("=" * 70)

while True:
    question = input("\nAsk question: ").strip()
    if question.lower() in {"exit", "quit", "q"}:
        break
    if not question:
        continue

    print("\n[Agent thinking & retrieving]...")
    result = answer_with_agent(question, k=3, kb_name=None)
    answer = result.answer
    sources = result.sources
    activity = result.activity
    reasoning = getattr(result, "reasoning", None)
    answer_type = getattr(result, "answer_type", "book")

    print("\n" + "=" * 70)
    print(f"📌 ANSWER TYPE: {answer_type.upper()}")
    print("=" * 70)

    if reasoning:
        print(f"\n🧠 [Agent Dynamic Reasoning & Strategy]:\n{reasoning}\n")

    print(f"💬 [Synthesized Answer]:\n{answer}\n")

    print("-" * 70)
    print("⚙️ [Tool Execution Trace & Diagnostic]:")
    print("-" * 70)
    for step in activity:
        if step.get("action") == "Agent Dynamic Reasoning & Strategy":
            continue

        tool_name = step.get("tool", step.get("action", "Step"))
        step_num = step.get("step", "?")
        print(f"\nStep {step_num}: `{tool_name}`")

        if step.get("tool_reasoning"):
            print(f"  🎯 Why called: {step['tool_reasoning']}")
        if step.get("why_repeated"):
            print(f"  🔁 Why repeated: {step['why_repeated']}")
        if step.get("why_web_instead_of_semantic"):
            print(f"  🌐 Why web search: {step['why_web_instead_of_semantic']}")
        if step.get("input"):
            print(f"  📤 Input: {step['input']}")
        if step.get("found_summary"):
            print(f"  📥 Found: {step['found_summary']}")
        if step.get("what_found_items"):
            for item in step["what_found_items"][:3]:
                print(f"      • {item}")

    if sources:
        print("\n" + "-" * 70)
        print("📚 [Retrieved Sources & Citations]:")
        print("-" * 70)
        for rank, source in enumerate(sources, start=1):
            if source.get("source") == "web":
                print(f"  [{source['id']}] (Web) {source.get('title','')} — {source.get('url','')}")
            else:
                lines_info = (
                    f", lines {source['line_start']}-{source['line_end']}"
                    if "line_start" in source
                    else ""
                )
                print(
                    f"  [{source['id']}] {source.get('book', 'Book')} — {source.get('section', '')} "
                    f"(Chapter {source.get('chapter', '')}, pages {source['page_start']}-{source['page_end']}{lines_info})"
                )
