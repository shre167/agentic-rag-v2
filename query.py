from agentic_rag import answer_with_agent

while True:
    question = input("\nAsk question across all books (or type exit): ").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue

    result = answer_with_agent(question, k=3, kb_name=None)
    answer = result[0]
    sources = result[1]
    activity = result[2]
    reasoning = getattr(result, "reasoning", None)
    if not reasoning:
        for step in activity:
            if step.get("reasoning"):
                reasoning = step["reasoning"]
                break

    if reasoning:
        print(f"\n[Agent Strategy & Dynamic Reasoning]:\n{reasoning}\n")

    print(f"Answer:\n{answer}")
    for step in activity:
        if step.get("reasoning"):
            continue
        print(
            f"\n{step.get('tool', step.get('action', 'Step'))}: "
            f"{step.get('input', '')} -> {step.get('result', '')}"
        )
    for rank, source in enumerate(sources, start=1):
        line_info = (
            f", lines {source['line_start']}-{source['line_end']}"
            if "line_start" in source
            else ""
        )
        print(
            f"\n[Source {rank}] {source.get('book', 'Book')} - {source['section']} "
            f"(pages {source['page_start']}-{source['page_end']}{line_info}, "
            f"distance {source['distance']:.3f})"
        )
