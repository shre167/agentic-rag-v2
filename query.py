from agentic_rag import answer_with_agent

while True:
    question = input("\nSearch indexed PDF (or type exit): ").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue

    answer, sources, activity = answer_with_agent(question, k=3)
    print(f"\nAnswer:\n{answer}")
    for step in activity:
        print(
            f"\n{step.get('tool', step.get('action', 'Step'))}: "
            f"{step.get('input', '')} -> {step.get('result', '')}"
        )
    for rank, source in enumerate(sources, start=1):
        print(
            f"\n[Source {rank}] {source['section']} "
            f"(pages {source['page_start']}-{source['page_end']}, "
            f"distance {source['distance']:.3f})"
        )