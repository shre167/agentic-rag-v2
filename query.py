from retrieval import answer_question

while True:
    question = input("\nSearch indexed PDF (or type exit): ").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue

    answer, sources = answer_question(question, k=3)
    print(f"\nAnswer:\n{answer}")
    for rank, source in enumerate(sources, start=1):
        print(
            f"\n[Source {rank}] {source['section']} "
            f"(pages {source['page_start']}-{source['page_end']}, "
            f"distance {source['distance']:.3f})"
        )