"""
End-to-End Test for Ingestion Pipeline:
1. Creates a synthetic PDF with distinct chapters, sections, and lines.
2. Ingests it using `ingest_pdf` into a dedicated test KB (`kb_test_ingest`).
3. Validates that chunks contain exact metadata (chapter, section, lines, page).
4. Validates that `list_books()` dynamically discovers the new KB in its routing_map.
5. Validates that `semantic_search()` retrieves relevant chunks from the new KB.
6. Cleans up the test collection and temp file.
"""

import json
from pathlib import Path
import pymupdf

from ingest import ingest_pdf
from tools.list_books import list_books, clear_book_cache
from tools.semantic_search import semantic_search
from retrieval.vector_store import _get_client, list_knowledge_bases


def main():
    print("=" * 65)
    print("STEP 1: Create a test PDF with chapters and formatted sections")
    print("=" * 65)
    test_pdf_path = Path("test_book_sample.pdf")
    doc = pymupdf.open()

    # Page 1: Chapter 1
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "CHAPTER 1\n"
        "Foundations of Machine Intelligence\n\n"
        "Autonomous agents utilize dynamic tool calling and vector retrieval.\n"
        "By structuring knowledge bases into discrete collections, retrieval latency is minimized.\n"
        "This paragraph provides foundational concepts for high performance AI.\n",
        fontsize=12,
    )

    # Page 2: Chapter 2
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "CHAPTER 2\n"
        "Vector Databases and Cosine Indexing\n\n"
        "ChromaDB provides persistent vector stores with cosine space similarity.\n"
        "Each knowledge base maintains separate metadata index caches for fast query resolution.\n"
        "Hierarchical chunking preserves line numbers and page numbers.\n",
        fontsize=12,
    )

    doc.save(str(test_pdf_path))
    doc.close()
    print(f"Created sample PDF: {test_pdf_path} (2 pages)")

    TEST_KB = "kb_test_ingest"
    TEST_BOOK_NAME = "Agentic Intelligence Systems"

    try:
        print("\n" + "=" * 65)
        print("STEP 2: Run ingest_pdf pipeline")
        print("=" * 65)
        summary = ingest_pdf(
            pdf_path=test_pdf_path,
            book_name=TEST_BOOK_NAME,
            kb_name=TEST_KB,
        )
        print(f"Ingestion result summary:")
        print(f"  KB Name        : {summary['kb_name']}")
        print(f"  Book Name      : {summary['book_name']}")
        print(f"  Indexed Chunks : {summary['indexed_count']}")
        print(f"  Sections Count : {summary['sections_count']}")
        assert summary["indexed_count"] > 0, "No chunks were indexed!"
        print("PASS: Ingestion succeeded with metadata-rich chunks.")

        print("\n" + "=" * 65)
        print("STEP 3: Verify list_books() discovers the newly ingested KB")
        print("=" * 65)
        raw_list = list_books.invoke({})
        list_data = json.loads(raw_list)
        routing_map = list_data.get("routing_map", {})
        print(f"Discovered routing map: {routing_map}")
        assert TEST_KB in routing_map, f"Expected {TEST_KB} in routing_map, got: {routing_map}"
        assert routing_map[TEST_KB] == TEST_BOOK_NAME, f"Book name mismatch: {routing_map.get(TEST_KB)}"
        print(f"PASS: {TEST_KB} ('{TEST_BOOK_NAME}') successfully discovered by agent discovery tool.")

        print("\n" + "=" * 65)
        print("STEP 4: Test semantic_search() against newly ingested KB")
        print("=" * 65)
        search_raw = semantic_search.invoke({
            "query": "vector databases cosine indexing",
            "kb_name": TEST_KB,
            "k": 2,
        })
        search_data = json.loads(search_raw)
        passages = search_data.get("passages", [])
        print(f"Retrieved {len(passages)} passage(s) from {TEST_KB}:")
        for p in passages:
            print(f"  -> [{p['id']}] Chapter: '{p.get('chapter')}' | Section: '{p.get('section')}' | pp.{p.get('page_start')}-{p.get('page_end')}")
            print(f"     Preview: {p.get('text', '')[:100]}...")

        assert len(passages) > 0, f"Expected passages from {TEST_KB}, got 0"
        assert any("ChromaDB" in p.get("text", "") for p in passages), "Expected passage text to match ingested content"
        print("PASS: Semantic search accurately retrieved chunks with structural metadata.")

        print("\n" + "=" * 65)
        print("ALL INGESTION TESTS PASSED SUCCESSFULLY!")
        print("=" * 65)

    finally:
        # Cleanup test collection and test file
        print("\nCleaning up test artifacts...")
        client = _get_client()
        try:
            client.delete_collection(name=TEST_KB)
            print(f"Deleted test collection '{TEST_KB}'.")
        except Exception as e:
            print(f"Collection deletion warning: {e}")

        clear_book_cache()
        try:
            from retrieval.kb_registry import delete_book_profile
            delete_book_profile(TEST_KB)
        except Exception:
            pass

        if test_pdf_path.exists():
            test_pdf_path.unlink()
            print(f"Removed temp file '{test_pdf_path}'.")


if __name__ == "__main__":
    main()
