import argparse
from pathlib import Path

from chunking import extract_sections, make_chunks
from retrieval.vector_store import index_chunks
from settings import DEFAULT_PDF


def main():
    parser = argparse.ArgumentParser(
        description="Extract, chunk, and index a PDF into a specific knowledge base."
    )

    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF file to ingest",
    )

    parser.add_argument(
        "--book",
        help="Book name stored in metadata (defaults to PDF filename).",
    )

    parser.add_argument(
        "--kb",
        required=True,
        choices=["kb1", "kb2"],
        help="Target knowledge base.",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show chunk examples without indexing.",
    )

    args = parser.parse_args()

    pdf_path = args.pdf.resolve()

    if not pdf_path.exists():
        parser.error(f"PDF not found: {pdf_path}")

    book_name = (args.book or pdf_path.stem).strip()

    if not book_name:
        parser.error("Book name cannot be empty.")

    try:
        print(f"\nLoading PDF: {pdf_path.name}")
        print(f"Target KB: {args.kb}")
        print(f"Book Name: {book_name}")

        sections = extract_sections(
            pdf_path=pdf_path,
            book_name=book_name,
        )

        chunks = make_chunks(sections)

    except Exception as error:
        parser.error(f"Failed to process PDF: {error}")

    if not chunks:
        parser.error("No chunks were generated.")

    print(f"\nSections Extracted: {len(sections)}")
    print(f"Chunks Generated: {len(chunks)}")

    preview_chunks = [
        chunk
        for chunk in chunks
        if chunk[1]["chapter"] != "Front matter"
    ][:3]

    for idx, (text, metadata) in enumerate(preview_chunks, start=1):
        print("\n" + "=" * 80)
        print(f"Preview Chunk {idx}")
        print("=" * 80)

        print(f"Book      : {metadata['book']}")
        print(f"Chapter   : {metadata['chapter']}")
        print(f"Section   : {metadata['section']}")
        print(
            f"Pages     : "
            f"{metadata['page_start']} - {metadata['page_end']}"
        )
        if "line_start" in metadata:
            print(
                f"Lines     : "
                f"{metadata['line_start']} - {metadata['line_end']}"
            )

        print("\nText Preview:")
        print(text[:500])
        print()

    if args.preview:
        print("\nPreview mode enabled. No indexing performed.")
        return

    indexed_count = index_chunks(
        chunks=chunks,
        source=pdf_path,
        kb_name=args.kb,
    )

    print("\n" + "=" * 80)
    print(f"Indexed {indexed_count} chunks")
    print(f"Knowledge Base : {args.kb}")
    print(f"Book           : {book_name}")
    print("=" * 80)

    print("\nLaunch:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()
