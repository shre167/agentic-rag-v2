import argparse
import re
from pathlib import Path

from chunking import extract_sections, make_chunks
from retrieval.vector_store import get_collection, index_chunks, list_knowledge_bases
from settings import DEFAULT_PDF


def get_next_kb_name() -> str:
    """
    Find the next available knowledge base ID (e.g. kb1, kb2, kb3...).
    Ensures every new book receives its own isolated knowledge base automatically.
    """
    existing = list_knowledge_bases()
    nums = []
    for name in existing:
        m = re.match(r"^kb(\d+)$", name.strip(), re.IGNORECASE)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums, default=0) + 1
    return f"kb{next_num}"


def find_kb_for_book(book_name: str) -> str | None:
    """Check if a book is already indexed in an existing knowledge base."""
    target_clean = book_name.strip().casefold()
    for kb in list_knowledge_bases():
        try:
            col = get_collection(kb)
            col_book = (getattr(col, "metadata", None) or {}).get("book")
            if col_book and col_book.strip().casefold() == target_clean:
                return kb
            sample = col.get(limit=3, include=["metadatas"])["metadatas"]
            for m in sample:
                if m.get("book", "").strip().casefold() == target_clean:
                    return kb
        except Exception:
            continue
    return None


def ingest_pdf(
    pdf_path: Path | str,
    book_name: str | None = None,
    kb_name: str | None = None,
    preview: bool = False,
) -> dict:
    """
    Extract, chunk with structure-aware metadata, and index a PDF into a dedicated knowledge base.

    Args:
        pdf_path: Path to the PDF file.
        book_name: Human-readable book title (defaults to PDF stem).
        kb_name: Target knowledge base (e.g. 'kb1', 'kb2', 'kb3').
                 If None, automatically reuses existing KB for this book or creates the next 'kbN'.
        preview: If True, extracts and chunks without writing to ChromaDB.

    Returns:
        dict: Ingestion summary including kb_name, book_name, chunks_count, and sections_count.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Determine clean book name
    if not book_name:
        book_name = pdf_path.stem.replace("_", " ").strip()
    book_name = book_name.strip()
    if not book_name:
        raise ValueError("Book name cannot be empty.")

    # Determine or auto-assign knowledge base
    if not kb_name:
        existing_kb = find_kb_for_book(book_name)
        kb_name = existing_kb if existing_kb else get_next_kb_name()
    kb_name = kb_name.strip()

    # Extract sections with chapter, heading, page, and line metadata
    sections = extract_sections(pdf_path=pdf_path, book_name=book_name)
    if not sections:
        raise ValueError("No text sections could be extracted from the PDF.")

    # Generate metadata-rich chunks
    chunks = make_chunks(sections)
    if not chunks:
        raise ValueError("No chunks were generated from the extracted sections.")

    indexed_count = 0
    if not preview:
        indexed_count = index_chunks(
            chunks=chunks,
            source=pdf_path,
            kb_name=kb_name,
        )

    return {
        "kb_name": kb_name,
        "book_name": book_name,
        "chunks_count": len(chunks),
        "indexed_count": indexed_count,
        "sections_count": len(sections),
        "source": str(pdf_path),
        "preview": preview,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract, chunk, and index a PDF into an isolated knowledge base."
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
        help="Target knowledge base (optional: defaults to next auto-assigned kb, e.g. kb3).",
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

    try:
        print(f"\nProcessing PDF: {pdf_path.name}")
        summary = ingest_pdf(
            pdf_path=pdf_path,
            book_name=args.book,
            kb_name=args.kb,
            preview=args.preview,
        )

        print(f"Target KB       : {summary['kb_name']}")
        print(f"Book Name       : {summary['book_name']}")
        print(f"Sections Count  : {summary['sections_count']}")
        print(f"Chunks Count    : {summary['chunks_count']}")

        if summary["preview"]:
            print("\nPreview mode enabled. No indexing performed.")
        else:
            print(f"Indexed Chunks  : {summary['indexed_count']}")
            print("=" * 60)
            print("✅ Ingestion complete. Knowledge base is ready for agentic retrieval.")
            print("=" * 60)

    except Exception as error:
        parser.error(f"Ingestion failed: {error}")


if __name__ == "__main__":
    main()
