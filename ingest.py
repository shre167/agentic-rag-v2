import argparse
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from chunking import extract_sections, make_chunks
from retrieval.vector_store import (
    get_collection,
    index_chunks,
    list_knowledge_bases,
    delete_knowledge_base,
)
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
    overwrite: bool = False,
    progress_callback: Any = None,
) -> dict:
    """
    Extract, chunk with structure-aware metadata, and index a PDF into a dedicated knowledge base.
    Also generates an LLM-based book profile alongside the vector indexing.

    Pipeline:
      1. Document processing: Extraction → Chunking → Embeddings → Vector DB (per-book collection)
      2. Book profiling: Extracted content → LLM analysis → Summary + topic profile → KB Registry

    Args:
        pdf_path: Path to the PDF file.
        book_name: Human-readable book title (defaults to PDF stem).
        kb_name: Target knowledge base (e.g. 'kb1', 'kb2', 'kb3').
                 If None, automatically reuses existing KB for this book or creates the next 'kbN'.
        preview: If True, extracts and chunks without writing to ChromaDB.
        overwrite: If True, removes existing chunks and profile for this KB before re-indexing.
        progress_callback: Optional callable(str) receiving live status updates for the entire procedure.

    Returns:
        dict: Ingestion summary including kb_name, book_name, chunks_count, sections_count,
              page_count, chapters_count, profile_generated status, and the generated profile dict.
    """
    def _notify(msg: str):
        if callable(progress_callback):
            progress_callback(msg)
        print(msg)

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    _notify(f"File verified: {pdf_path.name} ({file_size_mb:.1f} MB)")

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
    _notify(f"Assigned Knowledge Base: '{kb_name}' for '{book_name}'")

    t_start = time.perf_counter()
    parse_sec = 0.0
    chunk_sec = 0.0
    index_sec = 0.0
    profile_sec = 0.0

    # If overwrite requested and KB already exists, purge it completely first
    if not preview and overwrite and kb_name in list_knowledge_bases():
        _notify(f"Purging existing collection '{kb_name}' and previous chunks...")
        delete_knowledge_base(kb_name)

    # Extract sections with chapter, heading, page, and line metadata
    _notify(f"Parsing PDF document and extracting structural sections...")
    t_parse = time.perf_counter()
    sections = extract_sections(pdf_path=pdf_path, book_name=book_name)
    if not sections:
        raise ValueError("No text sections could be extracted from the PDF.")
    parse_sec = round(time.perf_counter() - t_parse, 2)

    total_pages = max((s.page_end for s in sections), default=0)
    unique_chapters = sorted({s.chapter for s in sections if s.chapter and s.chapter != "Front matter"})
    _notify(f"Extracted {len(sections)} sections across {total_pages} pages ({len(unique_chapters)} chapters identified) in {parse_sec}s.")

    # Generate metadata-rich chunks
    _notify("Chunking sections into hierarchical passages with line-level coordinates...")
    t_chunk = time.perf_counter()
    chunks = make_chunks(sections)
    if not chunks:
        raise ValueError("No chunks were generated from the extracted sections.")
    chunk_sec = round(time.perf_counter() - t_chunk, 2)
    _notify(f"Generated {len(chunks)} chunks with line-level and page-level metadata in {chunk_sec}s.")

    indexed_count = 0
    profile_generated = False
    profile = None

    if not preview:
        # Pipeline 1: Index chunks into the book's dedicated vector DB collection
        _notify(f"Indexing {len(chunks)} chunks into ChromaDB collection '{kb_name}' (Cosine space)...")
        t_index = time.perf_counter()
        indexed_count = index_chunks(
            chunks=chunks,
            source=pdf_path,
            kb_name=kb_name,
        )
        index_sec = round(time.perf_counter() - t_index, 2)
        _notify(f"Stored {indexed_count} vector records into collection '{kb_name}' in {index_sec}s.")

        # Pipeline 2: Generate LLM-based book profile alongside vector indexing
        t_profile = time.perf_counter()
        try:
            from retrieval.kb_registry import generate_book_profile, clear_kb_registry_cache
            try:
                from tools.list_books import clear_book_cache
                clear_book_cache()
            except ImportError:
                pass

            # Clear cache so the new book is discoverable
            clear_kb_registry_cache()

            _notify("Sampling representative excerpts from across the book for profiling...")
            _notify("Running LLM analysis to generate summary, main topics, and key concepts...")
            profile = generate_book_profile(
                kb_name=kb_name,
                book_name=book_name,
                collection_name=kb_name,
            )
            profile_sec = round(time.perf_counter() - t_profile, 2)
            profile_generated = profile is not None
            if profile_generated:
                _notify(f"Content profile generated and saved for '{book_name}' in {profile_sec}s.")
            else:
                _notify(f"Profile generated using metadata fallback in {profile_sec}s.")
        except Exception as e:
            profile_sec = round(time.perf_counter() - t_profile, 2)
            _notify(f"Profile generation note: {e}")
            profile_generated = False

        _notify(f"Invalidated router, discovery, and metadata caches.")
        _notify(f"Knowledge base '{kb_name}' is fully active and ready for routing.")

    total_sec = round(time.perf_counter() - t_start, 2)
    _notify(f"Total ingestion latency: {total_sec}s (Parse: {parse_sec}s | Chunk: {chunk_sec}s | Index: {index_sec}s | Profile: {profile_sec}s)")

    latencies = {
        "parse_sec": parse_sec,
        "chunk_sec": chunk_sec,
        "index_sec": index_sec,
        "profile_sec": profile_sec,
        "total_sec": total_sec,
    }

    if profile:
        profile["ingestion_latency_sec"] = total_sec
        profile["ingestion_latencies"] = latencies
        try:
            from retrieval.kb_registry import _load_profiles, _save_profiles
            all_profs = _load_profiles()
            if kb_name in all_profs:
                all_profs[kb_name]["ingestion_latency_sec"] = total_sec
                all_profs[kb_name]["ingestion_latencies"] = latencies
                _save_profiles(all_profs)
        except Exception:
            pass

    return {
        "kb_name": kb_name,
        "book_name": book_name,
        "chunks_count": len(chunks),
        "indexed_count": indexed_count,
        "sections_count": len(sections),
        "chapters_count": len(unique_chapters),
        "page_count": total_pages,
        "source": str(pdf_path),
        "preview": preview,
        "profile_generated": profile_generated,
        "profile": profile,
        "total_latency_sec": total_sec,
        "latencies": latencies,
    }


def register_existing_books() -> list[dict]:
    """
    Generate profiles for all previously ingested books that don't have profiles yet.
    Does NOT re-index or duplicate any vector records.
    """
    from retrieval.kb_registry import (
        get_kb_registry,
        generate_book_profile,
        _load_profiles,
        clear_kb_registry_cache,
    )

    clear_kb_registry_cache()
    existing_profiles = _load_profiles()
    results = []

    for kb_name in list_knowledge_bases():
        if kb_name in existing_profiles:
            results.append({
                "kb_name": kb_name,
                "status": "already_profiled",
                "book_name": existing_profiles[kb_name].get("kb_name", "Unknown"),
            })
            continue

        try:
            col = get_collection(kb_name)
            count = col.count()
            if count == 0:
                continue

            # Get book name from metadata
            sample = col.get(limit=5, include=["metadatas"])["metadatas"]
            titles = sorted({m.get("book", "") for m in sample if m.get("book")})
            book_title = titles[0] if titles else kb_name

            profile = generate_book_profile(
                kb_name=kb_name,
                book_name=book_title,
                collection_name=kb_name,
            )

            results.append({
                "kb_name": kb_name,
                "status": "profile_generated" if profile else "failed",
                "book_name": book_title,
            })
        except Exception as e:
            results.append({
                "kb_name": kb_name,
                "status": f"error: {e}",
                "book_name": "Unknown",
            })

    clear_kb_registry_cache()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract, chunk, and index a PDF into an isolated knowledge base with LLM profiling."
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe and overwrite the knowledge base if it already exists.",
    )

    parser.add_argument(
        "--delete-kb",
        metavar="KB_NAME",
        help="Permanently delete a knowledge base and all its chunks (e.g. --delete-kb kb3).",
    )

    parser.add_argument(
        "--register-existing",
        action="store_true",
        help="Generate profiles for all previously ingested books (no re-indexing).",
    )

    args = parser.parse_args()

    if args.delete_kb:
        target = args.delete_kb.strip()
        print(f"\nDeleting knowledge base: '{target}'...")
        # Check if target is a book title rather than KB ID
        from retrieval.kb_registry import get_kb_registry
        reg = get_kb_registry()
        if target not in reg:
            for kid, prof in reg.items():
                if prof.get("kb_name", "").strip().casefold() == target.casefold():
                    target = kid
                    break
        success = delete_knowledge_base(target)
        if success:
            print(f"✅ Successfully deleted '{target}' and purged all associated chunks, profiles, and caches.")
        return

    if args.register_existing:
        print("\nRegistering existing books...")
        results = register_existing_books()
        for r in results:
            print(f"  {r['kb_name']}: {r['book_name']} → {r['status']}")
        print(f"\n✅ Processed {len(results)} knowledge base(s).")
        return

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
            overwrite=args.overwrite,
        )

        print(f"Target KB       : {summary['kb_name']}")
        print(f"Book Name       : {summary['book_name']}")
        print(f"Sections Count  : {summary['sections_count']}")
        print(f"Chunks Count    : {summary['chunks_count']}")

        if summary["preview"]:
            print("\nPreview mode enabled. No indexing performed.")
        else:
            print(f"Indexed Chunks  : {summary['indexed_count']}")
            print(f"Profile Generated: {'✅ Yes' if summary['profile_generated'] else '⚠️ No (fallback metadata used)'}")
            print("=" * 60)
            print("✅ Ingestion complete. Knowledge base is ready for agentic retrieval.")
            print("=" * 60)

    except Exception as error:
        parser.error(f"Ingestion failed: {error}")


if __name__ == "__main__":
    main()
