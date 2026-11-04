import argparse
from pathlib import Path

from chunking import extract_sections, make_chunks
from settings import DEFAULT_PDF
from vector_store import index_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Extract, chunk, and index a PDF in the local ChromaDB."
    )
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show chunk examples without writing to ChromaDB.",
    )
    args = parser.parse_args()
    pdf_path = args.pdf.resolve()

    if not pdf_path.is_file():
        parser.error(f"PDF not found: {pdf_path}")

    try:
        sections = extract_sections(pdf_path)
        chunks = make_chunks(sections)
    except Exception as error:
        parser.error(f"Could not read PDF: {error}")

    if not chunks:
        parser.error("No text chunks were extracted from the PDF.")

    print(f"PDF: {pdf_path.name}")
    print(f"Sections: {len(sections)} | Chunks: {len(chunks)}")

    examples = [
        chunk
        for chunk in chunks
        if chunk[1]["chapter"] != "Front matter" and chunk[1]["page_start"] > 10
    ][:3]
    for index, (text, metadata) in enumerate(examples, start=1):
        print(
            f"\nChunk {index} | {metadata['chapter']} | {metadata['section']} "
            f"| pages {metadata['page_start']}-{metadata['page_end']}"
        )
        print(text[:500])

    if args.preview:
        return

    indexed_count = index_chunks(chunks, pdf_path)
    print(f"\nIndexed {indexed_count} chunks for {pdf_path.name}.")
    print("Open the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
