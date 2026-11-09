"""PDF text extraction with structure-aware chunking."""

import re
import statistics
from pathlib import Path
from dataclasses import dataclass

import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class Section:
    text: str
    book: str
    chapter: str
    chapter_number: int  # NEW: Numeric chapter for easy filtering
    heading: str
    page_start: int
    page_end: int
    page_texts: list[tuple[int, str]]


def extract_sections(pdf_path: Path, book_name: str) -> list[Section]:
    """Extract text grouped by PDF headings, retaining chapter and page metadata."""
    pdf = pymupdf.open(pdf_path)
    try:
        pages = [page.get_text("dict", sort=True) for page in pdf]
        font_sizes = [
            span["size"]
            for page in pages
            for block in page["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
        ]

        body_size = statistics.median(font_sizes) if font_sizes else 12
        sections = []
        chapter = "Front matter"
        chapter_number = 0  # Front matter = 0
        heading = "Front matter"
        body: list[tuple[int, str]] = []
        section_start = 1

        def save_section():
            page_parts = {}
            for body_page, part in body:
                page_parts.setdefault(body_page, []).append(part)
            page_texts = [
                (body_page, "\n\n".join(parts).strip())
                for body_page, parts in sorted(page_parts.items())
            ]
            text = "\n\n".join(part for _, part in page_texts if part).strip()
            if text:
                sections.append(Section(
                    text=text,
                    book=book_name,
                    chapter=chapter,
                    chapter_number=chapter_number,
                    heading=heading,
                    page_start=section_start,
                    page_end=section_end,
                    page_texts=page_texts,
                ))

        for page_number, page in enumerate(pages, start=1):
            page_text = " ".join(
                span["text"]
                for block in page["blocks"]
                if "lines" in block
                for line in block["lines"]
                for span in line["spans"]
            )
            if "TABLE OF CONTENTS" in page_text.upper():
                in_contents = True
            if in_contents:
                if re.search(r"\bCHAPTER\s+\d+\b", page_text.upper()):
                    continue
                in_contents = False

            for block in page["blocks"]:
                if "lines" not in block:
                    continue

                heading_texts = []
                body_lines = []
                in_heading = False

                for line in block["lines"]:
                    spans = line["spans"]
                    if not spans:
                        continue

                    text = "".join(span["text"] for span in spans)
                    font_size = spans[0]["size"]

                    is_heading = (
                        font_size > body_size
                        or text.upper() == text
                        and len(text) < 100
                    )

                    if is_heading and not body:
                        in_heading = True
                        heading_texts.append(text)
                        continue

                    if in_heading:
                        if (
                            not body
                            and section_start == page_number
                            and heading != "Front matter"
                        ):
                            pass
                        save_section()
                        body.clear()
                        section_start = page_number
                        heading = text
                        chapter_match = re.match(
                            r"C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)\b.*",
                            text,
                            re.IGNORECASE,
                        )
                        if chapter_match:
                            chapter = f"Chapter {chapter_match.group(1)}"
                            chapter_number = int(chapter_match.group(1))  # Extract number
                        elif heading == "Front matter" and page_number > 1:
                            chapter = "Main text"
                            chapter_number = 1
                        in_heading = False
                        continue

                    if lines:
                        body.append((page_number, " ".join(lines)))
                        section_end = page_number

                lines = []
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"])
                    if line_text.strip():
                        lines.append(line_text)

        save_section()
        return sections
    finally:
        pdf.close()


def make_chunks(
    sections: list[Section], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[tuple[str, dict]]:
    """Split each heading section independently so chunks do not cross headings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for section in sections:
        for page_number, page_text in section.page_texts:
            documents = splitter.create_documents(
                [page_text],
                metadatas=[
                    {
                        "book": section.book,
                        "chapter": section.chapter,
                        "chapter_number": section.chapter_number,  # NEW
                        "section": section.heading,
                        "page_start": page_number,
                        "page_end": page_number,
                    }
                ],
            )
            chunks.extend(
                (document.page_content, document.metadata) for document in documents
            )
    return chunks