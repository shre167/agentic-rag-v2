import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Section:
    text: str
    chapter: str
    heading: str
    page_start: int
    page_end: int


def extract_sections(pdf_path: Path) -> list[Section]:
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
            if span["text"].strip()
        ]
        body_size = statistics.median(font_sizes) if font_sizes else 12
        sections = []
        chapter = "Front matter"
        heading = "Front matter"
        body = []
        section_start = 1
        section_end = 1
        in_contents = False

        def save_section():
            text = "\n\n".join(part for part in body if part).strip()
            if text:
                sections.append(
                    Section(text, chapter, heading, section_start, section_end)
                )

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
                lines = []
                for line in block["lines"]:
                    spans = [span for span in line["spans"] if span["text"].strip()]
                    text = " ".join(span["text"].strip() for span in spans).strip()
                    if not text:
                        continue

                    largest_font = max(span["size"] for span in spans)
                    letters = re.sub(r"[^A-Za-z]", "", text)
                    uppercase_heading = (
                        len(text) <= 100
                        and len(letters) >= 3
                        and letters == letters.upper()
                    )
                    is_heading = largest_font >= body_size * 1.2 or uppercase_heading

                    if is_heading:
                        if (
                            not body
                            and section_start == page_number
                            and heading != "Front matter"
                        ):
                            heading = f"{heading} {text}".strip()
                            continue
                        save_section()
                        body.clear()
                        section_start = page_number
                        section_end = page_number
                        heading = text
                        chapter_match = re.match(
                            r"C\s*H\s*A\s*P\s*T\s*E\s*R\s+(\d+)\b.*",
                            text,
                            re.IGNORECASE,
                        )
                        if chapter_match:
                            chapter = f"Chapter {chapter_match.group(1)}"
                        elif heading == "Front matter" and page_number > 1:
                            chapter = "Main text"
                        continue

                    lines.append(text)

                if lines:
                    body.append(" ".join(lines))
                    section_end = page_number

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
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for section in sections:
        documents = splitter.create_documents(
            [section.text],
            metadatas=[
                {
                    "chapter": section.chapter,
                    "section": section.heading,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                }
            ],
        )
        chunks.extend((document.page_content, document.metadata) for document in documents)
    return chunks
