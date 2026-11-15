"""PDF text extraction with structure-aware chunking and line-level metadata."""

import re
import statistics
from pathlib import Path
from dataclasses import dataclass, field

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Section:
    text: str
    book: str
    chapter: str
    chapter_number: int  # Numeric chapter for easy filtering
    heading: str
    page_start: int
    page_end: int
    page_texts: list[tuple[int, str]] = field(default_factory=list)
    page_data: list[tuple[int, list[tuple[int, str]]]] = field(default_factory=list)


def extract_sections(pdf_path: Path, book_name: str) -> list[Section]:
    """Extract text grouped by PDF headings, retaining chapter, section, page, and line metadata."""
    pdf = pymupdf.open(pdf_path)
    try:
        pages = [page.get_text("dict", sort=True) for page in pdf]
        font_sizes = [
            span["size"]
            for page in pages
            for block in page.get("blocks", [])
            if "lines" in block
            for line in block["lines"]
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]

        body_size = statistics.median(font_sizes) if font_sizes else 12
        sections: list[Section] = []
        chapter = "Front matter"
        chapter_number = 0  # Front matter = 0
        heading = "Front matter"
        body: list[tuple[int, int, str]] = []  # (page_number, line_number, text)
        section_start = 1
        section_end = 1
        in_contents = False

        def save_section():
            nonlocal section_end
            page_dict: dict[int, list[tuple[int, str]]] = {}
            for body_page, line_num, line_text in body:
                page_dict.setdefault(body_page, []).append((line_num, line_text))

            page_data = sorted(page_dict.items())
            page_texts = [
                (body_page, "\n".join(t for _, t in lines).strip())
                for body_page, lines in page_data
            ]
            text = "\n\n".join(part for _, part in page_texts if part).strip()
            if text:
                end_page = max((p for p, _, _ in body), default=section_end)
                sections.append(
                    Section(
                        text=text,
                        book=book_name,
                        chapter=chapter,
                        chapter_number=chapter_number,
                        heading=heading,
                        page_start=section_start,
                        page_end=end_page,
                        page_texts=page_texts,
                        page_data=page_data,
                    )
                )

        for page_number, page in enumerate(pages, start=1):
            page_text = " ".join(
                span["text"]
                for block in page.get("blocks", [])
                if "lines" in block
                for line in block["lines"]
                for span in line.get("spans", [])
            )
            if "TABLE OF CONTENTS" in page_text.upper():
                in_contents = True
            if in_contents:
                if re.search(r"\bCHAPTER\s+\d+\b", page_text.upper()):
                    continue
                in_contents = False

            page_line_num = 0
            for block in page.get("blocks", []):
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    page_line_num += 1
                    spans = [
                        span
                        for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    ]
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
                    chap_keyword_match = bool(
                        re.match(
                            r"^(?:C\s*H\s*A\s*P\s*T\s*E\s*R|Rule|Part|Section|Book|Principle|Step)\s+(?:#\s*)?(\w+)\b",
                            text,
                            re.IGNORECASE,
                        )
                        or text.strip().lower() in {
                            "introduction", "preface", "prologue", "foreword",
                            "conclusion", "epilogue", "afterword", "appendix"
                        }
                    )
                    is_heading = largest_font >= body_size * 1.12 or uppercase_heading or chap_keyword_match

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

                        WORD_NUMS = {
                            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
                            "nineteen": 19, "twenty": 20,
                            "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
                            "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
                            "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15,
                        }
                        chap_m = re.match(
                            r"(?:C\s*H\s*A\s*P\s*T\s*E\s*R|Rule|Part|Section|Book|Principle|Step)\s+(?:#\s*)?(\w+)\b",
                            text,
                            re.IGNORECASE,
                        )
                        if chap_m:
                            val = chap_m.group(1).lower()
                            num = WORD_NUMS.get(val, int(val) if val.isdigit() else 1)
                            raw_match = chap_m.group(0).lower()
                            if "rule" in raw_match:
                                prefix = "Rule"
                            elif "part" in raw_match:
                                prefix = "Part"
                            elif "principle" in raw_match:
                                prefix = "Principle"
                            elif "step" in raw_match:
                                prefix = "Step"
                            else:
                                prefix = "Chapter"
                            chapter = f"{prefix} {num}"
                            chapter_number = num
                        elif text.strip().lower() in {"introduction", "preface", "prologue", "foreword"}:
                            chapter = text.strip().title()
                            chapter_number = 0
                        elif text.strip().lower() in {"conclusion", "epilogue", "afterword", "appendix"}:
                            chapter = text.strip().title()
                            chapter_number = 999
                        elif heading == "Front matter" and page_number > 1:
                            chapter = "Main text"
                            chapter_number = 1
                        continue

                    body.append((page_number, page_line_num, text))
                    section_end = page_number

        save_section()
        return sections
    finally:
        pdf.close()


def make_chunks(
    sections: list[Section], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[tuple[str, dict]]:
    """Split each section into chunks, tracking exact page and line number boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[tuple[str, dict]] = []
    for section in sections:
        # Use line-level page_data if available
        if getattr(section, "page_data", None):
            for page_number, lines in section.page_data:
                if not lines:
                    continue

                char_offset = 0
                line_map: list[tuple[int, int, int]] = []
                text_parts: list[str] = []

                for line_num, line_text in lines:
                    if not line_text.strip():
                        continue
                    start_char = char_offset
                    end_char = start_char + len(line_text)
                    line_map.append((start_char, end_char, line_num))
                    text_parts.append(line_text)
                    char_offset = end_char + 1  # newline separator

                full_text = "\n".join(text_parts)
                if not full_text.strip() or not line_map:
                    continue

                documents = splitter.create_documents(
                    [full_text],
                    metadatas=[{"book": section.book}],
                )

                for doc in documents:
                    start_idx = doc.metadata.get("start_index", 0)
                    end_idx = start_idx + len(doc.page_content)

                    # Determine start and end line for this chunk
                    covering_lines = [
                        line_num
                        for s_char, e_char, line_num in line_map
                        if e_char > start_idx and s_char < end_idx
                    ]
                    line_start = covering_lines[0] if covering_lines else line_map[0][2]
                    line_end = covering_lines[-1] if covering_lines else line_map[-1][2]

                    metadata = {
                        "book": section.book,
                        "chapter": section.chapter,
                        "chapter_number": section.chapter_number,
                        "section": section.heading,
                        "page_start": page_number,
                        "page_end": page_number,
                        "line_start": line_start,
                        "line_end": line_end,
                    }
                    chunks.append((doc.page_content, metadata))
        else:
            # Fallback for sections with only page_texts
            for page_number, page_text in section.page_texts:
                if not page_text.strip():
                    continue
                documents = splitter.create_documents(
                    [page_text],
                    metadatas=[
                        {
                            "book": section.book,
                            "chapter": section.chapter,
                            "chapter_number": section.chapter_number,
                            "section": section.heading,
                            "page_start": page_number,
                            "page_end": page_number,
                            "line_start": 1,
                            "line_end": max(1, page_text.count("\n") + 1),
                        }
                    ],
                )
                chunks.extend(
                    (doc.page_content, doc.metadata) for doc in documents
                )

    return chunks
