from langchain.agents import create_agent

from .prompts import SYSTEM_PROMPT

from tools.semantic_search import semantic_search
from tools.chapter_search import chapter_search
from tools.section_search import section_search
from tools.page_search import page_search
from tools.list_book_structure import list_book_structure
from tools.list_books import list_books
from tools.calculator import calculator


def build_agent(llm):
    return create_agent(
        model=llm,
        tools=[
            semantic_search,
            chapter_search,
            section_search,
            page_search,
            list_books,
            list_book_structure,
            calculator,
        ],
        system_prompt=SYSTEM_PROMPT,
    )