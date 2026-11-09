"""
Agent Builder
=============
Constructs a ReAct agent with three tools:
  1. list_books      — discover which KBs exist (always call first)
  2. semantic_search — search indexed book content with metadata filters
  3. web_search      — DuckDuckGo search for non-book / general queries
"""

from langchain.agents import create_agent

from agent.prompts import SYSTEM_PROMPT
from tools.list_books import list_books
from tools.semantic_search import semantic_search
from tools.web_search import web_search


def build_agent(llm):
    """Build ReAct agent with list_books, semantic_search, and web_search tools."""
    tools = [
        list_books,       # Step 1: discover routing_map
        semantic_search,  # Step 2A: book content (page/chapter/section/line aware)
        web_search,       # Step 2B: live web for non-book questions
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
