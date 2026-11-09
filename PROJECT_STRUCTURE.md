# Agentic RAG - Project Structure & Pipeline

## Overview
This is a structured Agentic RAG system that dynamically retrieves information from multiple PDF books using an LLM agent.

## Architecture

```
User Query
    ↓
[app.py] - UI Layer (Streamlit)
    ↓
[core/rag_engine.py] - RAG Orchestrator
    ├─ Initialize LLM
    ├─ Build Agent
    └─ Invoke Agent Loop
    ↓
[agent/agent_builder.py] - Agent Factory
    ├─ Load Tools
    └─ Create ReAct Agent
    ↓
[agent/tools/] - Tool Definitions
    ├─ list_books (discover KBs)
    ├─ semantic_search (retrieve chunks)
    └─ page_search (page-specific retrieval)
    ↓
[retrieval/retrieval_engine.py] - Retrieval Pipeline
    ├─ Parse query + extract metadata
    ├─ Build where clause (page/chapter/section filters)
    └─ Query ChromaDB + Rerank Results
    ↓
[chroma_db/] - Vector Database
    ├─ kb1 collection (Mind Management)
    └─ kb2 collection (Deep Work)
    ↓
Answer with Citations
```

## File Organization

### Root Level
- **app.py** - Streamlit UI entry point
- **settings.py** - Configuration loading (paths, constants)
- **requirements.txt** - Dependencies
- **.env** - Environment variables (LLM keys, API endpoints)

### /config
- **llm_config.py** - LLM initialization (Ollama, OpenAI, etc.)
- **constants.py** - App-wide constants

### /core
- **rag_engine.py** - Main RAG orchestrator that ties everything together
- **types.py** - Type definitions for tool outputs

### /agent
- **agent_builder.py** - Creates ReAct agent with tools
- **tools/** - Tool implementations
  - **list_books.py** - Lists available knowledge bases
  - **semantic_search.py** - Semantic search with optional filters
  - **page_search.py** - Page-specific search (auto-extracts page numbers)

### /retrieval
- **retrieval_engine.py** - Core retrieval logic with filtering
- **vector_store.py** - ChromaDB wrapper
- **reranking.py** - Result reranking

### /ui
- **components.py** - Streamlit UI components
- **display.py** - Result formatting and display

## Function Call Flow

### When user submits query:

1. **Streamlit UI** (`app.py`)
   - User enters: "What's on page 50 of Deep Work?"
   - Calls: `answer_with_agent(question)`

2. **RAG Engine** (`core/rag_engine.py`)
   - Initializes LLM with config
   - Builds agent with tools
   - Invokes: `agent.invoke({"messages": [...]})`
   - Returns: `(answer, sources, activity)`

3. **Agent Execution Loop** (LangChain ReAct)
   - Receives: user query
   - Decides: which tool to use
   - Calls: `list_books()` → `page_search()`
   - Gets: retrieved chunks
   - Generates: answer with citations

4. **Tool: page_search** (`agent/tools/page_search.py`)
   - Input: `query="What's on page 50"`, `kb_name="kb2"`
   - Extracts: page number (50) using regex
   - Calls: `retrieve_chunks(page=50, kb_name="kb2")`
   - Returns: formatted tool result

5. **Retrieval Engine** (`retrieval/retrieval_engine.py`)
   - Input: `page=50, kb_name="kb2"`
   - Builds: where clause `{"page_start": {"$lte": 50}, "page_end": {"$gte": 50}}`
   - Queries: ChromaDB with filters
   - Reranks: results by relevance
   - Returns: list of chunks with metadata

6. **Vector Store** (`retrieval/vector_store.py`)
   - Gets: collection "kb2"
   - Executes: ChromaDB query with where clause
   - Returns: matching chunk IDs, texts, metadata

7. **Agent Formats Response**
   - Extracts answer text
   - Validates citations
   - Returns: structured response

8. **UI Displays Results** (`app.py`)
   - Shows: answer
   - Shows: sources with page/chapter/section metadata
   - Shows: agent reasoning (optional)

## Tool Definitions

### Tool 1: list_books
```
Purpose: Discover available knowledge bases
Input: None
Output: List of available books with their descriptions
Why: Agent needs to know which KBs exist before searching
Dynamic: Yes - loaded from ChromaDB collections
```

### Tool 2: semantic_search
```
Purpose: Semantic search with optional metadata filtering
Input: query (text), kb_name (kb1|kb2), chapter (optional), section (optional)
Output: Relevant chunks with metadata
Why: General search for concepts, with optional structure filtering
Dynamic: Yes - supports any query + optional filters
```

### Tool 3: page_search
```
Purpose: Page-specific search (auto-extracts page numbers)
Input: query (e.g., "page 50"), kb_name (kb1|kb2)
Output: Chunks from specific page only
Why: Quick page lookups without requiring agent to parse numbers
Dynamic: Yes - extracts page numbers from query text
```

## No Hardcoding Principles

- ✅ Books discovered dynamically from ChromaDB
- ✅ Page/chapter/section extracted by tools, not hardcoded
- ✅ Tool choices made by LLM, not scripted
- ✅ No if/else logic for KB selection
- ✅ LLM decides which KB to search, which tool to use

## LLM Decision Points

The LLM makes these decisions dynamically:

1. **Which tool to use?**
   - Sees: user query, available tools
   - Decides: list_books → page_search vs semantic_search vs web_search

2. **Which KB to search?**
   - After list_books: knows available KBs
   - Decides: kb1, kb2, or both

3. **How to filter search?**
   - Extracts: page number, chapter, section from query
   - Passes: as parameters to semantic_search

4. **When to stop searching?**
   - Evaluates: retrieved results
   - Decides: sufficient info or search more

## Key Files to Understand

1. **app.py** - Read first (UI layer, entry point)
2. **core/rag_engine.py** - Main orchestrator
3. **agent/agent_builder.py** - Agent creation
4. **agent/tools/** - What the agent can do
5. **retrieval/retrieval_engine.py** - How retrieval works
