# Quick Start - Minimal Book Agentic RAG

## ✓ 2 Tools Only - Metadata-Aware

- ✓ **list_books** - Discover books
- ✓ **semantic_search** - Search everything (pages, chapters, sections, concepts)
- ✓ **Structure-aware chunking** - Metadata handles pages/chapters/sections automatically
- ✓ **Multi-source** - Agent searches kb1, kb2, or both dynamically

## Start (2 Commands)

### 1. Start Ollama
```powershell
ollama run qwen2.5:3b
```

### 2. Start Streamlit
```powershell
cd "c:\Users\shrey\Downloads\New folder (3)\agentic-rag-v2"
.\.venv\Scripts\streamlit.exe run app.py
```

## Test Questions

```
"What's on page 66 of Deep Work?"
→ semantic_search finds chunks with page=66 metadata

"What does Chapter 3 say?"
→ semantic_search finds chunks with chapter=3 metadata

"What does Deep Work say about focus?"
→ semantic_search finds focus-related chunks

"Compare both books on productivity"
→ semantic_search on kb1 + kb2
```

## Why 2 Tools?

Your structure-aware chunking has page/chapter/section metadata.  
**semantic_search retrieves it all automatically.**  
No need for separate page_search, chapter_search, etc.

---

**Simple, fast, metadata-aware!** 🚀
