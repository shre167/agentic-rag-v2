# Agentic PDF RAG

A Streamlit ebook assistant that indexes PDFs into separate local ChromaDB knowledge bases and uses a LangChain agent to answer with retrieved evidence and citations.

## Project Layout

```text
simple-rag/
|-- agent/
|   |-- bookagent.py
|   `-- prompts.py
|-- data1/
|   `-- Mind Management, Not Time Management.pdf
|-- data2/
|   `-- Deep Work.pdf
|-- retrieval/
|   |-- reranking.py
|   |-- retrieval.py
|   `-- vector_store.py
|-- tools/
|   |-- calculator.py
|   |-- chapter_search.py
|   |-- list_books.py
|   |-- list_book_structure.py
|   |-- page_search.py
|   |-- section_search.py
|   `-- semantic_search.py
|-- agentic_rag.py
|-- app.py
|-- chunking.py
|-- ingest.py
|-- query.py
`-- settings.py
```

`data1` is indexed as `kb1` (Mind Management), and `data2` is indexed as `kb2` (Deep Work). Each knowledge base has its own Chroma collection; chunk metadata also stores the readable book title, chapter, section, and PDF page.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GE_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` for your OpenAI-compatible chat endpoint.
4. Put each authorized PDF in its matching source folder and index it into the corresponding knowledge base:

   ```powershell
   python ingest.py "data1/Mind Management, Not Time Management.pdf" --book "Mind Management" --kb kb1 --preview
   python ingest.py "data1/Mind Management, Not Time Management.pdf" --book "Mind Management" --kb kb1
   python ingest.py "data2/Deep Work.pdf" --book "Deep Work" --kb kb2 --preview
   python ingest.py "data2/Deep Work.pdf" --book "Deep Work" --kb kb2
   python -m streamlit run app.py
   ```

Run ingestion again after changing a PDF or its chunking settings. Keep the `--book` title and `--kb` assignment consistent when re-indexing.

The source book PDF, local Chroma database, environment files, and virtual environments are intentionally excluded from Git. Add only documents you are authorized to use; place the PDF locally before indexing.

## Design

- `chunking.py` extracts heading-aware sections and splits them into page-aware chunks.
- `retrieval/vector_store.py` stores chunks in the selected Chroma knowledge base.
- `retrieval/retrieval.py` applies book/chapter/section/page filters and the no-source relevance gate.
- `retrieval/reranking.py` combines semantic distance with lexical and heading matches.
- `tools/` contains the agent's book discovery, scoped search, page search, and safe calculator tools.
- `agent/` constructs the LangChain agent and defines its operating instructions.
- `agentic_rag.py` orchestrates the agent and validates cited chunk IDs against retrieved evidence.
- `app.py` provides the Streamlit chat and visible tool activity.
- `ingest.py` indexes a PDF into a selected knowledge base; `query.py` provides a terminal chat loop.

## Useful Ebook Tools

The current tools cover source selection, general retrieval, chapter/section/page-scoped retrieval, book-structure discovery, and arithmetic. Good next additions are:

- `find_in_book(query, book=None)`: locate topics, names, or phrases and return cited page ranges; keep it as a retrieval mode rather than a duplicate search implementation.
- `get_passage_context(chunk_id, before=1, after=1)`: expand a cited chunk with adjacent chunks for context, constrained to the same book and chapter.
- `compare_books(question, books)`: orchestrate a search in each named book and return separate evidence sets before the LLM compares them.
- `chapter_overview(book, chapter)`: return the chapter's section headings and page spans; use the PDF metadata, not a generated summary, as navigation data.

Build evaluation tests before adding more tools: single-book questions, explicit book selection, chapter/page scope, cross-book comparisons, and off-topic questions that should return no sources. Calculator expressions use an allowlist of arithmetic operators; Python `eval` is not used.


<img width="1515" height="543" alt="image" src="https://github.com/user-attachments/assets/1a14182a-30e6-42f6-b386-a910feaf27be" />
📚 Autonomous Agentic Book Chat
Ask any question about indexed books, chapters, sections, pages, or lines. The agent dynamically decides which collections to search, reasons about its strategy, and synthesizes answers with citations.

so how do i improve creative thinking?

📖 Book answer

Here are practical, book-grounded steps to improve creative thinking, building on your earlier question about a “creative spot”:

Know the building blocks: treat creative thinking as moments of insight and as a process that moves through Four Stages of Creativity. Create room for each stage instead of forcing linear progress [668a799d7f7a-00064].

Find your Creative Sweet Spot and protect it daily: pick one cleared block of time (the author suggests experimenting with the First Hour Rule — many people are most divergent/insightful early) and use it for Generate work (divergent thinking) rather than execution [668a799d7f7a-00490][668a799d7f7a-00064].

Match mental state to task: learn the mental states (the book describes Seven Mental States) and deliberately choose the right state for what you’re doing — Generate for ideation, Converge/Execute later for finishing ideas [668a799d7f7a-00065].

Work with Creative Cycles: track your natural ups and downs and schedule demanding creative work during high-energy phases; respect low-energy phases for different types of tasks or rest so insight can incubate [668a799d7f7a-00266].

Use the right tools: use “grippy” tools (which resist distraction) when you need deep ideation and “slippy” tools for execution; when executing, make the slippy tool grippy only for that task if needed [668a799d7f7a-00265].

Prefer event-time over strict clock-time: don’t try to force outcomes into rigid time slices — use time blocks as guides to allocate creative energy toward goals, not to squeeze insight on demand [668a799d7f7a-00490].

Build rituals and small habits (your “personal placebo”): consistent cues and routines help you enter creative states more reliably and conserve mental energy for ideas [668a799d7f7a-00266].

A simple daily routine to try:

Block your Creative Sweet Spot (start with 30–60 minutes). Experiment with the First Hour Rule. Use a grippy tool and focus on Generate tasks.
After ideation, step away (incubation) or switch to low-effort tasks. Later, schedule a convergent session with slippy tools to refine and execute.
Track when you feel most creative across a week and align future blocks with that rhythm.
If you want, I can help you identify your Creative Sweet Spot (based on your daily schedule and energy patterns) and design a 2-week experiment to lock in a routine.

🧠 Agent Dynamic Reasoning & Strategy

⚡ Tool Execution & Latency

🔍 Semantic Search & Router (Step 1) — Mind Management explicitly focuses on creative states, mental energy, and the Four Stages of Creativity, making it the most relevant for improving creative thinking.

📥 Retrieved 5 chunk(s) from Mind Management in 'FROM A TIME MANAGEMENT WORLD TO A MIND MANAGEMENT WORLD, FROM A TIME MANAGEMENT WORLD TO A MIND MANAGEMENT WORLD, YOUR OWN PERSONAL PLACEBO'

🔎 View retrieved items (5)

⏱️ 4.81 s

📖 Book Sources & Citations

i am addicted to my phone how do i prevent these distractions and focus

📖 Book answer

Practical, book-grounded steps to beat phone addiction and protect focus — with a short 2-week experiment you can try.

Steps to start today

Schedule offline (deep) blocks and explicit Internet blocks. Decide when you’ll next allow yourself to use the phone, then don’t use it until that time — this strengthens your attention muscles and prevents constant switching [8a642cf3d808-00311][8a642cf3d808-00317].

Put the phone physically away during offline blocks. Keep it out of reach or in another room, or use airplane/Do Not Disturb so it cannot intrude (physical distance reduces the reflex to check) [8a642cf3d808-00317][8a642cf3d808-00304].

Embrace boredom. When you resist the urge to check, you rebuild tolerance for low-stimulation moments and reduce on-demand distraction habits that make deep focus impossible [8a642cf3d808-00311][8a642cf3d808-00304].

Quit or severely limit social media and other network tools that fragment attention. Either remove apps, log out, or schedule long, infrequent social-media blocks rather than leaving them always available [8a642cf3d808-00349].

Use the right tools for the mental state you want. For idea generation or deep thinking, choose “grippy” tools that resist distraction (pen & paper, single-purpose apps). If you must use a “slippy” tool (phone/computer), deliberately make it grippy for the session (app blockers, focus modes, single-app fullscreen) [668a799d7f7a-00242][668a799d7f7a-00265].

Build a simple ritual to cue focus (e.g., clear desk, put phone in drawer, 2-minute breathing, start timer). Rituals conserve willpower and make entering a focused state easier over time [668a799d7f7a-00265].

Track and iterate. Note when you felt tempted, how long you lasted, and which tactics helped. Align deep blocks with your high-energy creative times (your Creative Sweet Spot) [668a799d7f7a-00265].

A 2-week experiment (easy, repeatable)

Week 1: Start with one daily deep block of 30 minutes (preferably early). Before you begin, set the next Internet block time (e.g., 45–60 minutes after starting). Put your phone in another room or on airplane mode. Use a grippy tool (not the phone) for ideation [8a642cf3d808-00311][668a799d7f7a-00242].
Week 2: Increase to two deep blocks (30–60 min each). Add a rule: no social apps outside scheduled Internet blocks; use an app blocker or remove apps if needed. Track cravings and note improvements in focus [8a642cf3d808-00349][668a799d7f7a-00265].
If you want, I can:

design a personalized 2-week schedule based on your daily routine and energy peaks, or
suggest specific phone settings/apps (Focus modes, app lockers) and a short ritual to start each deep block.
🧠 Agent Dynamic Reasoning & Strategy

⚡ Tool Execution & Latency

🔍 Semantic Search & Router (Step 1) — Deep Work specifically addresses distraction-free concentration, digital minimalism, quitting social media, and techniques to eliminate phone-based distractions and attention residue.

📥 Retrieved 5 chunk(s) from Deep Work in 'Rule #2 Embrace Boredom, Rule #3 Quit Social Media, Rule #2 Embrace Boredom'

🔎 View retrieved items (5)

⏱️ 4.55 s

🔍 Semantic Search & Router (Step 2) — Explicit KB specified: kb1

📥 Retrieved 5 chunk(s) from Mind Management in 'YOUR OWN PERSONAL PLACEBO, EPILOGUE IN MY BLOG’S income report at the beginning of 2020, I said,, TOOLS FOR THOUGHT: THE SLIPPY & THE GRIPPY'

🔎 View retrieved items (5)

⏱️ 286 ms

📖 Book Sources & Citations

how to presvent dostractions

📖 Book answer

Short, practical plan to prevent distractions (phone and others) and protect focus — grounded in Deep Work and Mind Management:

Immediate actions (do today)

Schedule one offline deep block (30–60 min). Before it starts, set the next allowed Internet/check time and put your phone in another room or on airplane/Do Not Disturb so it cannot intrude [8a642cf3d808-00318][8a642cf3d808-00185].
Replace the phone with a grippy tool for creative work (pen & paper, a single-purpose app, or a fullscreen writing app) so the device itself isn’t a temptation [fc742ecfbd2b-00251].
If you must use a computer/phone, make it temporarily grippy (app/site blockers, single-app mode) to prevent switching [fc742ecfbd2b-00251].
Behavioral habits to build (weeks 1–4)

Embrace boredom: when you feel the urge to check, wait and sit with the boredom; this retrains your attention and reduces reflex checking over time [8a642cf3d808-00318].
Schedule explicit Internet/social-media blocks rather than constant availability; consider removing social apps or logging out if they fragment attention [8a642cf3d808-00349][8a642cf3d808-00318].
Create a short ritual to cue focus (clear desk, 2-minute breathing, start timer, put phone away). Rituals act like a “personal placebo” that makes entering deep states repeatable [fc742ecfbd2b-00267].
Two-week experiment (easy, testable)

Week 1: One daily deep block (30 min, preferably early). Set that block as phone-free; use a grippy tool. Track how often you resisted checking and what helped [8a642cf3d808-00318][fc742ecfbd2b-00251].
Week 2: Increase to two daily blocks (30–60 min). Add a rule: no social apps outside scheduled Internet blocks and use an app/site blocker during deep blocks [8a642cf3d808-00349][fc742ecfbd2b-00251].
Environmental and tool tweaks

Physically increase distance from your phone (another room, drawer) — physical separation reduces reflex checking and the device’s power to hijack attention [8a642cf3d808-00318][8a642cf3d808-00185].
Use app/site restriction tools (or built-in Focus modes) to limit access during deep blocks; many apps allow restrictions for specific times [fc742ecfbd2b-00251].
How to iterate

Track cravings, success rate, and your energy levels each day. Ask: What worked? When were cravings strongest? Adjust block length, timing, and rituals based on your energy pattern [fc742ecfbd2b-00057].
Why this works (brief theory)

Modern network tools fragment attention and create a powerful urge to switch; scheduling offline blocks, removing easy access to those tools, and practicing boredom retrains attention autonomy and makes deep focus possible [8a642cf3d808-00185][8a642cf3d808-00349][8a642cf3d808-00318].
Matching tools, environment, and ritual to the mental state you want (grippy for ideation, slippy for execution) preserves creative energy and reduces switching costs [fc742ecfbd2b-00251][fc742ecfbd2b-00057].
If you want, I can now:

design a personalized 2-week schedule around your daily routine and energy peaks, or
suggest specific Focus-mode settings and a 60-second ritual you can use to start each deep block.
🧠 Agent Dynamic Reasoning & Strategy

⚡ Tool Execution & Latency

🔍 Semantic Search & Router (Step 1) — Explicit KB specified: kb2

📥 Retrieved 5 chunk(s) from Deep Work in 'Rule #1 Work Deeply, Chapter Two Deep Work Is Rare, Rule #3 Quit Social Media'

🔎 View retrieved items (5)

⏱️ 358 ms

🔍 Semantic Search & Router (Step 2) — Explicit KB specified: kb4

📥 Retrieved 5 chunk(s) from Mind Management, Not Time Management in 'YOUR OWN PERSONAL PLACEBO, BEYOND TIME MANAGEMENT, TOOLS FOR THOUGHT: THE SLIPPY & THE GRIPPY'

🔎 View retrieved items (5)

⏱️ 358 ms

📖 Book Sources & Citations

what was my previous query

🤖 General answer

Your previous query was: "how to presvent dostractions" — (likely you meant "how to prevent distractions").

🧠 Agent Dynamic Reasoning & Strategy

Ask a question across indexed books or any external/web topic...


whats on page 60 of mind managemnt time management

📖 Book answer

Page 60 is in Chapter 2 (section: “THE HIGH INTEREST RATE OF BORROWED TIME”). It summarizes research showing that being busy reduces creativity (about a 19% drop normally and up to a 45% drop when extremely busy), and compares that loss to paying interest on borrowed time—effects that compound and carry over to subsequent days [fc742ecfbd2b-00144][fc742ecfbd2b-00145]. The page also describes the author’s shift from clock-time to event-time for creative work, noting that event-time is often better for creativity while clock-time is appropriate for punctual tasks like catching a flight [fc742ecfbd2b-00146].

🧠 Agent Dynamic Reasoning & Strategy

I searched the Mind Management, Not Time Management book (kb4) for “page 60” using semantic_search with kb_name set to kb4. The tool returned three passages from page 60 in Chapter 2 (IDs fc742ecfbd2b-00144, fc742ecfbd2b-00145, fc742ecfbd2b-00146). I synthesized those passages: the first two contain the study results and the ‘interest on borrowed time’ metaphor (creativity drops of ~19% and up to 45%, compounding effects), and the third passage discusses the shift from clock-time to event-time and when each is appropriate. I cited the specific chunk IDs that contained each point.

⚡ Tool Execution & Latency

🔍 Semantic Search & Router (Step 1) — Explicit KB specified: kb4

📥 Retrieved 3 chunk(s) from Mind Management, Not Time Management in 'THE HIGH INTEREST RATE OF BORROWED TIME, THE HIGH INTEREST RATE OF BORROWED TIME, THE HIGH INTEREST RATE OF BORROWED TIME'

🔎 View retrieved items (3)

⏱️ 15 ms

📖 Book Sources & Citations
