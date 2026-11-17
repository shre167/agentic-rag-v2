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


hows the concept of focus defined in mind managemnt

📖 Book answer

In Mind Management, 'focus' is framed not simply as allocating time but as creating and protecting the mental conditions that allow creative energy and valuable thought to occur. Kadavy contrasts time management with mind management: instead of optimizing minutes, mind management optimizes creative energy and the state of mind needed to produce good ideas [668a799d7f7a-00053][668a799d7f7a-00051]. Focus therefore involves (1) managing interruptions and the environment so that a state of focus can be achieved and preserved [668a799d7f7a-00070], and (2) aligning your mental state or mood with the work you intend to do (e.g., asking “What work am I in the mood to do right now?”) so the right tasks match your current creative energy [668a799d7f7a-00055]. The book also links this to finding your Creative Sweet Spot—the conditions and times when your mind naturally produces its best work—which is central to sustaining focus for creative tasks [668a799d7f7a-00070].

🧠 Agent Dynamic Reasoning & Strategy

⚡ Tool Execution & Latency

🔍 Semantic Search & Router (Step 1) — Semantic router selected KB(s): Retrieved 5 passages across [kb1] with MEDIUM confidence (max_score=0.84): The user explicitly asks for the definition of 'focus' in David Kadavy's Mind Management, which is contained in kb1.

📄 Retrieved 5 chunk(s) from Mind Management in 'ABOUT THE AUTHOR, FROM A TIME MANAGEMENT WORLD TO A MIND MANAGEMENT WORLD, BEYOND TIME MANAGEMENT'

⏱️ 8.64 s

📖 Book Sources & Citations

how do you not be ditracted in life and focus

📖 Book answer

Practical, book-grounded steps to be less distracted and more focused:

Pick a small number of wildly important goals and aim deep work at them. Narrowing what matters focuses attention and motivation, so you’ll invest concentrated time where it counts [8a642cf3d808-00262].

Protect your environment and block interruptions. Create a quiet, interruption-minimizing workspace or schedule blocks when you can be alone from external demands — this preserves creative energy and the state of focus [668a799d7f7a-00070][8a642cf3d808-00183].

Find and use your Creative Sweet Spot: identify when and where your mind produces its best ideas and reserve that time for generative work (e.g., a daily first-hour block) rather than shallow tasks [668a799d7f7a-00490][668a799d7f7a-00058].

Match mood to task (or vice versa). Ask: What work do I need to do? What mood will help me do it? Then use rituals, places, or small habits to recreate that mood (e.g., a specific location, a pre-work routine) so the right mental state is triggered reliably [668a799d7f7a-00211][668a799d7f7a-00058].

Separate mental states and task types. Don’t mix generation, exploration, research, and polishing in the same session — each requires a different mood and wastes creative energy if mixed prematurely [668a799d7f7a-00264].

Clear mental clutter with a trusted system. Use an input-tracking system (like GTD) so your mind isn’t holding lots of open items; that frees bandwidth for focused work [668a799d7f7a-00053].

Rewire attention: embrace boredom and schedule ‘internet blocks.’ Train yourself not to take breaks from focus by instead taking breaks from distraction (e.g., scheduled online/offline periods), which reduces the habitual urge to check stimuli and strengthens sustained concentration [8a642cf3d808-00306][8a642cf3d808-00318].

Track and celebrate deep-work time. Keep a scoreboard or tally of focused hours to make progress visible and motivate consistency; tracking shifts behavior toward more deep work [8a642cf3d808-00265].

Practical micro-habits: batch email/administrative tasks to shallow-work slots, use device do-not-disturb or full-phone-away during focus blocks, establish short pre-work rituals (cue → routine → reward) to enter the desired mood, and schedule regular recharging so creative energy replenishes [668a799d7f7a-00211][668a799d7f7a-00264][668a799d7f7a-00070].

If you want, I can turn this into a one-week plan (specific blocks, rituals, and tracking format) tailored to your typical day.

🧠 Agent Dynamic Reasoning & Strategy

⚡ Tool Execution & Latency

🔍 Semantic Search & Router (Step 1) — Semantic router selected KB(s): Retrieved 16 passages across [kb1, kb2] with HIGH confidence (max_score=0.86): The user explicitly references Mind Management concepts (Creative Sweet Spot, mood matching) and asks about avoiding distractions—a topic also covered by Deep Work—so both KBs are relevant for retrieving complementary focus strategies.

📄 Retrieved 16 chunk(s) from Deep Work, Mind Management in 'FROM A TIME MANAGEMENT WORLD TO A MIND MANAGEMENT WORLD, EPILOGUE IN MY BLOG’S income report at the beginning of 2020, I said,, FROM A TIME MANAGEMENT WORLD TO A MIND MANAGEMENT WORLD'

⏱️ 8.66 s

📖 Book Sources & Citations

whats the main motivation behind being productive

🤖 General answer

People are productive for many overlapping reasons; the main motivations typically fall into a few categories:

Purpose and meaning: Doing work that feels meaningful or aligned with your values gives a strong, sustained reason to be productive.
Progress and competence: Humans enjoy making progress and getting better. Small wins and visible improvement motivate continued effort.
Autonomy and control: Being productive creates freedom and control over your time and outcomes, which is intrinsically motivating.
Impact and contribution: The desire to help others, leave a legacy, or make a difference drives people to get things done.
Security and resources: Practical needs—money, stability, and safety—push people to be productive.
Pleasure and flow: The intrinsic reward of deep engagement or flow states makes productive work itself pleasurable.
Social factors: Expectations, recognition, and relationships (family, team, community) can motivate productivity.
Which of these matters most varies by person and situation. A useful way to increase productive behavior is to clarify which motivations matter to you, link daily tasks to those motivations, break work into small, achievable steps so progress is visible, and design your environment and energy-management practices to support sustained effort. If you’d like, I can help you identify your strongest motivations and create a short plan to align your routine with them.

🧠 Agent Dynamic Reasoning & Strategy

This is a broad, conceptual question about human motivation rather than a request for a passage from an indexed book or live web facts. Per the agent rules, no tools were called and no external citations are required; the response is drawn from general psychological and practical knowledge.

explain productivity highlighted in deep work nd mind managemnet

📖 Book answer

Summary of how each book frames productivity

Deep Work (Cal Newport)

Productivity is framed as high-quality output produced by concentrated effort: High-Quality Work Produced = (Time Spent) x (Intensity of Focus). The emphasis is on increasing intensity (ability to concentrate) during allocated deep-work time as well as increasing the time spent in that state [8a642cf3d808-00077].
Practical recommendations center on building habits and rituals to protect uninterrupted concentration, scheduling regular deep-work blocks, tracking deep-work hours as a lead measure, and eliminating or minimizing shallow distractions (e.g., social media, fragmented attention) so deep sessions yield more return per hour [8a642cf3d808-00224][8a642cf3d808-00265][8a642cf3d808-00262].
Training your attention is explicit: embrace boredom (practice not switching to distracting stimuli) and use rituals to reliably enter high-concentration states [8a642cf3d808-00305][8a642cf3d808-00191].
Mind Management (David Kadavy)

Productivity is reframed from time management to managing creative energy and mental states. The core idea is to optimize the mind (creative energy, mood, and conditions for insight) rather than merely squeezing more minutes from the day [668a799d7f7a-00053][668a799d7f7a-00064].
Practical recommendations focus on finding your Creative Sweet Spot (when and where your mind produces its best ideas), matching or shifting your mood to the type of work required, and using rituals and environments to reliably trigger the right mental state for different work types [668a799d7f7a-00210][668a799d7f7a-00058].
The book differentiates multiple mental states for creative work (not just deep vs shallow) and recommends protecting creative energy from interruptions and clearing mental clutter with trusted systems so the mind can do valuable creative thinking [668a799d7f7a-00213][668a799d7f7a-00070].
Key similarities

Both stress protecting focus from interruptions and designing routines/rituals to enter productive states (Deep Work: rituals and scheduling; Mind Management: rituals and mood cues) [8a642cf3d808-00224][668a799d7f7a-00210][668a799d7f7a-00070].
Both recommend minimizing shallow tasks or distractions and measuring/structuring time spent in focused work (Deep Work explicitly tracks deep-work hours; Mind Management advocates clearing mental clutter so creative energy can be used for the right tasks) [8a642cf3d808-00265][668a799d7f7a-00053].
Key differences

Conceptual focus: Deep Work is centered on training and protecting uninterrupted concentration as a general productivity lever (time × intensity), while Mind Management centers on optimizing creative energy and matching mental states to task types (finding a Creative Sweet Spot) [8a642cf3d808-00077][668a799d7f7a-00064].
Granularity of mental states: Mind Management outlines multiple distinct mental states for different creative tasks and recommends switching rituals accordingly; Deep Work operates with a clearer deep vs shallow distinction and emphasizes increasing the capacity for sustained deep concentration [668a799d7f7a-00213][8a642cf3d808-00305].
Prescriptions about technology/social media: Deep Work includes stronger prescriptions like quitting or drastically limiting social media and deliberately draining shallow work; Mind Management focuses more on managing energy and environment rather than blanket tech bans, recommending systems and rituals to preserve creative energy [8a642cf3d808-00305][668a799d7f7a-00070].
Practical combined takeaway (actionable steps drawing on both books)

Protect uninterrupted blocks: schedule dedicated focus sessions and create rituals to enter them (Deep Work) while ensuring those blocks align with your Creative Sweet Spot (Mind Management) [8a642cf3d808-00224][668a799d7f7a-00210].
Train attention and tolerance for boredom: practice resisting quick digital distractions to increase concentration capacity (Deep Work) [8a642cf3d808-00305].
Match mood to task: identify the mental state each task needs and use location, pre-work routines, or small cues to reliably produce that mood (Mind Management) [668a799d7f7a-00058].
Reduce shallow work: batch admin and shallow tasks into designated times and protect deep/creative slots from them (both books) [8a642cf3d808-00262][668a799d7f7a-00053].
Measure and iterate: keep a simple tally of deep/creative hours or a scoreboard to make progress visible and refine scheduling and rituals over time (Deep Work’s lead measures and Mind Management’s focus on conditions) [8a642cf3d808-00265][668a799d7f7a-00064].
If you want, I can convert these combined principles into a one-week schedule or a short checklist tailored to your typical day and types of tasks.

🧠 Agent Dynamic Reasoning & Strategy

⚡ Tool Execution & Latency

🔍 Semantic Search & Router (Step 1) — Semantic router selected KB(s): Retrieved 16 passages across [kb1, kb2] with HIGH confidence (max_score=0.85): The user asks for a comparison between how 'Mind Management' and 'Deep Work' define and recommend improving productivity, so both KBs are directly relevant.

📄 Retrieved 16 chunk(s) from Deep Work, Mind Management in 'FROM A TIME MANAGEMENT WORLD TO A MIND MANAGEMENT WORLD, ABOUT THE AUTHOR, I KNOW THE WAY by heart now: Descend the stairs from'

⏱️ 7.99 s

📖 Book Sources & Citations

whats the main motivation behind being productive

🤖 General answer

People become productive for several overlapping, common reasons. The main motivations usually include:

Purpose and meaning: Work that aligns with your values or contributes to something bigger gives sustained motivation.
Progress and mastery: Seeing improvement and gaining competence fuels continued effort (small wins matter).
Autonomy and control: Being productive creates freedom over your time and choices.
Impact and contribution: Wanting to help others or leave a legacy motivates people to act.
Security and resources: Practical needs like money, stability, or reputation push people to be productive.
Pleasure and flow: The enjoyment of deep engagement or flow makes productive work rewarding in itself.
Social factors: Expectations, recognition, and relationships (family, team) often provide incentive.
How to use this: 1) Reflect briefly: which of the above matters most to you right now? 2) Link daily tasks to that motivation (e.g., if meaning matters, state how today’s task serves your values). 3) Break work into small, visible steps so you get progress-driven motivation. 4) Design your environment and routines to reduce friction and protect focus. 5) Track small wins and celebrate them to sustain momentum.

If you want, tell me which of the motivators feels strongest for you and I’ll help build a short plan that ties your daily routine to that reason.

🧠 Agent Dynamic Reasoning & Strategy

how do i stay detached

🤖 General answer

What you probably mean by "staying detached": being emotionally non-reactive to triggers, not over-identifying with outcomes or other people’s moods, and making choices from calm rather than from impulse. Healthy detachment reduces stress and improves decision-making; unhealthy detachment is emotional numbness or avoidance. Practical steps to build healthy detachment: 1) Clarify your why — decide what detachment will help you do (reduce anxiety, improve focus, protect time). 2) Notice and name emotions — when something upsets you, pause and silently label it: “I’m feeling anger,” or “I’m anxious.” Naming reduces reactivity. 3) Use a breathing anchor — take 3–6 slow diaphragmatic breaths when you feel triggered to create space between stimulus and response. 4) Practice cognitive defusion/reappraisal — treat thoughts as passing events: “I’m having the thought that…”; ask, “Is this helpful? What’s another way to see this?” 5) Focus on the circle of control — list what you can change vs what you can’t; act on the former and let go of the latter. 6) Build boundaries and limit exposure — say no, mute notifications, schedule specific times for difficult conversations or news. 7) Schedule worry-time — if your mind spins, give it 15 minutes at a set time to process concerns, then return to the present. 8) Create rituals that reset you — short walks, quick journaling, a cup of tea, or a 5-minute breathing practice before responding to messages. 9) Keep a brief emotional-hygiene habit — daily 3-minute journaling: What happened? How did I feel? What can I do next? 10) Use Stoic pre-practice (premeditatio) — briefly imagine small losses or setbacks so they feel less overwhelming if they happen. 11) Cultivate self-compassion — detachment is not coldness; allow yourself to feel but choose your response kindly. 12) Watch for avoidance — if detachment becomes numbness, withdrawal, or prevents needed action, that’s unhealthy; consider professional support. Quick daily routine you can try: Morning: 5 minutes of breathing/setting an intention to stay calm. Midday: a 1–2 minute check-in (name current feelings). Evening: 5 minutes journaling a trigger, your response, and one small improvement for tomorrow. If you want, tell me a typical situation (work stress, relationship conflict, social media anger, etc.) and I’ll give a tailored script or short plan you can use in that moment.

🧠 Agent Dynamic Reasoning & Strategy

This is a general psychological/self-management question, so I did not call any book or web tools. I will (1) define the kind of detachment people usually mean (non-reactive, outcome-detached), (2) give practical psychological and behavioral techniques drawn from mindfulness, cognitive reappraisal, boundary-setting, and habits, (3) warn about unhealthy avoidance vs healthy detachment, and (4) offer a short daily routine and next steps you can ask for. I kept advice actionable and stepwise so you can try techniques immediately.

0
