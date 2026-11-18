from pathlib import Path
from typing import List, Dict

import streamlit as st

from agentic_rag import answer_with_agent
from ingest import ingest_pdf, register_existing_books
from retrieval.vector_store import get_collection, list_knowledge_bases
from retrieval.kb_registry import get_all_book_profiles, get_kb_registry
from settings import ROOT
from chat_storage import (
    load_chat_history,
    save_chat_history,
    clear_chat_history,
    get_all_sessions,
    get_cross_session_summary,
)


st.set_page_config(
    page_title="Agentic Book Chat",
    page_icon="📚",
    layout="wide",
)

# Answer type display config
_ANSWER_TYPE_META = {
    "book":    {"icon": "📖", "label": "Book answer",    "color": "green"},
    "web":     {"icon": "🌐", "label": "Web answer",     "color": "blue"},
    "mixed":   {"icon": "🔀", "label": "Mixed answer",   "color": "orange"},
    "general": {"icon": "🤖", "label": "General answer", "color": "gray"},
}

# Tool display config
_TOOL_META = {
    "semantic_search": {"icon": "🔍", "label": "Semantic Search & Router"},
    "web_search":      {"icon": "🌐", "label": "Web Search"},
}


def render_agent_activity(activity: List[Dict]):
    """Render tool execution trace with tool name, latency, routing decision, and found summary."""
    tool_steps = [s for s in activity if s.get("tool")]
    if not tool_steps:
        return

    st.caption("⚡ **Tool Execution & Latency**")
    for step_number, step in enumerate(tool_steps, start=1):
        tool_name = step.get("tool")
        meta = _TOOL_META.get(tool_name, {"icon": "🔧", "label": tool_name or "Tool"})

        latency = step.get("latency_ms")
        if latency is not None:
            latency_str = f"{latency:.0f} ms" if latency < 1000 else f"{latency/1000:.2f} s"
        else:
            latency_str = "< 10 ms"

        found_summary = step.get("found_summary", "")
        tool_reasoning = step.get("tool_reasoning", "")
        is_no_match = "NO_MATCH" in found_summary or "NO_MATCH" in str(step.get("result", ""))

        reason = tool_reasoning or found_summary or "Executed for query context"

        col1, col2 = st.columns([5, 1])
        with col1:
            if is_no_match:
                st.markdown(
                    f"**{meta['icon']} {meta['label']}** (Step {step_number}) — "
                    f"⚠️ *No matching book profile found — routing to web fallback.*"
                )
            else:
                st.markdown(f"**{meta['icon']} {meta['label']}** (Step {step_number}) — {reason}")
            
            if found_summary and not is_no_match and found_summary != tool_reasoning:
                st.caption(f"📥 {found_summary}")
            
            what_found = step.get("what_found_items", [])
            if what_found and not is_no_match:
                with st.expander(f"🔎 View retrieved items ({len(what_found)})", expanded=False):
                    for itm in what_found:
                        st.markdown(f"- {itm}")
        with col2:
            st.caption(f"⏱️ `{latency_str}`")


# --------------------------------------------------
# Ingestion Dialog
# --------------------------------------------------

@st.dialog("📥 Ingest New Book")
def show_ingest_dialog():
    st.caption("Upload PDF book(s). Each book is placed in its own isolated knowledge base with an automatic LLM content profile.")
    dialog_files = st.file_uploader("Select PDF Book", type=["pdf"], accept_multiple_files=True, key="dialog_book_uploader")
    dialog_custom_name = ""
    if dialog_files and len(dialog_files) == 1:
        dialog_custom_name = st.text_input(
            "Book Title (optional)",
            value=Path(dialog_files[0].name).stem.replace("_", " "),
            help="Custom title stored in metadata (defaults to file name).",
        )
    
    if dialog_files:
        if st.button("🚀 Ingest & Index", type="primary", use_container_width=True):
            upload_dir = ROOT / "uploads"
            upload_dir.mkdir(exist_ok=True)
            with st.status("Ingesting book into isolated collection...", expanded=True) as status:
                for idx, uf in enumerate(dialog_files, start=1):
                    status.write(f"📖 Processing ({idx}/{len(dialog_files)}): **{uf.name}**...")
                    temp_path = upload_dir / uf.name
                    with open(temp_path, "wb") as f:
                        f.write(uf.getbuffer())
                    b_name = dialog_custom_name if (len(dialog_files) == 1 and dialog_custom_name) else None
                    try:
                        summary = ingest_pdf(pdf_path=temp_path, book_name=b_name)
                        p_msg = "📋 Profile generated" if summary.get("profile_generated") else "⚠️ Profile pending"
                        status.write(
                            f"✅ **{summary['book_name']}** indexed into **`{summary['kb_name']}`** "
                            f"({summary['indexed_count']} chunks across {summary['sections_count']} sections). {p_msg}"
                        )
                    except Exception as err:
                        status.write(f"❌ Failed to ingest {uf.name}: {err}")
                status.update(label="🎉 Ingestion Complete!", state="complete", expanded=False)
            st.success("Book successfully ingested and available for routing!")
            st.rerun()


# --------------------------------------------------
# Sidebar: Dynamic Knowledge Bases & Profiles
# --------------------------------------------------

knowledge_bases = list_knowledge_bases()

with st.sidebar:
    st.title("📚 Multi-Source Knowledge Bases")

    st.success(
        "✅ The agent autonomously decides which books to search. "
        "No manual selection needed — it reasons dynamically from your question."
    )

    if st.button("➕ Ingest New Book", use_container_width=True, key="sidebar_ingest_btn"):
        show_ingest_dialog()

    if knowledge_bases:
        kb_info = {}
        for kb in knowledge_bases:
            col = get_collection(kb)
            count = col.count()
            sample = col.get(limit=1, include=["metadatas"])["metadatas"]
            title = sample[0].get("book", "Unknown") if sample else "Unknown"
            kb_info[kb] = (title, count)

        total_chunks = sum(c for _, c in kb_info.values())
        st.metric("Total Indexed Chunks", total_chunks)

        st.markdown("### Indexed Books")
        for kb, (title, count) in kb_info.items():
            st.markdown(f"- **`{kb}`** → *{title}* ({count} chunks)")

        # Show book profiles
        profiles = get_all_book_profiles()
        if profiles:
            st.markdown("### 📋 Book Profiles (LLM-Generated)")
            for p in profiles:
                with st.expander(f"📖 {p.get('kb_name', p['kb_id'])}"):
                    if p.get("summary"):
                        st.markdown(f"**Summary:** {p['summary']}")
                    if p.get("main_topics"):
                        st.markdown(f"**Topics:** {', '.join(p['main_topics'][:8])}")
                    if p.get("key_concepts"):
                        st.markdown(f"**Key Concepts:** {', '.join(p['key_concepts'][:8])}")
                    if p.get("coverage"):
                        st.markdown(f"**Coverage:** {p['coverage']}")

        # Button to register existing books that don't have profiles
        registry = get_kb_registry()
        unregistered = [kb for kb in knowledge_bases if kb not in registry]
        if unregistered:
            if st.button("📋 Generate Missing Profiles", use_container_width=True):
                with st.spinner("Generating profiles for unregistered books..."):
                    results = register_existing_books()
                    for r in results:
                        if r["status"] == "profile_generated":
                            st.success(f"✅ {r['book_name']} profiled")
                        elif r["status"] != "already_profiled":
                            st.warning(f"⚠️ {r['book_name']}: {r['status']}")
                st.rerun()

        st.divider()

        st.markdown("### 🔍 Chunk Inspector *(read-only)*")
        st.caption("Browse raw chunks stored in a collection.")

        viewer_kb = st.selectbox(
            "Browse collection:",
            options=knowledge_bases,
            index=0,
            key="viewer_kb",
        )

        with st.expander("Show chunks"):
            viewer_col = get_collection(viewer_kb)
            stored = viewer_col.get(include=["documents", "metadatas"])

            if stored["ids"]:
                rows = sorted(
                    zip(
                        stored["ids"],
                        stored["documents"],
                        stored["metadatas"],
                    ),
                    key=lambda row: row[0],
                )

                books = sorted({m.get("book", "Unknown") for _, _, m in rows})
                selected_book = st.selectbox("Book Filter", ["All Books", *books], key="chunk_book")

                if selected_book != "All Books":
                    rows = [r for r in rows if r[2].get("book") == selected_book]

                sections = sorted({m.get("section", "Unknown") for _, _, m in rows})
                selected_section = st.selectbox("Section Filter", ["All Sections", *sections], key="chunk_section")

                if selected_section != "All Sections":
                    rows = [r for r in rows if r[2].get("section") == selected_section]

                page_size = 5
                page_count = max(1, (len(rows) + page_size - 1) // page_size)
                page_number = st.number_input("Page", min_value=1, max_value=page_count, value=1)
                start = (page_number - 1) * page_size

                for chunk_id, text, metadata in rows[start : start + page_size]:
                    lines_str = (
                        f" | ll. {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
                        if "line_start" in metadata
                        else ""
                    )
                    title = f"{chunk_id} | pp. {metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}{lines_str}"

                    with st.expander(title):
                        st.markdown(f"**{metadata.get('section', 'Unknown')}**")
                        lines_caption = (
                            f" | Lines: {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
                            if "line_start" in metadata
                            else ""
                        )
                        st.caption(f"Chapter: {metadata.get('chapter', 'Unknown')}{lines_caption}")
                        st.write(text)
            else:
                st.info("No chunks available in this collection.")


# --------------------------------------------------
# Main Chat UI
# --------------------------------------------------

st.title("📚 Autonomous Agentic Book Chat")
st.caption(
    "Ask any question about indexed books, chapters, sections, pages, or lines. "
    "The agent dynamically decides which collections to search, reasons about its strategy, and synthesizes answers with citations."
)

# Session management in sidebar
with st.sidebar:
    st.divider()
    st.markdown("### 💬 Chat Sessions")
    
    all_sessions = get_all_sessions()
    if "current_session" not in st.session_state:
        st.session_state.current_session = all_sessions[0] if all_sessions else "default"

    # Ensure current_session is in all_sessions
    if st.session_state.current_session not in all_sessions:
        all_sessions.append(st.session_state.current_session)
        all_sessions = sorted(list(set(all_sessions)))

    current_idx = all_sessions.index(st.session_state.current_session) if st.session_state.current_session in all_sessions else 0

    def on_session_change():
        selected = st.session_state.session_selector
        st.session_state.current_session = selected
        st.session_state.messages = load_chat_history(selected)

    selected_session = st.selectbox(
        "Select session",
        options=all_sessions,
        index=current_idx,
        key="session_selector",
        on_change=on_session_change,
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history(st.session_state.current_session)
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("➕ New Session", use_container_width=True):
            st.session_state.show_new_session_input = True
    
    # Show input for new session name
    if st.session_state.get("show_new_session_input"):
        new_session_name = st.text_input(
            "Enter session name",
            value="",
            placeholder="e.g., Deep Work Notes",
            key="new_session_name_input",
        )
        if new_session_name:
            clean_session_name = new_session_name.strip().replace(" ", "_")
            if st.button("✅ Create Session", use_container_width=True):
                # Save empty chat to create the session file immediately
                save_chat_history([], clean_session_name)
                st.session_state.current_session = clean_session_name
                st.session_state.messages = []
                st.session_state.show_new_session_input = False
                st.rerun()
        
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_new_session_input = False
            st.rerun()

# Load chat history for the active session
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(st.session_state.current_session)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Answer type badge for replayed assistant messages
        if message["role"] == "assistant" and message.get("answer_type"):
            at_meta = _ANSWER_TYPE_META.get(message["answer_type"], _ANSWER_TYPE_META["book"])
            st.caption(f"{at_meta['icon']} **{at_meta['label']}**")

        st.markdown(message["content"])

        if message.get("reasoning"):
            with st.expander("🧠 Agent Dynamic Reasoning & Strategy", expanded=False):
                st.markdown(message["reasoning"])

        if message.get("activity"):
            render_agent_activity(message["activity"])

        if message.get("sources"):
            book_sources = [s for s in message["sources"] if s.get("source") != "web"]
            web_sources  = [s for s in message["sources"] if s.get("source") == "web"]

            if book_sources:
                with st.expander("📖 Book Sources & Citations"):
                    for source in book_sources:
                        lines_info = (
                            f", lines {source['line_start']}-{source['line_end']}"
                            if "line_start" in source
                            else ""
                        )
                        kb_info = f" [{source.get('kb_name', 'kb')}]" if source.get('kb_name') else ""
                        st.markdown(
                            f"**[{source['id']}] {source.get('book', 'Book')}{kb_info} — {source['section']}** "
                            f"({source.get('chapter','')}, "
                            f"pp. {source['page_start']}-{source['page_end']}{lines_info})"
                        )
                        st.write(source["text"])

            if web_sources:
                with st.expander("🌐 Web Sources"):
                    for source in web_sources:
                        st.markdown(
                            f"**[{source['id']}] [{source.get('title', source['id'])}]({source.get('url', '')})**"
                        )
                        st.write(source.get("text", ""))


if not knowledge_bases:
    st.info("👋 No books have been indexed yet. Use the **➕ Ingest PDF Book** button below to get started!")

if not st.session_state.messages and knowledge_bases:
    st.info(
        "💡 **Ask any question to get started. Examples:**\n"
        "- 📖 **Multi-Book Comparison**: *'Compare focus and distraction in Mind Management and Deep Work'* *(agent queries both collections & explains why)*\n"
        "- 🎯 **Line & Page Lookup**: *'What does line 12 of page 10 say in Mind Management?'* *(precise metadata retrieval)*\n"
        "- 📚 **Chapter Deep Dive**: *'What are the core arguments in Deep Work chapter 1?'* *(chapter-filtered retrieval)*\n"
        "- 🌐 **External / Web Query**: *'Who is the CEO of Google and what are recent AI developments?'* *(agent detects topic is outside books & switches to web search)*"
    )

# Bottom chat bar layout: [➕ Ingest] button on the left, [Chat Input] on the right
upload_col, input_col = st.columns([1, 11])

with upload_col:
    if st.button("➕", help="Ingest a new PDF book into its own collection", use_container_width=True, key="bottom_ingest_btn"):
        show_ingest_dialog()

with input_col:
    question = st.chat_input("Ask a question across indexed books or any external/web topic...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages[:-1]
    ]

    try:
        with st.chat_message("assistant"):
            with st.spinner("Agent analyzing query, dynamically choosing collections, and retrieving chunks..."):
                cross_context = get_cross_session_summary(st.session_state.current_session)
                result = answer_with_agent(
                    question,
                    history=history,
                    kb_name=None,  # No hardcoding - agent dynamically decides
                    cross_session_context=cross_context,
                )

            answer = result[0]
            sources = result[1]
            activity = result[2]
            reasoning = getattr(result, "reasoning", None)
            answer_type = getattr(result, "answer_type", "book")

            # Fallback: get reasoning from activity if not on result attribute
            if not reasoning:
                for step in activity:
                    if step.get("reasoning"):
                        reasoning = step["reasoning"]
                        break

            # Answer type badge
            at_meta = _ANSWER_TYPE_META.get(answer_type, _ANSWER_TYPE_META["book"])
            st.caption(f"{at_meta['icon']} **{at_meta['label']}**")

            st.markdown(answer)

            if reasoning:
                with st.expander("🧠 Agent Dynamic Reasoning & Strategy", expanded=True):
                    st.markdown(reasoning)

            if activity:
                render_agent_activity(activity)

            if sources:
                # Separate book sources from web sources
                book_sources = [s for s in sources if s.get("source") != "web"]
                web_sources  = [s for s in sources if s.get("source") == "web"]

                if book_sources:
                    with st.expander("📖 Book Sources & Citations"):
                        for source in book_sources:
                            lines_info = (
                                f", lines {source['line_start']}-{source['line_end']}"
                                if "line_start" in source
                                else ""
                            )
                            kb_info = f" [{source.get('kb_name', 'kb')}]" if source.get('kb_name') else ""
                            st.markdown(
                                f"**[{source['id']}] {source.get('book', 'Book')}{kb_info} — {source['section']}** "
                                f"({source.get('chapter','')}, "
                                f"pp. {source['page_start']}-{source['page_end']}{lines_info})"
                            )
                            st.write(source["text"])

                if web_sources:
                    with st.expander("🌐 Web Sources"):
                        for source in web_sources:
                            st.markdown(
                                f"**[{source['id']}] [{source.get('title', source['id'])}]({source.get('url', '')})**"
                            )
                            st.write(source.get("text", ""))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "activity": activity,
                "reasoning": reasoning,
                "answer_type": answer_type,
            }
        )
        
        # Save to persistent storage
        save_chat_history(st.session_state.messages, st.session_state.current_session)

    except Exception as error:
        error_message = f"Could not answer this question: {error}"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )
        # Save to persistent storage
        save_chat_history(st.session_state.messages, st.session_state.current_session)
        
        with st.chat_message("assistant"):
            st.error(error_message)
