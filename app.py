from typing import List, Dict

import streamlit as st

from agentic_rag import answer_with_agent
from retrieval.vector_store import get_collection, list_knowledge_bases


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
    "list_books":      {"icon": "📋", "label": "List Books"},
    "semantic_search": {"icon": "🔍", "label": "Semantic Search (Book)"},
    "web_search":      {"icon": "🌐", "label": "Web Search"},
}


def render_agent_activity(activity: List[Dict]):
    """Render detailed tool execution trace with per-tool reasoning, repeated call explanations, and findings."""
    trace_steps = [
        s for s in activity
        if s.get("action") != "Agent Dynamic Reasoning & Strategy"
    ]
    if not trace_steps:
        return

    with st.expander("⚙️ Tool Execution Trace (Step-by-Step Reasons & Discoveries)", expanded=False):
        for step_number, step in enumerate(trace_steps, start=1):
            tool_name = step.get("tool")
            action = step.get("action")
            meta = _TOOL_META.get(tool_name, {"icon": "🔧", "label": tool_name or action})

            st.markdown(f"---\n### {meta['icon']} Step {step_number} — {meta['label']}")

            # 1. WHY this tool was called
            tool_reasoning = step.get("tool_reasoning")
            if tool_reasoning:
                st.info(f"🎯 **Why this tool was called:**\n\n{tool_reasoning}")

            # 2. WHY a repeated call occurred (if this tool was invoked >1 time)
            why_repeated = step.get("why_repeated")
            if why_repeated:
                st.warning(f"🔁 **Why this repeated call was needed:**\n\n{why_repeated}")

            # 3. WHY web search instead of semantic search (for web_search)
            why_web = step.get("why_web_instead_of_semantic")
            if why_web:
                st.markdown(
                    f"> 🌐 **Decision: Why Web Search instead of Book Search?**  \n"
                    f"> *{why_web}*"
                )

            # 4. WHAT was sent to the tool
            action_input = step.get("input")
            if action_input:
                st.markdown(f"**📤 Query Sent:** `{action_input}`")

            # 5. WHAT list_books discovered (routing map)
            routing_map = step.get("routing_map")
            if routing_map and isinstance(routing_map, dict):
                st.markdown("**📋 Discovered Knowledge Bases:**")
                for kb, title in routing_map.items():
                    st.markdown(f"&nbsp;&nbsp;• `{kb}` = *{title}*")

            # 6. WHAT was found
            found_summary = step.get("found_summary") or step.get("result", "Completed")
            matches = step.get("matches", 0)

            st.markdown(f"**📥 What was found:** {found_summary}")

            what_found_items = step.get("what_found_items") or []
            if what_found_items and tool_name == "semantic_search":
                with st.expander(f"View {len(what_found_items)} retrieved chunk metadata (pages & lines)", expanded=False):
                    for item in what_found_items:
                        st.markdown(f"- {item}")

            elif tool_name == "web_search":
                web_results = step.get("web_results") or []
                if web_results:
                    with st.expander(f"View {len(web_results)} web search results & links", expanded=False):
                        for r in web_results:
                            st.markdown(
                                f"**[{r['rank']}] [{r['title']}]({r['url']})** *(via {r.get('engine','Web')})*  \n"
                                f"{r.get('snippet', '')}"
                            )

            # 7. Citation status
            citation_status = step.get("citation_status")
            answer_type_label = step.get("answer_type", "")
            if citation_status == "validated":
                st.success(
                    f"✓ Citations validated: {step.get('validated_citations', 0)} "
                    f"verified ({answer_type_label} answer)."
                )
            elif citation_status == "rejected":
                st.error("✗ Citation validation failed.")
            elif citation_status == "not_required":
                st.info("ℹ️ No citation validation required for this answer.")



# --------------------------------------------------
# Load Knowledge Bases
# --------------------------------------------------

knowledge_bases = list_knowledge_bases()

if not knowledge_bases:
    st.warning("No knowledge bases are indexed yet.")

    st.code(
        '.\\.venv\\Scripts\\python.exe ingest.py '
        '"data1\\Mind Management, Not Time Management.pdf" '
        '--book "Mind Management" --kb kb1'
    )

    st.stop()


# --------------------------------------------------
# Sidebar: Dynamic Knowledge Bases & Viewer
# --------------------------------------------------

with st.sidebar:
    st.title("📚 Multi-Source Knowledge Bases")

    st.success(
        "✅ The agent autonomously decides which books to search. "
        "No manual selection needed — it reasons dynamically from your question."
    )

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

    st.divider()

    st.markdown("### 🔍 Chunk Inspector *(read-only — does NOT affect agent)*")
    st.caption("Browse raw chunks stored in a collection. This selector has no effect on how the agent searches.")

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

if "messages" not in st.session_state:
    st.session_state.messages = []


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
                        st.markdown(
                            f"**[{source['id']}] {source.get('book', 'Book')} — {source['section']}** "
                            f"(Chapter {source.get('chapter','')}, "
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


if not st.session_state.messages:
    st.info(
        "💡 **Ask any question to get started. Examples:**\n"
        "- 📖 **Multi-Book Comparison**: *'Compare focus and distraction in Mind Management and Deep Work'* *(agent queries both collections & explains why)*\n"
        "- 🎯 **Line & Page Lookup**: *'What does line 12 of page 10 say in Mind Management?'* *(precise metadata retrieval)*\n"
        "- 📚 **Chapter Deep Dive**: *'What are the core arguments in Deep Work chapter 1?'* *(chapter-filtered retrieval)*\n"
        "- 🌐 **External / Web Query**: *'Who is the CEO of Google and what are recent AI developments?'* *(agent detects topic is outside books & switches to web search)*"
    )


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
                result = answer_with_agent(
                    question,
                    history=history,
                    kb_name=None,  # No hardcoding - agent dynamically decides
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
                            st.markdown(
                                f"**[{source['id']}] {source.get('book', 'Book')} — {source['section']}** "
                                f"(Chapter {source.get('chapter','')}, "
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

    except Exception as error:
        error_message = f"Could not answer this question: {error}"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )
        with st.chat_message("assistant"):
            st.error(error_message)
