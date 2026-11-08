import sys

# Increase recursion limit
sys.setrecursionlimit(10000)

# Monkey-patch to fix Python 3.13 + truststore recursion bug
import ssl
if sys.version_info >= (3, 13):
    _SSLContext = ssl.SSLContext
    _original_init = _SSLContext.__init__
    
    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        try:
            self.check_hostname = False
            self._verify_mode = ssl.CERT_NONE
        except:
            pass
    
    _SSLContext.__init__ = _patched_init

import streamlit as st

from agentic_rag import answer_with_agent
from retrieval.vector_store import get_collection, list_knowledge_bases


st.set_page_config(page_title="Mind Management Chat", page_icon="📚", layout="wide")


def render_agent_activity(activity: list[dict]):
    with st.expander("Agent activity"):
        for step_number, step in enumerate(activity, start=1):
            st.markdown(f"**Step {step_number}**")
            if step.get("action"):
                st.markdown(f"**Action:** {step['action']}")
            elif step.get("tool"):
                st.markdown(f"**Tool:** `{step['tool']}`")

            action_input = step.get("input")
            if action_input:
                st.markdown(f"**Input:** {action_input}")

            outcome = step.get("result", "Completed")
            tool_name = step.get("tool")
            if tool_name == "list_book_structure" and outcome.startswith("Found "):
                outcome = f"Retrieved book structure ({outcome.removeprefix('Found ')})"
            elif tool_name in {"chapter_search", "section_search", "page_search", "semantic_search"}:
                outcome = outcome.replace("Found ", "Found ", 1).replace(
                    " passages", " supporting passages"
                )
            elif tool_name == "calculator" and outcome.startswith("Result:"):
                outcome = f"Calculation complete: {outcome.removeprefix('Result:').strip()}"
            st.markdown(f"**Outcome:** {outcome}")

            citation_status = step.get("citation_status")
            if citation_status == "validated":
                st.success(
                    f"Citation validation passed: {step.get('validated_citations', 0)} "
                    "citation(s) match chunks retrieved this turn."
                )
            elif citation_status == "rejected":
                st.error("Citation validation failed. Unsupported citations were rejected.")
            elif citation_status == "not_required":
                st.info("Citation validation not required for this tool-only answer.")


knowledge_bases = list_knowledge_bases()
if not knowledge_bases:
    st.warning("No knowledge bases are indexed yet.")
    st.code(
        '.\\.venv\\Scripts\\python.exe ingest.py "data1\\Mind Management, Not Time Management.pdf" --book "Mind Management" --kb kb1'
    )
    st.stop()

with st.sidebar:
    selected_kbs = st.multiselect(
        "Knowledge bases", 
        knowledge_bases,
        default=[knowledge_bases[0]] if knowledge_bases else []
    )
    
if not selected_kbs:
    st.warning("Please select at least one knowledge base")
    st.stop()

# For now, use the first selected KB (single KB mode)
# Multi-KB queries would need agent to query each KB separately
selected_kb = selected_kbs[0]

collection = get_collection(selected_kb)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Mind Management")
    st.caption(f"Searching {selected_kb}")
    st.metric("Indexed chunks", collection.count())

    with st.expander("Browse indexed chunks"):
        stored = collection.get(include=["documents", "metadatas"])
        rows = sorted(
            zip(stored["ids"], stored["documents"], stored["metadatas"]),
            key=lambda row: row[0],
        )
        books = sorted({metadata.get("book", "Unknown") for _, _, metadata in rows})
        selected_book = st.selectbox("Book", ["All books", *books], key="chunk_book")
        if selected_book != "All books":
            rows = [row for row in rows if row[2].get("book") == selected_book]
        sections = sorted({metadata.get("section", "Unknown") for _, _, metadata in rows})
        selected_section = st.selectbox(
            "Section", ["All sections", *sections], key="chunk_section"
        )
        if selected_section != "All sections":
            rows = [row for row in rows if row[2].get("section") == selected_section]

        page_size = 5
        page_count = max(1, (len(rows) + page_size - 1) // page_size)
        page_number = st.number_input(
            "Chunk page", min_value=1, max_value=page_count, value=1, key="chunk_page"
        )
        start = (page_number - 1) * page_size

        for chunk_id, text, metadata in rows[start : start + page_size]:
            title = (
                f"{chunk_id} | pp. {metadata.get('page_start', '?')}-"
                f"{metadata.get('page_end', '?')}"
            )
            with st.expander(title):
                st.markdown(f"**{metadata.get('section', 'Unknown')}**")
                st.caption(f"Chapter: {metadata.get('chapter', 'Unknown')}")
                st.write(text)

st.title("Mind Management Chat")
st.caption("Ask about the book. Answers use relevant passages and include page citations.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("activity"):
            render_agent_activity(message["activity"])
        if message.get("sources"):
            with st.expander("Sources"):
                for rank, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**[{source['id']}] {source['section']}** "
                        f"(Chapter {source['chapter']}, pages {source['page_start']}-"
                        f"{source['page_end']})"
                    )
                    st.write(source["text"])

if not st.session_state.messages:
    st.info("Ask a question about the book to start a conversation.")

question = st.chat_input("Ask a question about the book...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[:-1]
    ]
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the book..."):
                answer, sources, activity = answer_with_agent(
                    question, history=history, kb_name=selected_kb
                )
            st.markdown(answer)
            if activity:
                render_agent_activity(activity)
            if sources:
                with st.expander("Sources"):
                    for rank, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**[{source['id']}] {source['section']}** "
                            f"(Chapter {source['chapter']}, pages {source['page_start']}-"
                            f"{source['page_end']})"
                        )
                        st.write(source["text"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "activity": activity,
            }
        )
    except Exception as error:
        answer = f"Could not answer this question: {error}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.error(answer)