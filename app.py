from typing import List, Dict

import streamlit as st

from agentic_rag import answer_with_agent
from retrieval.vector_store import get_collection, list_knowledge_bases


st.set_page_config(
    page_title="Agentic Book Chat",
    page_icon="📚",
    layout="wide",
)


def render_agent_activity(activity: List[Dict]):
    with st.expander("Agent Activity"):
        for step_number, step in enumerate(activity, start=1):
            st.markdown(f"### Step {step_number}")

            if step.get("action"):
                st.markdown(f"**Action:** {step['action']}")

            elif step.get("tool"):
                st.markdown(f"**Tool:** `{step['tool']}`")

            action_input = step.get("input")
            if action_input:
                st.markdown(f"**Input:** {action_input}")

            outcome = step.get("result", "Completed")
            tool_name = step.get("tool")

            if (
                tool_name == "list_book_structure"
                and outcome.startswith("Found ")
            ):
                outcome = (
                    f"Retrieved book structure "
                    f"({outcome.removeprefix('Found ')})"
                )

            elif tool_name in {
                "chapter_search",
                "section_search",
                "page_search",
                "semantic_search",
            }:
                outcome = outcome.replace(
                    " passages",
                    " supporting passages",
                )

            elif (
                tool_name == "calculator"
                and outcome.startswith("Result:")
            ):
                outcome = (
                    f"Calculation complete: "
                    f"{outcome.removeprefix('Result:').strip()}"
                )

            st.markdown(f"**Outcome:** {outcome}")

            citation_status = step.get("citation_status")

            if citation_status == "validated":
                st.success(
                    f"Citation validation passed: "
                    f"{step.get('validated_citations', 0)} "
                    f"citation(s) verified."
                )

            elif citation_status == "rejected":
                st.error(
                    "Citation validation failed. "
                    "Unsupported citations were rejected."
                )

            elif citation_status == "not_required":
                st.info(
                    "Citation validation not required "
                    "for this answer."
                )


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
# Sidebar
# --------------------------------------------------

with st.sidebar:
    selected_kb = st.selectbox(
        "Knowledge Base",
        options=knowledge_bases,
        index=0,
    )

collection = get_collection(selected_kb)

with st.sidebar:
    st.title("📚 Agentic Book Chat")
    st.caption(f"Knowledge Base: {selected_kb}")

    st.metric(
        "Indexed Chunks",
        collection.count(),
    )

    with st.expander("Browse Indexed Chunks"):
        stored = collection.get(
            include=["documents", "metadatas"]
        )

        if stored["ids"]:
            rows = sorted(
                zip(
                    stored["ids"],
                    stored["documents"],
                    stored["metadatas"],
                ),
                key=lambda row: row[0],
            )

            books = sorted(
                {
                    metadata.get("book", "Unknown")
                    for _, _, metadata in rows
                }
            )

            selected_book = st.selectbox(
                "Book",
                ["All Books", *books],
                key="chunk_book",
            )

            if selected_book != "All Books":
                rows = [
                    row
                    for row in rows
                    if row[2].get("book") == selected_book
                ]

            sections = sorted(
                {
                    metadata.get("section", "Unknown")
                    for _, _, metadata in rows
                }
            )

            selected_section = st.selectbox(
                "Section",
                ["All Sections", *sections],
                key="chunk_section",
            )

            if selected_section != "All Sections":
                rows = [
                    row
                    for row in rows
                    if row[2].get("section") == selected_section
                ]

            page_size = 5

            page_count = max(
                1,
                (len(rows) + page_size - 1) // page_size,
            )

            page_number = st.number_input(
                "Chunk Page",
                min_value=1,
                max_value=page_count,
                value=1,
            )

            start = (page_number - 1) * page_size

            for (
                chunk_id,
                text,
                metadata,
            ) in rows[start : start + page_size]:

                title = (
                    f"{chunk_id} | "
                    f"pp. {metadata.get('page_start', '?')}-"
                    f"{metadata.get('page_end', '?')}"
                )

                with st.expander(title):
                    st.markdown(
                        f"**{metadata.get('section', 'Unknown')}**"
                    )

                    st.caption(
                        f"Chapter: "
                        f"{metadata.get('chapter', 'Unknown')}"
                    )

                    st.write(text)
        else:
            st.info("No chunks available.")


# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("📚 Agentic Book Chat")

st.caption(
    "Ask questions about the selected knowledge base. "
    "Answers include supporting citations."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("activity"):
            render_agent_activity(message["activity"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['id']}] "
                        f"{source['section']}** "
                        f"(Chapter {source['chapter']}, "
                        f"pages {source['page_start']}-"
                        f"{source['page_end']})"
                    )

                    st.write(source["text"])


if not st.session_state.messages:
    st.info(
        "Ask a question about the book to start a conversation."
    )


question = st.chat_input(
    "Ask a question about the selected knowledge base..."
)

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
            with st.spinner("Searching knowledge base..."):
                answer, sources, activity = answer_with_agent(
                    question,
                    history=history,
                    kb_name=selected_kb,
                )

            st.markdown(answer)

            if activity:
                render_agent_activity(activity)

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(
                            f"**[{source['id']}] "
                            f"{source['section']}** "
                            f"(Chapter {source['chapter']}, "
                            f"pages {source['page_start']}-"
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
        error_message = (
            f"Could not answer this question: {error}"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )

        with st.chat_message("assistant"):
            st.error(error_message)