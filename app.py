import streamlit as st

from agentic_rag import answer_with_agent
from retrieval import indexed_collection


st.set_page_config(page_title="Mind Management Chat", page_icon="📚", layout="wide")

try:
    collection = indexed_collection()
except Exception:
    st.warning("No indexed PDF found yet.")
    st.code("python ingest.py")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Mind Management")
    st.caption("Answers grounded in the indexed book")
    st.metric("Indexed chunks", collection.count())

    with st.expander("Browse indexed chunks"):
        stored = collection.get(include=["documents", "metadatas"])
        rows = sorted(
            zip(stored["ids"], stored["documents"], stored["metadatas"]),
            key=lambda row: row[0],
        )
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
            with st.expander("Agent activity"):
                for step in message["activity"]:
                    action_input = step.get("input") or "(no input)"
                    st.write(
                        f"**{step['tool']}**: {action_input} -> "
                        f"{step['result']}"
                    )
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
                    question, history=history
                )
            st.markdown(answer)
            if activity:
                with st.expander("Agent activity"):
                    for step in activity:
                        action_input = step.get("input") or "(no input)"
                        st.write(
                            f"**{step['tool']}**: {action_input} -> "
                            f"{step['result']}"
                        )
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