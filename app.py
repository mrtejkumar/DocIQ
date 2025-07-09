import streamlit as st
import time
from utils.processing import extract_text, summarize_text
from utils.qa import (
    get_embeddings_model,
    create_vector_store,
    retrieve_chunks,
    load_local_llm,
    generate_answer
)

st.set_page_config(page_title="DocIQ", layout="wide")
st.title("📚 DocIQ - Your Document Intelligence Assistant")
st.caption("🚀 Created by Tej Kumar Sahu | Powered by open-source models")

# --- SESSION STATE INIT ---
st.session_state.setdefault("summary", "")
st.session_state.setdefault("show_summary", True)
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("embed_model", None)
st.session_state.setdefault("index", None)
st.session_state.setdefault("chunk_embeddings", None)
st.session_state.setdefault("extracted_text", "")

# --- FILE/TEXT INPUT SECTION ---
st.markdown("### 📤 Upload a Document or Paste Text")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "json"])
text_input = st.text_area("Paste raw text below 👇", height=200)

submit_text = st.button("📥 Submit Text")

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    st.session_state.extracted_text = extract_text(uploaded_file, file_type)
elif submit_text and text_input.strip():
    st.session_state.extracted_text = text_input.strip()

if st.session_state.extracted_text:
    st.markdown("### 📝 Extracted Text")
    with st.expander("🔍 Click to view extracted text"):
        st.text_area("Extracted Text View", st.session_state.extracted_text[:3000], height=300, disabled=True)

    if st.button("🧠 Generate Summary"):
        with st.spinner("Generating summary..."):
            st.session_state.summary = summarize_text(st.session_state.extracted_text)
            st.session_state.show_summary = True

    if st.session_state.summary:
        col1, col2 = st.columns([1, 5])
        with col1:
            toggle = st.button("👁️ Toggle Summary")
            if toggle:
                st.session_state.show_summary = not st.session_state.show_summary

        if st.session_state.show_summary:
            st.markdown("### 📌 Summary")
            st.markdown(st.session_state.summary)

    if st.button("🔎 Prepare for Question Answering"):
        st.session_state.chunks = [st.session_state.extracted_text[i:i + 1000] for i in range(0, len(st.session_state.extracted_text), 1000)]
        st.session_state.embed_model = get_embeddings_model()
        st.session_state.index, st.session_state.chunk_embeddings = create_vector_store(
            st.session_state.chunks,
            st.session_state.embed_model
        )
        st.success("✅ Document indexed and ready for Q&A!")

# --- Q&A SECTION ---
if st.session_state.index:
    st.markdown("### 💬 Ask a question about the document")
    question = st.text_input("Enter your question below")

    if st.button("🚀 Submit Question"):
        st.markdown("#### 📡 Processing Status")
        status = st.empty()
        progress = st.progress(0)

        try:
            status.info("🔄 Question submitted...")
            progress.progress(10)
            time.sleep(0.4)

            status.info("📚 Analyzing document...")
            progress.progress(30)
            time.sleep(0.4)

            status.info("✂️ Tokenizing data...")
            progress.progress(50)
            time.sleep(0.4)

            status.info("📊 Vectorizing question...")
            progress.progress(70)
            time.sleep(0.4)

            chunks = retrieve_chunks(
                question,
                st.session_state.chunks,
                st.session_state.embed_model,
                st.session_state.index,
                st.session_state.chunk_embeddings
            )

            status.info("🤖 Generating response...")
            progress.progress(90)

            with st.spinner("⏳ This might take a few seconds..."):
                llm = load_local_llm()
                answer = generate_answer(llm, "\n".join(chunks), question)

            status.success("✅ Done!")
            progress.progress(100)
            st.markdown("### 🧠 Answer")
            st.write(answer)

        except Exception as e:
            status.error("❌ Something went wrong!")
            st.error(f"Error: {e}")
