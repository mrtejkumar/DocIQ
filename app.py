import streamlit as st
from utils.processing import extract_text, summarize_text
from utils.qa import (
    get_embeddings_model,
    create_vector_store,
    retrieve_chunks,
    load_local_llm,
    generate_answer
)
import time

st.set_page_config(page_title="DocIQ", layout="wide")
st.title("📚 DocIQ - Your Document Intelligence Assistant")
st.caption("🚀 Created by Tej Kumar Sahu | Works using open-source models")

# --- SESSION STATE INIT ---
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "show_summary" not in st.session_state:
    st.session_state["show_summary"] = True
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "embed_model" not in st.session_state:
    st.session_state["embed_model"] = None
if "index" not in st.session_state:
    st.session_state["index"] = None
if "chunk_embeddings" not in st.session_state:
    st.session_state["chunk_embeddings"] = None

# --- FILE/TEXT INPUT SECTION ---
st.markdown("### 📤 Upload a Document or Paste Text")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "csv", "json"])
text_input = st.text_area("Or paste raw text below 👇", height=200)

extracted_text = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    extracted_text = extract_text(uploaded_file, file_type)
elif text_input.strip() and st.button("📥 Submit Text"):
    extracted_text = text_input.strip()

if extracted_text:
    st.markdown("### 📝 Extracted Text")
    with st.expander("Click to view extracted text"):
        st.text_area("Extracted Text", extracted_text[:3000], height=300, disabled=True)

    if st.button("🧠 Generate Summary"):
        with st.spinner("Generating summary..."):
            st.session_state["summary"] = summarize_text(extracted_text)
            st.session_state["show_summary"] = True

    if st.session_state["summary"]:
        if st.button("👁️ Toggle Summary"):
            st.session_state["show_summary"] = not st.session_state["show_summary"]

        if st.session_state["show_summary"]:
            st.markdown("### 📌 Summary")
            st.markdown(st.session_state["summary"])

    if st.button("🔎 Prepare for Question Answering"):
        st.session_state["chunks"] = [extracted_text[i:i + 1000] for i in range(0, len(extracted_text), 1000)]
        st.session_state["embed_model"] = get_embeddings_model()
        st.session_state["index"], st.session_state["chunk_embeddings"] = create_vector_store(
            st.session_state["chunks"],
            st.session_state["embed_model"]
        )
        st.success("✅ Document indexed and ready for Q&A!")

# --- Q&A SECTION ---
if st.session_state.get("index"):
    st.markdown("### 💬 Ask a question about the document")
    question = st.text_input("Your Question")

    if st.button("🚀 Submit Question"):
        with st.spinner("🔄 Submitting your question..."):
            st.info("🧠 Analyzing document...")
            time.sleep(0.5)
            st.info("✂️ Tokenizing text...")
            time.sleep(0.5)
            st.info("📊 Creating vector representation...")
            time.sleep(0.5)

            try:
                chunks = retrieve_chunks(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embed_model"],
                    st.session_state["index"],
                    st.session_state["chunk_embeddings"]
                )

                st.info("🤖 Generating answer… (Please wait)")
                with st.spinner("⏳ This might take a few seconds..."):
                    llm = load_local_llm()
                    answer = generate_answer(llm, "\n".join(chunks), question)
                    st.success("✅ Answer generated!")

                st.markdown("### 🧠 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
