import streamlit as st
from utils.processing import extract_text, summarize_text
from utils.qa import (
    get_embeddings_model,
    create_vector_store,
    retrieve_chunks,
    load_local_llm,
    generate_answer
)

st.set_page_config(page_title="DocIQ - Doc Summarizer + QA", layout="wide")
st.title("📚 DocIQ — AI-powered Document Summarizer & Q&A")
st.caption("🚀 Created by Tej Kumar Sahu | Works offline using open-source models")

# File upload
uploaded_file = st.file_uploader("📂 Upload a document", type=["pdf", "docx", "txt", "csv", "json"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = extract_text(uploaded_file, file_type)
    st.text_area("📄 Extracted Text", text[:3000], height=300)

    if st.button("🧠 Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = summarize_text(text)
            st.subheader("📌 Summary")
            st.write(summary)

    if st.button("🔎 Prepare for Q&A"):
        with st.spinner("Indexing document..."):
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            embed_model = get_embeddings_model()
            index, chunk_embeddings = create_vector_store(chunks, embed_model)

            st.session_state['chunks'] = chunks
            st.session_state['embed_model'] = embed_model
            st.session_state['index'] = index
            st.session_state['chunk_embeddings'] = chunk_embeddings

            st.success("✅ Document ready for question answering!")

# QA Section
if all(key in st.session_state for key in ['index', 'chunks', 'embed_model', 'chunk_embeddings']):
    query = st.text_input("💬 Ask a question from the document:")

    if query:
        with st.spinner("Generating answer..."):
            relevant_chunks = retrieve_chunks(
                query,
                st.session_state['chunks'],
                st.session_state['embed_model'],
                st.session_state['index'],
                st.session_state['chunk_embeddings']
            )

            # Cache the LLM to avoid reloading on every rerun
            if 'llm' not in st.session_state:
                st.session_state['llm'] = load_local_llm()

            llm = st.session_state['llm']
            context = "\n".join(relevant_chunks)
            answer = generate_answer(llm, context, query)

            st.subheader("🧠 Answer")
            st.write(answer)
