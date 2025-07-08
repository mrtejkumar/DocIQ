import streamlit as st
from utils.processing import extract_text, summarize_text
from utils.qa import get_embeddings_model, create_vector_store, retrieve_chunks, load_local_llm, generate_answer

st.title("📚 Document Summarizer + QA (Open-Source)")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "csv", "json"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = extract_text(uploaded_file, file_type)
    st.text_area("📄 Extracted Text", text[:3000], height=300)

    if st.button("🧠 Generate Summary"):
        summary = summarize_text(text)
        st.subheader("📌 Summary")
        st.write(summary)

    if st.button("🔎 Index for QA"):
        st.session_state['chunks'] = [text[i:i+1000] for i in range(0, len(text), 1000)]
        st.session_state['embed_model'] = get_embeddings_model()
        st.session_state['index'], st.session_state['chunk_embeddings'] = create_vector_store(
            st.session_state['chunks'],
            st.session_state['embed_model']
        )
        st.success("✅ Document indexed!")

if 'index' in st.session_state:
    query = st.text_input("💬 Ask a question from the document:")
    if query:
        chunks = retrieve_chunks(query, st.session_state['chunks'],
                                 st.session_state['embed_model'],
                                 st.session_state['index'],
                                 st.session_state['chunk_embeddings'])
        llm = load_local_llm()
        answer = generate_answer(llm, "\n".join(chunks), query)
        st.subheader("🧠 Answer")
        st.write(answer)
