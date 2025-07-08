import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM

def get_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(text_chunks, embed_model):
    embeddings = embed_model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_chunks(query, text_chunks, embed_model, index, chunk_embeddings, top_k=3):
    query_embedding = embed_model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return [text_chunks[i] for i in indices[0]]

def load_local_llm():
    return AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=20
    )

def generate_answer(llm, context, question):
    prompt = f"Answer the question based on this context:\n\n{context}\n\nQuestion: {question}"
    return llm(prompt)
