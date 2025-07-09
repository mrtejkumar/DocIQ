# 📚 DocIQ - AI-Powered Document Summarizer & Q&A

DocIQ is a Streamlit-based web application that allows you to:

- ✅ Upload documents in multiple formats (PDF, DOCX, TXT, CSV, JSON)
- ✅ Automatically extract and summarize the content
- ✅ Ask questions about the document and get accurate answers
- ✅ Use completely free and open-source AI models offline

---

## 🚀 Features

| Feature                | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| Multi-format Support   | Upload PDF, Word (.docx), text, CSV, or JSON files                 |
| Text Extraction        | Extracts and displays text in a readable format                    |
| AI Summarization       | Uses google/flan-t5-base with guided prompts to summarize content  |
| Local Q&A              | Uses Mistral-7B-Instruct (GGUF format) via ctransformers           |
| Embedding & Retrieval  | Uses all-MiniLM-L6-v2 for embedding & FAISS for similarity search  |
| Offline Support        | No need for internet or OpenAI API keys                            |

---

## 💂 Folder Structure

```
docIQ/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── models/
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf   # LLM model file
├── utils/
│   ├── processing.py            # File parsing & summarization code
│   └── qa.py                    # Embedding, vector store, and LLM interaction
```

---

## 🧠 AI Models Used

### 1. Summarization Model: `google/flan-t5-base`
- **Type:** Instruction-tuned encoder-decoder model  
- **Source:** Hugging Face - flan-t5-base  
- **Task:** Abstractive summarization with guided prompts  
- **Usage:**
    ```python
    from transformers import pipeline
    summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = "Summarize the purpose of the following document: ..."
    summarizer(prompt, max_new_tokens=200)
    ```
- **Token Limit:** ~2048 tokens input context  
- **Why this model?:** More robust at generalizing across bullet points, paragraphs, lists, and mixed content. Performs well with prompts.

### 2. Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Type:** Sentence-level BERT-style embeddings  
- **Source:** Hugging Face - Sentence Transformers  
- **Task:** Converts text chunks into embeddings for vector search  
- **Why?:** Small, fast, and provides surprisingly good semantic results  
- **Embedding Dim:** 384

### 3. Vector Store: FAISS
- **Type:** Local similarity search index  
- **Tool:** FAISS by Facebook AI  
- **Why?:** Fast and memory-efficient for document chunk retrieval

### 4. LLM for Q&A: `Mistral-7B-Instruct-v0.1.Q4_K_M.gguf`
- **Type:** Quantized local large language model  
- **Format:** GGUF (for CPU/GPU inference with ctransformers)  
- **Source:** TheBloke - HuggingFace  
- **Quantization:** Q4_K_M (4-bit, optimized for size/speed)  
- **Context Limit:** 2048 tokens  
- **Engine:** ctransformers for fast local inference  
- **Prompt Format:**
    ```python
    prompt = f"[INST] Context:\n{context}\n\nQuestion: {question} [/INST]"
    answer = llm(prompt)
    ```

---

## 🛠️ Setup Instructions

1. **Clone the Repository**
    ```sh
    git clone https://github.com/your-username/dociq.git
    cd dociq
    ```
2. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```
3. **Download the Mistral GGUF Model**
    - Go to: [Mistral-7B-Instruct-v0.1-GGUF on Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
    - Download: `mistral-7b-instruct-v0.1.Q4_K_M.gguf`
    - Place in: `models/` folder

4. **Run the App**
    ```sh
    streamlit run app.py
    ```

---

## 🧚‍♂️ Example Usage

- Upload a file (e.g. `timetable.pdf`)
- See extracted content in the text area
- Click **🧠 Generate Summary** to get a brief version
- Click **🔎 Prepare for Q&A** to index content
- Ask: *"What is the office work timing?"* → Get answer from document

---

## ⚠️ Limitations

- The LLM context is capped at ~2048 tokens. Avoid very large inputs.
- Local models are slower and may require RAM (6–8GB recommended).
- Accuracy depends on chunking + model capabilities.

---

## 📌 Future Improvements

- Multi-document support
- Chat history and memory
- Support for more formats (e.g. PPTX, HTML)
- Optional online fallback via Hugging Face Inference API

---

## 👨‍💼 Credits

Created by **Tej Kumar Sahu** using open-source AI tools:

- Streamlit
- Hugging Face Transformers
- cTransformers + GGUF LLMs
- SentenceTransformers
- FAISS
- PyMuPDF, docx, pandas, etc.

---

## 📄 License

MIT License. Free to use, fork, and contribute.