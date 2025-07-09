import fitz  # PyMuPDF
import pandas as pd
import docx
import json
from transformers import pipeline

def extract_text(file, file_type):
    if file_type == "pdf":
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif file_type == "docx":
        return "\n".join([p.text for p in docx.Document(file).paragraphs])
    elif file_type == "txt":
        return file.read().decode("utf-8")
    elif file_type == "csv":
        df = pd.read_csv(file)
        return df.to_string(index=False)
    elif file_type == "json":
        return json.dumps(json.load(file), indent=2)
    else:
        return "Unsupported file type"

# Use an instruction-tuned model (abstractive)
summarizer = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")


def split_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def summarize_text(text):
    instruction = (
        "Summarize the purpose and structure of the following document in 3-5 sentences. "
        "Focus on what the document is about, not just repeating its content.\n\n"
    )

    chunks = split_text(text)
    summaries = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 100:
            continue
        prompt = instruction + chunk
        try:
            result = summarizer(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
            summaries.append(result)
        except Exception as e:
            summaries.append(f"⚠️ Error in chunk {i+1}: {e}")

    return "\n\n".join(summaries)
