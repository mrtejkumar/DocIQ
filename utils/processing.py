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

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)[0]['summary_text']
