import fitz  # PyMuPDF
import pandas as pd
import docx
import json
from transformers import pipeline

# Load once
summarizer = pipeline(
    "summarization",
    model="google/flan-t5-base", 
    tokenizer="google/flan-t5-base"
)

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

# def summarize_text(text):
#     if len(text) > 20000:
#         return "❌ Text too long to summarize. Please upload a smaller document (max ~3 pages or 20,000 characters)."

#     instruction_prompt = (
#         "You are an intelligent summarization assistant.\n\n"
#         "Your job is to analyze the content below and generate a clear, structured summary:\n"
#         "- If it's a timetable or schedule, list the key time blocks and what happens in each.\n"
#         "- If it's a story, extract key events and the moral.\n"
#         "- If it's a report or informative text, summarize key points and conclusions.\n\n"
#         "Output should be well-formatted with headings, bullet points, and a goal/moral if applicable.\n\n"
#         "Content:\n"
#         f"{text.strip()}"
#     )

#     try:
#         result = summarizer(instruction_prompt, max_length=512, min_length=100, do_sample=False)[0]['summary_text']
#         return result
#     except Exception as e:
#         return f"⚠️ Summarization failed: {e}"
    
def summarize_text(text):
    if len(text) > 20000:
        return "❌ Text too long to summarize. Please upload a smaller document (max ~3 pages or 20,000 characters)."

    prompt = (
        "You are an intelligent summarization assistant.\n\n"
        "Your job is to analyze the content below and generate a clear, structured summary:\n"
        "- If it's a timetable or schedule, provide the details what is this about and list the key time blocks and what happens in each.\n"
        "- If it's a story, extract key events and the moral with proper headings.\n"
        "- If it's a report or informative text, summarize key points and conclusions.\n\n"
        "Output should be well-formatted with headings, bullet points, and a goal/moral if applicable.\n\n"
        "Content:\n"
        f"{text.strip()}"
    )

    try:
        result = summarizer(prompt, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
        return result
    except Exception as e:
        return f"⚠️ Summarization failed: {e}"


