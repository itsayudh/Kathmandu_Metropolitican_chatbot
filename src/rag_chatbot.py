import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent 
PERSIST_DIR_EN = BASE_DIR / "chroma_db_en"

# --- Load API key securely from .env file ---
# NEVER hard-code your API key in the script.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")
genai.configure(api_key=api_key)
# --- END FIX ---

gemini_model = "gemini-2.5-flash"

# ----------------------------------------------------------------------
# EMBEDDING & VECTOR STORE
# ----------------------------------------------------------------------
print("Loading English vector store and embedding model...")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR_EN),
    embedding_function=embedder
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# ----------------------------------------------------------------------
# LANGUAGE DETECTION
# ----------------------------------------------------------------------
NEPALI_PATTERN = re.compile(r'[\u0900-\u097F]')

def detect_language(text: str) -> str:
    if NEPALI_PATTERN.search(text):
        return "nepali"
    return "english"

# ----------------------------------------------------------------------
# GEMINI PROMPTS
# ----------------------------------------------------------------------
def build_prompt(question: str, context_docs: List[Document], lang: str) -> str:
    context = "\n\n".join([doc.page_content for doc in context_docs])

    if lang == "nepali":
        system = "तपाईं एक सहयोगी सरकारी सेवा च्याटबट हुनुहुन्छ। प्रश्न नेपालीमा सोधिएको छ, जवाफ पनि नेपालीमा दिनुहोस्।"
        instruction = f"प्रश्न: {question}\n\nसन्दर्भ जानकारी:\n{context}\n\nजवाफ छोटो, स्पष्ट र नेपालीमा दिनुहोस्।"
    else:
        system = "You are a helpful government service chatbot. Answer in English."
        instruction = f"Question: {question}\n\nContext:\n{context}\n\nAnswer concisely in English."

    return f"{system}\n\n{instruction}"

# ----------------------------------------------------------------------
# TRANSLATION (using Gemini)
# ----------------------------------------------------------------------
def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "nepali":
        prompt = f"Translate the following English text to natural Nepali:\n\n{text}"
    else:
        prompt = f"Translate the following Nepali text to natural English:\n\n{text}"

    try:
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # fallback

# ----------------------------------------------------------------------
# RAG PIPELINE
# ----------------------------------------------------------------------
def rag_answer(question: str) -> Tuple[str, str]:
    lang = detect_language(question)

    # Retrieve in English (vector DB is English)
    docs = retriever.invoke(question)

    if not docs:
        if lang == "nepali":
            return "माफ गर्नुहोस्, मसँग यो सेवाको बारेमा जानकारी छैन।", lang
        else:
            return "Sorry, I don't have information about this service.", lang

    # Build prompt
    prompt = build_prompt(question, docs, lang)

    try:
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        answer = response.text.strip()

        # If question in Nepali but answer in English → translate
        if lang == "nepali" and not NEPALI_PATTERN.search(answer):
            answer = translate_text(answer, "nepali")
        # If question in English but answer has Nepali → translate
        elif lang == "english" and NEPALI_PATTERN.search(answer):
            answer = translate_text(answer, "english")

        return answer, lang

    except Exception as e:
        print(f"Gemini error: {e}")
        fallback = "माफ गर्नुहोस्, जवाफ दिने क्रममा समस्या भयो।" if lang == "nepali" else "Sorry, there was an error generating the response."
        return fallback, lang

# ----------------------------------------------------------------------
# CHAT LOOP
# ----------------------------------------------------------------------
def chat():
    print("\nGovernment Service Chatbot (EN/NP)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        if not question:
            continue

        answer, lang = rag_answer(question)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    chat()