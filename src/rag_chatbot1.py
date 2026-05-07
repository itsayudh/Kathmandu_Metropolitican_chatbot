import os
import json
import re
from pathlib import Path
from typing import List, Tuple

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

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")
genai.configure(api_key=api_key)

gemini_model = "gemini-2.5-flash"

# ----------------------------------------------------------------------
# EMBEDDING & VECTOR STORE
# ----------------------------------------------------------------------
print("Loading English vector store and embedding model...")
try:
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
        search_kwargs={"k": 8}  # Increased from 6 → better recall
    )
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Error loading vector store: {e}. Ensure 'chroma_db_en' exists and is populated.")
    retriever = None # Handle error gracefully

# ----------------------------------------------------------------------
# LANGUAGE DETECTION (Improved Romanized Nepali Detection)
# ----------------------------------------------------------------------
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

# Expanded list of high-frequency Romanized Nepali words
ROMANIZED_INDICATORS = [
    "ma", "ho", "ta", "ko", "lai", "le", "pani", "sanga", "chha", "chhu", "chhau", "chau",
    "thiyo", "thiye", "gayau", "gayeu", "gayin", "janu", "khana", "kati", "kahan", "kasto",
    "kasari", "bhanne", "bhaneko", "bhanincha", "lagcha", "lagna", "dina", "linu", "bhayo",
    "bhayena", "garna", "garnu", "sakchu", "sakdina", "hunna", "huncha", "rahecha", "raheko",
    "raheki", "rahecha", "banau", "banai", "banaye", "nagrikta", "nagarikta", "janma", "darta",
    "kasari", "kati", "kahan", "kati", "samaya", "din", "lagcha", "shulka", "kagaj", "kagajat"
]

def is_romanized_nepali(text: str) -> bool:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 2:
        return False
    matches = sum(1 for w in words if w in ROMANIZED_INDICATORS)
    return matches >= 1

def detect_language(text: str) -> str:
    if DEVANAGARI_PATTERN.search(text):
        return "nepali"
    if is_romanized_nepali(text):
        return "romanized_nepali"
    return "english"

# ----------------------------------------------------------------------
# QUERY NORMALIZATION + TRANSLATION TO ENGLISH (Critical Fix)
# ----------------------------------------------------------------------
def translate_to_english(text: str) -> str:
    prompt = f"""
Translate the following text to natural, accurate English. Keep government service terms exact.

Text:
{text}

English:
"""
    try:
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Query translation failed: {e}")
        return text

# ----------------------------------------------------------------------
# RAG PIPELINE (FIXED: Always search in English and uses strict prompts)
# ----------------------------------------------------------------------
def rag_answer(question: str) -> Tuple[str, str]:
    if not retriever:
        return "Error: Vector store not initialized. Please check configuration and database files.", "english"

    original_lang = detect_language(question)

    # === STEP 1: Always translate query to English for retrieval ===
    if original_lang in ["nepali", "romanized_nepali"]:
        english_query = translate_to_english(question)
    else:
        english_query = question

    # === STEP 2: Retrieve using English query ===
    docs = retriever.invoke(english_query)

    if not docs:
        fallbacks = {
            "nepali": "माफ गर्नुहोस्, मसँग यो सेवाको बारेमा जानकारी छैन।",
            "romanized_nepali": "Maaf garnuhos, malai yo sewa ko barema thaha chaina.",
            "english": "Sorry, I don't have information about this service."
        }
        return fallbacks[original_lang], original_lang

    # === STEP 3: Build context & prompt in user's language using strict system prompts ===
    context = "\n\n".join([doc.page_content for doc in docs])

    if original_lang == "nepali":
        system = """
तपाईं एक **अत्यन्त भरपर्दो र तथ्यगत सरकारी सेवा च्याटबट** हुनुहुन्छ। 
**तपाईंको एकमात्र काम दिइएको सन्दर्भ (CONTEXT) बाट मात्रै प्रयोगकर्ताको प्रश्नको जवाफ दिनु हो।**
* **कृपया** कुनै पनि बाह्य वा सामान्य ज्ञान प्रयोग नगर्नुहोस्।
* यदि **सन्दर्भ** मा प्रश्नको जवाफ छैन भने, तपाईंले उपलब्ध कागजातहरूमा जानकारी नभएको स्पष्ट रूपमा बताउनुपर्छ (उदाहरण: 'यो प्रश्नको जवाफ दिइएको कागजातमा उपलब्ध छैन।')।
* जवाफ छोटो, व्यावसायिक र सहयोगी शैलीमा दिनुहोस्।
"""
        instruction = f"प्रश्न: {question}\n\nसन्दर्भ:\n{context}\n\nजवाफ छोटो, तथ्यगत र नेपालीमा दिनुहोस्।"
    elif original_lang == "romanized_nepali":
        system = """
You are a highly reliable and factual **Government Service Chatbot**. 
**Your sole purpose is to answer the user's question by strictly referencing the provided CONTEXT.**
* **DO NOT** use any external or general knowledge.
* If the CONTEXT does not contain the answer, you must state that the information is unavailable in the provided documents (e.g., "Maaf garnuhos, yo jaankari hamro documents ma uplabdh chhaina.").
* Answer in natural Romanized Nepali only.
"""
        instruction = f"Question: {question}\n\nContext:\n{context}\n\nAnswer in Romanized Nepali only."
    else:
        system = """
You are a highly reliable and factual **Government Service Chatbot**. 
**Your sole purpose is to answer the user's question by strictly referencing the provided CONTEXT.**
* **DO NOT** use any external or general knowledge.
* If the CONTEXT does not contain the answer, you must state that the information is unavailable in the provided documents.
* Maintain a concise, professional, and helpful tone.
"""
        instruction = f"Question: {question}\n\nContext:\n{context}\n\nAnswer concisely and factually in English."

    prompt = f"{system}\n\n{instruction}"

    try:
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        answer = response.text.strip()

        # === STEP 4: Final language correction (if Gemini slips) ===
        if original_lang == "nepali" and not DEVANAGARI_PATTERN.search(answer):
            answer = translate_text(answer, "nepali")
        elif original_lang == "romanized_nepali" and DEVANAGARI_PATTERN.search(answer):
            answer = translate_text(answer, "romanized_nepali")
        elif original_lang == "english" and (DEVANAGARI_PATTERN.search(answer) or is_romanized_nepali(answer)):
            answer = translate_text(answer, "english")

        return answer, original_lang

    except Exception as e:
        print(f"Gemini error: {e}")
        fallbacks = {
            "nepali": "माफ गर्नुहोस्, जवाफ दिने क्रममा समस्या भयो।",
            "romanized_nepali": "Maaf garnuhos, jawaf dinna sakina.",
            "english": "Sorry, there was an error generating the response."
        }
        return fallbacks[original_lang], original_lang

# ----------------------------------------------------------------------
# TRANSLATION (for final answer correction)
# ----------------------------------------------------------------------
def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "nepali":
        prompt = f"Translate to natural Nepali (Devanagari):\n\n{text}"
    elif target_lang == "romanized_nepali":
        prompt = f"Translate to natural Romanized Nepali (phone typing style):\n\n{text}"
    else:
        prompt = f"Translate to natural English:\n\n{text}"

    try:
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# ----------------------------------------------------------------------
# CHAT LOOP
# ----------------------------------------------------------------------
def chat():
    print("\nGovernment Service Chatbot (EN / Nepali / Romanized Nepali)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        if not question:
            continue

        answer, lang = rag_answer(question)
        print(f"Bot ({lang}): {answer}\n")

# ----------------------------------------------------------------------
# CLI ENTRY POINT (Interactive Only)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Removed argparse and sys.exit for n8n integration.
    chat()