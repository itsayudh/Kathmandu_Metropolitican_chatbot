# adk_rag_agent.py
import os
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
import google.generativeai as genai

# ================== CONFIG ==================
BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR_EN = BASE_DIR / "chroma_db_en"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = "gemini-2.5-flash" # Use the defined model variable

# ================== EMBEDDING & VECTOR STORE ==================
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR_EN),
    embedding_function=embedder
)

# ================== LANGUAGE DETECTION (Enhanced) ==================
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

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
    return matches >= 2

def detect_language(text: str) -> str:
    if DEVANAGARI_PATTERN.search(text):
        return "nepali"
    if is_romanized_nepali(text):
        return "romanized_nepali"
    return "english"

# ================== TRANSLATION (Aggressively forced Romanized output) ==================
def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "nepali":
        prompt = f"Translate to natural Nepali (Devanagari script only):\n\n{text}"
    elif target_lang == "romanized_nepali":
        # ⭐ CRITICAL FIX: EXTREMELY STRONG INSTRUCTION FOR CORRECTION
        prompt = f"Translate the following text to natural, spoken **Romanized Nepali**. You must use the **Latin (English) alphabet** only, in the style of texting or mobile typing. **DO NOT use Devanagari script for any word.**\n\nText:\n{text}"
    else: # target_lang == "english"
        prompt = f"Translate to natural English only:\n\n{text}"

    try:
        model = genai.GenerativeModel(gemini_model)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def to_english(text: str) -> str:
    if detect_language(text) == "english":
        return text
    # Reuse translate_text for consistency
    return translate_text(text, "english")

# ================== SMART RAG TOOL (Most Robust Romanized Fix) ==================
def citizen_charter_rag(query: str) -> str:
    lang = detect_language(query)
    search_query = query if lang == "english" else to_english(query)
    
    docs = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(search_query)
    
    if not docs:
        fallbacks = {
            "nepali": "माफ गर्नुहोस्, जानकारी भेटिएन।",
            "romanized_nepali": "Maaf garnuhos, jaankari bhetiyena.",
            "english": "No information found."
        }
        return fallbacks.get(lang, "Error: Unknown language.")
    
    context = "\n\n".join([d.page_content for d in docs])
    
    # ⭐ DYNAMIC PROMPT: Inject aggressive system instruction when Romanized is needed
    if lang == "romanized_nepali":
        system_instruction = "You are a government chatbot who MUST reply in natural **Romanized Nepali** (using the Latin alphabet). NEVER use Devanagari script. Use texting/mobile style."
        instruction_line = f"Question: {query}\n\nContext:\n{context}\n\nAnswer concisely in Romanized Nepali only (Latin Script)."
    elif lang == "nepali":
        system_instruction = "तपाईं सरकारी सेवा च्याटबट हुनुहुन्छ।"
        instruction_line = f"प्रश्न: {query}\n\nसन्दर्भ:\n{context}\n\nजवाफ छोटो र नेपालीमा दिनुहोस्।"
    else:
        system_instruction = "You are a government service chatbot."
        instruction_line = f"Question: {query}\n\nContext:\n{context}\n\nAnswer concisely in English."

    prompt = f"{system_instruction}\n\n{instruction_line}"
    
    try:
        model = genai.GenerativeModel(gemini_model)
        resp = model.generate_content(prompt)
        answer = resp.text.strip()

        # === FINAL LANGUAGE CORRECTION (The Guardrail) ===
        if lang == "romanized_nepali" and DEVANAGARI_PATTERN.search(answer):
            # If the response is in Devanagari, but should be Romanized, correct it.
            answer = translate_text(answer, "romanized_nepali")
        elif lang == "nepali" and not DEVANAGARI_PATTERN.search(answer):
            answer = translate_text(answer, "nepali")
        elif lang == "english" and DEVANAGARI_PATTERN.search(answer):
            answer = translate_text(answer, "english")
            
        return answer
    except Exception as e:
        # Fallback to the original language
        fallbacks = {
            "nepali": "माफ गर्नुहोस्, जवाफ दिने क्रममा समस्या भयो।",
            "romanized_nepali": "Maaf garnuhos, jawaf dinna sakina.",
            "english": "Sorry, there was an error generating the response."
        }
        return fallbacks.get(lang, f"Error: {str(e)}")

# ================== ADK AGENT ==================
rag_tool = FunctionTool(citizen_charter_rag)
rag_tool.name = "citizen_charter_rag"
rag_tool.description = "Search Nepal government services. Works in Nepali, Romanized, English. The tool's output is the final answer and must not be re-processed."

root_agent = LlmAgent(
    name="DCC_Assistant",
    model=gemini_model,
    # ⭐ CRITICAL ADK AGENT INSTRUCTION CHANGE: Tell the agent to trust the tool's output implicitly
    instruction="""You are Nepal's official Digital Citizen Charter AI.
For ANY service question, ALWAYS use citizen_charter_rag tool.
The output of the tool is the final, ready-to-use answer, which must be returned directly without any modification, re-translation, or re-formatting.
Reply in user's exact language style.""",
    tools=[rag_tool]
)

print("AGENT READY! Run: adk web adk_rag_agent.py")