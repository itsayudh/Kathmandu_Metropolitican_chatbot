# offline_chat.py → Nepal's Sovereign Terminal AI (Phi-3.5-mini)

import os
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === ADD THESE LINES HERE ===
# 1. Define a writable cache path within your project directory
writable_cache_path = os.path.join(
    os.path.expanduser("~"), 
    "My Folder/Ninjainfosys/projects/chatbot_version_english", 
    ".hf_custom_cache"
)

# 2. Tell the Hugging Face libraries to use this path
os.environ['HF_HOME'] = writable_cache_path

# ============================================================
# LOAD YOUR PHI-3.5 MODEL (LOCAL, OFFLINE, CPU-ONLY)
# ============================================================
MODEL_PATH = "/home/ayudh/models/phi-3.5-mini-instruct"  # NEW PATH
print("Loading Nepal's brain: Phi-3.5-mini-instruct (3.8B)")

# Ensure model path exists
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

# Load model on CPU only
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float32,  # CPU-friendly precision
    device_map=None,            # Force CPU
    trust_remote_code=False
)
model.eval()

# ============================================================
# TEXT GENERATION FUNCTION
# ============================================================
def phi_generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()

# ============================================================
# VECTOR DB (English Knowledge Base)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
vectorstore = Chroma(
    persist_directory=str(BASE_DIR / "chroma_db_en"),
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# ============================================================
# LANGUAGE DETECTION (English / Nepali / Romanized Nepali)
# ============================================================
DEV = re.compile(r'[\u0900-\u097F]')
ROMAN_WORDS = {
    "ma","ho","chha","lai","ko","pani","nagarikta",
    "darta","janma","shulka","kati","din","rs","garna","kasari"
}

def detect_lang(q: str) -> str:
    if DEV.search(q): 
        return "nepali"
    words = re.findall(r'\w+', q.lower())
    if sum(w in ROMAN_WORDS for w in words) >= 2: 
        return "romanized"
    return "english"

# ============================================================
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ============================================================
def answer_question(q: str) -> str:
    lang = detect_lang(q)
    docs = vectorstore.as_retriever(k=8).invoke(q)
    context = "\n\n".join([d.page_content for d in docs]) if docs else "No info found."

    if lang == "romanized":
        prompt = f"""You are a Nepali government helper.
Reply ONLY in Romanized Nepali (phone typing style).
NEVER use Devanagari script.
Examples:
- janma darta → not जन्म दर्ता
- Rs 50 → not रु. ५०

Question: {q}
Context: {context}
Answer:"""
    elif lang == "nepali":
        prompt = f"""नेपालीमा छोटो र स्पष्ट जवाफ दिनुहोस्।
प्रश्न: {q}
सन्दर्भ: {context}
जवाफ:"""
    else:
        prompt = f"""Answer in natural English.
Question: {q}
Context: {context}
Answer:"""

    return phi_generate(prompt)

# ============================================================
# TERMINAL CHAT LOOP
# ============================================================
def chat():
    print("\n" + "="*60)
    print("   NEPAL'S SOVEREIGN OFFLINE AI (Phi-3.5-mini)")
    print("   Supports: English • नेपाली • Romanized (janma darta)")
    print("   Type 'exit' or 'quit' to stop")
    print("="*60 + "\n")

    while True:
        try:
            q = input("You: ").strip()
            if q.lower() in {"exit", "quit", "bye", "तयार", "धन्यवाद"}:
                print("धन्यवाद! नेपालको AI सधैं तपाईंको सेवामा छ 🇳🇵")
                break
            if not q:
                continue

            print("Bot: Thinking...", end="\r")
            response = answer_question(q)
            print(f"Bot: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Nepal Zindabad 🇳🇵")
            break
        except Exception as e:
            print(f"Error: {e}")

# ============================================================
# MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    chat()
