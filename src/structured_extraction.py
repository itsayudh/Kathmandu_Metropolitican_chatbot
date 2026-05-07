# src/structured_extraction.py
import os
import json
import re
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from data_ingestion import load_pdfs_advanced, extract_text_and_tables_pdfplumber  # assuming you've added the missing import from data_ingestion

# ----------------------------------------------------------------------
# Gemini configuration
# ----------------------------------------------------------------------
genai.configure(api_key="AIzaSyDoyZY6FgEWQRv_ACJ3aKU_ZI9FO6_IjLo")
model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------------------------------------------------
# Pydantic models (unchanged – field names are already English)
# ----------------------------------------------------------------------
class ServiceRequirement(BaseModel):
    requirement: str                     # e.g. "Online form (public.donidcr.gov.np)"

class CitizenService(BaseModel):
    service_number: str                  # e.g. "1"
    service_name: str                    # e.g. "Birth Registration"
    required_documents: List[ServiceRequirement]
    fee: str                             # e.g. "Rs. 50"
    processing_time: str                 # e.g. "7 working days"
    responsible_person: str              # e.g. "Ward Secretary"

class ServiceCatalog(BaseModel):
    services: List[CitizenService]

# ----------------------------------------------------------------------
# Helper: force English output
# ----------------------------------------------------------------------
def _force_english(text: str) -> str:
    """Strip any Nepali Unicode characters that might have leaked through."""
    return re.sub(r'[\u0900-\u097F]+', '', text).strip()

# ----------------------------------------------------------------------
# Gemini extraction – English only
# ----------------------------------------------------------------------
def extract_services_with_gemini(cleaned_text: str) -> ServiceCatalog:
    """
    Ask Gemini to return the exact JSON schema **in English**.
    """
    prompt = f"""
You are processing Nepali Citizen Charter PDFs for a government-services chatbot.

Translate **every** piece of information into **English** and return **only** valid JSON that exactly matches this structure:

```json
{{
  "services": [
    {{
      "service_number": "1",
      "service_name": "Birth Registration",
      "required_documents": [
        {{"requirement": "Online form (public.donidcr.gov.np)"}}
      ],
      "fee": "Rs. 50",
      "processing_time": "7 working days",
      "responsible_person": "Ward Secretary"
    }}
  ]
}}
{cleaned_text[:15000]}
"""
    response = model.generate_content(prompt)
    try:
        json_text = response.text.strip()
        # Strip possible markdown fences
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        data = json.loads(json_text)
        catalog = ServiceCatalog(**data)

        # ---- Post-process to guarantee English ----
        for svc in catalog.services:
            svc.service_number = _force_english(svc.service_number)
            svc.service_name   = _force_english(svc.service_name)
            svc.fee            = _force_english(svc.fee)
            svc.processing_time = _force_english(svc.processing_time)
            svc.responsible_person = _force_english(svc.responsible_person)
            for doc in svc.required_documents:
                doc.requirement = _force_english(doc.requirement)

        return catalog

    except Exception as e:
        print(f"Gemini JSON parsing error: {e}")
        return ServiceCatalog(services=[])

# ----------------------------------------------------------------------
# Text cleaning (unchanged – still useful for LLM)
# ----------------------------------------------------------------------
def clean_extracted_text_for_llm(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    text = text.replace('', '')                   # bad chars
    text = re.sub(r'--- Page \d+ ---', '', text)   # page markers
    text = re.sub(r'Page \d+', '', text)
    return text.strip()

# ----------------------------------------------------------------------
# Pipeline per PDF
# ----------------------------------------------------------------------
def process_pdf_to_structured_data(pdf_path: str, output_path: str):
    print(f"Processing: {os.path.basename(pdf_path)}")

    # 1. Extract raw text (you already have a robust extractor)
    raw = extract_text_and_tables_pdfplumber(pdf_path)

    # 2. Clean
    cleaned = clean_extracted_text_for_llm(raw['text'])
    if len(cleaned) < 100:
        print("Insufficient text after cleaning.")
        return

    # 3. Gemini → English JSON
    print("Calling Gemini for English extraction...")
    catalog = extract_services_with_gemini(cleaned)

    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(catalog.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Saved {len(catalog.services)} English services → {output_path}")

    # Preview
    for s in catalog.services[:2]:
        print(f"   • {s.service_name} ({s.service_number})")

# ----------------------------------------------------------------------
# Batch entry point
# ----------------------------------------------------------------------
def main():
    input_dir  = os.path.join("data", "raw")
    output_dir = os.path.join("data", "structured")
    os.makedirs(output_dir, exist_ok=True)

    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdfs:
        print("No PDFs in data/raw/")
        return

    for pdf in pdfs:
        src = os.path.join(input_dir, pdf)
        dst = os.path.join(output_dir, pdf.replace(".pdf", "_services_en.json"))
        process_pdf_to_structured_data(src, dst)

if __name__ == "__main__":
    main()