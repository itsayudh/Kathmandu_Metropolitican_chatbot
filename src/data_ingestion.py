
import os
from typing import List, Dict
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import re

def extract_text_and_tables_pdfplumber(pdf_path: str) -> Dict:
    """
    Extract both text and tables from PDF using pdfplumber.
    Returns a dictionary with text and structured table data.
    """
    result = {
        'text': '',
        'tables': [],
        'metadata': {}
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            result['metadata']['pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    result['text'] += f"--- Page {page_num+1} ---\n{text}\n\n"
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_num, table in enumerate(tables):
                        if table and any(any(cell for cell in row) for row in table):
                            result['tables'].append({
                                'page': page_num + 1,
                                'table_number': table_num + 1,
                                'data': table
                            })
                            
    except Exception as e:
        print(f"Error with pdfplumber on {pdf_path}: {e}")
    
    return result

def extract_text_ocr(pdf_path: str) -> str:
    """Extract text from scanned PDF using OCR with better error handling."""
    text = ""
    try:
        print(f"  -> Running OCR on {os.path.basename(pdf_path)}...")
        images = convert_from_path(pdf_path)
        
        for i, img in enumerate(images, start=1):
            # Try Nepali first, then fallback to English
            try:
                text_part = pytesseract.image_to_string(img, lang="nep+eng")
            except:
                text_part = pytesseract.image_to_string(img, lang="eng")
            
            text += f"--- Page {i} ---\n{text_part}\n\n"
            print(f"    OCR processed page {i}/{len(images)}")
            
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}")
        # Try alternative approach for Windows or different poppler setup
        try:
            from pdf2image import convert_from_bytes
            import requests
            
            # Alternative approach if file path issues
            with open(pdf_path, 'rb') as f:
                images = convert_from_bytes(f.read())
                for img in images:
                    text += pytesseract.image_to_string(img, lang="nep+eng") + "\n"
        except Exception as e2:
            print(f"Alternative OCR also failed: {e2}")
    
    return text.strip()

def is_meaningful_text(text: str, threshold: float = 0.1) -> bool:
    """
    Check if text contains meaningful Nepali/English content.
    """
    if not text or len(text.strip()) < 50:
        return False
    
    # Count Nepali Unicode characters
    nepali_chars = re.findall(r"[\u0900-\u097F]", text)
    nepali_ratio = len(nepali_chars) / max(len(text), 1)
    
    # Count English words
    english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    english_ratio = len(english_words) / max(len(text.split()), 1)
    
    return nepali_ratio > threshold or english_ratio > 0.05

def analyze_pdf_structure(pdf_path: str) -> str:
    """
    Analyze PDF and return its type for appropriate extraction method.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ""
            
            # Check if it's digital text PDF
            if text and len(text.strip()) > 100:
                return "digital_text"
            
            # Check if it has images (likely scanned)
            if first_page.images:
                return "scanned"
            
            return "unknown"
            
    except Exception as e:
        return f"error: {e}"

def load_pdfs_advanced(directory_path: str) -> List[Dict]:
    """
    Advanced PDF loading with structure analysis and appropriate extraction.
    """
    pdf_data = []

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory_path}")
        return pdf_data

    print(f"Found {len(pdf_files)} PDF file(s). Analyzing...")

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        print(f"\n=== Processing: {pdf_file} ===")
        
        # Analyze PDF type
        pdf_type = analyze_pdf_structure(file_path)
        print(f"  PDF Type: {pdf_type}")
        
        file_data = {
            'filename': pdf_file,
            'type': pdf_type,
            'text': '',
            'tables': [],
            'success': False
        }
        
        try:
            if "digital" in pdf_type:
                # Use pdfplumber for digital PDFs
                result = extract_text_and_tables_pdfplumber(file_path)
                file_data['text'] = result['text']
                file_data['tables'] = result['tables']
                file_data['success'] = bool(result['text'].strip())
                
            elif "scanned" in pdf_type or "error" in pdf_type:
                # Use OCR for scanned PDFs
                text = extract_text_ocr(file_path)
                file_data['text'] = text
                file_data['success'] = bool(text.strip())
                
            else:
                # Try both methods
                result = extract_text_and_tables_pdfplumber(file_path)
                if not result['text'] or not is_meaningful_text(result['text']):
                    text = extract_text_ocr(file_path)
                    file_data['text'] = text
                else:
                    file_data['text'] = result['text']
                    file_data['tables'] = result['tables']
                
                file_data['success'] = bool(file_data['text'].strip())
                
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")
            file_data['success'] = False
        
        pdf_data.append(file_data)
        
        if file_data['success']:
            print(f"  ✅ Successfully extracted content")
        else:
            print(f"  ❌ Failed to extract meaningful content")
    
    return pdf_data

def save_extracted_data(pdf_data: List[Dict], output_dir: str):
    """Save extracted data to files for analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    for data in pdf_data:
        if data['success']:
            filename_base = os.path.splitext(data['filename'])[0]
            
            # Save text
            text_path = os.path.join(output_dir, f"{filename_base}_text.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(data['text'])
            
            # Save tables info
            if data['tables']:
                tables_path = os.path.join(output_dir, f"{filename_base}_tables.txt")
                with open(tables_path, 'w', encoding='utf-8') as f:
                    for table in data['tables']:
                        f.write(f"Table from page {table['page']}:\n")
                        for row in table['data']:
                            f.write(" | ".join([str(cell) if cell else "" for cell in row]) + "\n")
                        f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sample_data_path = os.path.join(BASE_DIR, "../data/raw")
    output_path = os.path.join(BASE_DIR, "../data/processed")
    
    sample_data_path = os.path.normpath(sample_data_path)
    output_path = os.path.normpath(output_path)

    print(f"Input directory: {os.path.abspath(sample_data_path)}")
    print(f"Output directory: {os.path.abspath(output_path)}")

    pdf_data = load_pdfs_advanced(sample_data_path)
    
    successful = sum(1 for data in pdf_data if data['success'])
    print(f"\n=== SUMMARY ===")
    print(f"Successfully processed: {successful}/{len(pdf_data)} PDFs")
    
    # Save results
    save_extracted_data(pdf_data, output_path)
    print(f"Extracted data saved to: {output_path}")
    
    # Show samples
    for data in pdf_data:
        if data['success']:
            print(f"\n--- {data['filename']} (first 300 chars) ---")
            print(data['text'][:300])