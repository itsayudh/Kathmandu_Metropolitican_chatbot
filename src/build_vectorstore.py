# src/build_vectorstore.py
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR        = Path(__file__).resolve().parent.parent
STRUCTURED_DIR  = BASE_DIR / "data" / "structured"
PERSIST_DIR     = BASE_DIR / "chroma_db_en"

print(f"Project root   : {BASE_DIR}")
print(f"JSON folder    : {STRUCTURED_DIR.resolve()}")
print(f"Chroma folder  : {PERSIST_DIR.resolve()}")

def load_structured_jsons(structured_dir: Path) -> List[Dict[str, Any]]:
    if not structured_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {structured_dir}")

    json_files = list(structured_dir.glob("*_services_en.json"))
    print(f"Found {len(json_files)} *_services_en.json files")
    for f in json_files:
        print("  -", f.name)

    services: List[Dict[str, Any]] = []
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                batch = data.get("services", [])
                print(f"    → {len(batch)} services from {json_path.name}")
                for s in batch:
                    s["_source_file"] = json_path.name
                services.extend(batch)
        except Exception as e:
            print(f"    [ERROR] reading {json_path.name}: {e}")

    print(f"Total services loaded: {len(services)}")
    return services

def create_service_text_block(srv: Dict[str, Any]) -> str:
    lines = [
        f"Service Number: {srv.get('service_number', '')}",
        f"Service Name: {srv.get('service_name', '')}",
        "Required Documents:"
    ]
    for doc in srv.get("required_documents", []):
        lines.append(f"  • {doc.get('requirement', '')}")
    lines += [
        f"Fee: {srv.get('fee', '')}",
        f"Processing Time: {srv.get('processing_time', '')}",
        f"Responsible Person: {srv.get('responsible_person', '')}"
    ]
    return "\n".join(line.strip() for line in lines if line.strip())

def services_to_documents(services: List[Dict[str, Any]]) -> List[Document]:
    raw_texts = []
    metadatas = []

    for srv in services:
        content = create_service_text_block(srv)
        if len(content) < 20:
            continue
        raw_texts.append(content)
        metadatas.append({
            "service_number": srv.get("service_number", ""),
            "service_name": srv.get("service_name", ""),
            "source_file": srv.get("_source_file", "")
        })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", "  • ", " "]
    )

    docs: List[Document] = []
    for text, meta in zip(raw_texts, metadatas):
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk.strip(),
                metadata={**meta, "chunk_index": i, "total_chunks": len(chunks)}
            ))

    print(f"Created {len(docs)} chunks from {len(services)} services (avg {len(docs)/max(len(services),1):.2f} per service)")
    return docs

if __name__ == "__main__":
    print("\nInitializing all-MiniLM-L6-v2 embedding model...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    services = load_structured_jsons(STRUCTURED_DIR)
    if not services:
        raise ValueError("No English services found – check *_services_en.json files")

    docs = services_to_documents(services)
    if not docs:
        raise ValueError("No valid chunks created")

    if PERSIST_DIR.exists():
        print(f"\nRemoving old DB at {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    print(f"\nEmbedding {len(docs)} chunks into Chroma...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory=str(PERSIST_DIR)
    )

    print("\n" + "="*60)
    print("        ENGLISH VECTOR STORE BUILD COMPLETE")
    print("="*60)
    print(f"  JSON files processed : {len(list(STRUCTURED_DIR.glob('*_services_en.json')))}")
    print(f"  Services loaded      : {len(services)}")
    print(f"  Chunks created       : {len(docs)}")
    print(f"  Vector DB path       : {PERSIST_DIR}")
    print(f"  Embedding model      : all-MiniLM-L6-v2")
    print("="*60)
    print("Next: python src/rag_chat.py")
    print("="*60)