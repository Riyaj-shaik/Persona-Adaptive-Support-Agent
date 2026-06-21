"""
rag_pipeline.py - FAISS-based vector search.
"""

import os
import json
import numpy as np
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import faiss
from pypdf import PdfReader

from src.config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, CHUNK_SIZE,
    CHUNK_OVERLAP, TOP_K_RESULTS, CHROMA_DB_DIR,
    DATA_DIR, SUPPORTED_EXTENSIONS,
)

INDEX_FILE = os.path.join(CHROMA_DB_DIR, "faiss.index")
META_FILE  = os.path.join(CHROMA_DB_DIR, "metadata.json")


class LocalRAGPipeline:

    def __init__(self, db_dir: str = CHROMA_DB_DIR):
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self.index = None
        self.documents = []
        self._load_index()

    def _load_index(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

    def _save_index(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def is_indexed(self) -> bool:
        return self.index is not None and len(self.documents) > 0

    def get_embedding(self, text: str) -> list:
        response = self.genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )
        if hasattr(response, 'embeddings'):
            return response.embeddings[0].values
        elif hasattr(response, 'embedding'):
            return response.embedding.values
        else:
            raise ValueError(f"Unexpected embedding response: {response}")

    def parse_txt_or_md(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def parse_pdf(self, filepath: str) -> str:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def parse_document(self, filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        if ext in (".txt", ".md"):
            return self.parse_txt_or_md(filepath)
        elif ext == ".pdf":
            return self.parse_pdf(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def ingest_document(self, doc_name: str, content: str) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(content)

        embeddings = []
        for chunk in chunks:
            emb = self.get_embedding(chunk)
            embeddings.append(emb)
            self.documents.append({"text": chunk, "source": doc_name})

        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vectors)
        self._save_index()
        return len(chunks)

    def ingest_all_documents(self, data_dir: str = DATA_DIR) -> dict:
        summary = {}
        for filepath in Path(data_dir).iterdir():
            if filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    content = self.parse_document(str(filepath))
                    count = self.ingest_document(filepath.name, content)
                    summary[filepath.name] = count
                    print(f"  ✓ Ingested '{filepath.name}' → {count} chunks")
                except Exception as e:
                    print(f"  ✗ Failed to ingest '{filepath.name}': {e}")
        return summary

    def retrieve_context(self, query: str, top_k: int = TOP_K_RESULTS) -> list:
        if not self.is_indexed():
            return []

        query_vector = np.array([self.get_embedding(query)], dtype="float32")
        faiss.normalize_L2(query_vector)

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_vector, k)

        retrieved_items = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            retrieved_items.append({
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "score": round(float(score), 4)
            })

        return retrieved_items