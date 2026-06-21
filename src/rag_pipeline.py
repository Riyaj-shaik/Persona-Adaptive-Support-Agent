"""
rag_pipeline.py - Handles document ingestion (parsing, chunking, embedding)
and semantic retrieval via ChromaDB vector store.
"""

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import chromadb
from pypdf import PdfReader

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    SUPPORTED_EXTENSIONS,
)


class LocalRAGPipeline:

    def __init__(self, db_dir: str = CHROMA_DB_DIR):
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def is_indexed(self) -> bool:
        """Check if the vector DB already has documents."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def get_embedding(self, text: str) -> list:
        """Convert a text string into a dense vector embedding."""
        response = self.genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )
        if hasattr(response, 'embeddings'):
            return response.embeddings[0].values
        elif hasattr(response, 'embedding'):
            return response.embedding.values
        else:
            raise ValueError(f"Unexpected embedding response format: {response}")

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

        for idx, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            chunk_id = f"{doc_name}_chunk_{idx}"
            self.collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"source": doc_name, "chunk_index": idx}],
                documents=[chunk]
            )

        return len(chunks)

    def ingest_all_documents(self, data_dir: str = DATA_DIR) -> dict:
        summary = {}
        data_path = Path(data_dir)

        for filepath in data_path.iterdir():
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
        query_vector = self.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count())
        )

        retrieved_items = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                similarity = max(0.0, 1.0 - distance)
                retrieved_items.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "score": round(similarity, 4)
                })

        return retrieved_items