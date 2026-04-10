"""
RAG / Knowledge Memory
======================
ScopedMemory: ChromaDB-backed memory with per-agent scope filtering.
Set MEMORY_ENABLED=false in .env to disable entirely.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path


class ScopedMemory:
    """Simple ChromaDB wrapper with scope-filtered recall."""

    def __init__(self, path: str, collection: str):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection)

    def remember(self, scope: str, content: str) -> None:
        if not content.strip():
            return
        digest = hashlib.sha256(f"{scope}|{content}".encode()).hexdigest()
        self.collection.upsert(
            ids=[digest],
            documents=[content],
            metadatas=[{"scope": scope, "ts": datetime.now(timezone.utc).isoformat()}],
        )

    def recall(self, scope: str, query: str, limit: int = 4) -> list[str]:
        try:
            result = self.collection.query(
                query_texts=[query], n_results=limit, where={"scope": scope}
            )
            docs = result.get("documents", [[]])[0]
            return [d for d in docs if d.strip()]
        except Exception:
            return []

    def ingest_directory(self, directory: str, scope: str) -> int:
        """Chunk all .md files by ## headings and upsert into ChromaDB.
        Skips re-ingestion if the directory content hash hasn't changed."""
        md_files = sorted(Path(directory).glob("*.md"))
        if not md_files:
            return 0

        # Hash all file contents to detect changes
        content_hash = hashlib.sha256()
        for f in md_files:
            content_hash.update(f.read_bytes())
        digest = content_hash.hexdigest()

        # Check if already ingested (store hash as a sentinel document)
        sentinel_id = f"_knowledge_hash_{scope}"
        existing = self.collection.get(ids=[sentinel_id])
        if existing["documents"] and existing["documents"][0] == digest:
            return 0

        # Chunk by ## headings, keeping the heading with its content
        chunks = []
        for f in md_files:
            text = f.read_text()
            sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
            for section in sections:
                section = section.strip()
                if section and len(section) > 20:
                    chunks.append(section)

        # Upsert all chunks
        for chunk in chunks:
            chunk_id = hashlib.sha256(f"{scope}|{chunk}".encode()).hexdigest()
            self.collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{"scope": scope, "source": "knowledge"}],
            )

        # Store sentinel
        self.collection.upsert(
            ids=[sentinel_id], documents=[digest],
            metadatas=[{"scope": scope, "source": "_sentinel"}],
        )
        return len(chunks)
