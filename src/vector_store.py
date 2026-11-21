"""
Vector Store module for SkillBridge EdTech Policy Documents (Cell 6 logic).
Handles chunking, Google Gemini embeddings, FAISS indexing, persistence, and semantic retrieval.
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import (
    DATA_DIR,
    POLICIES_DIR,
    VECTOR_STORE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_google_api_key,
    get_gemini_client,
)

logger = logging.getLogger("rag_vector_store")

# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu is not installed. Falling back to NumPy cosine similarity.")


class DocumentChunk:
    """Represents a chunk of a policy document with metadata."""
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source_file: str,
        section_title: str = "General",
        char_start: int = 0,
        char_end: int = 0,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source_file = source_file
        self.section_title = section_title
        self.char_start = char_start
        self.char_end = char_end

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "section_title": self.section_title,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source_file=data["source_file"],
            section_title=data.get("section_title", "General"),
            char_start=data.get("char_start", 0),
            char_end=data.get("char_end", 0),
        )


class PolicyDocumentChunker:
    """
    Intelligent chunker for policy documents.
    Preserves section titles, formulas, tables, and numbered policy rules.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_markdown(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Splits markdown by headers (##, ###) while respecting chunk_size limits.
        """
        chunks: List[DocumentChunk] = []
        # Split by level-2 and level-3 markdown headers
        header_pattern = re.compile(r"^(#{2,3}\s+.*)$", re.MULTILINE)
        sections = []
        last_pos = 0
        current_title = "Introduction"

        for match in header_pattern.finditer(text):
            start = match.start()
            if start > last_pos:
                section_content = text[last_pos:start].strip()
                if section_content:
                    sections.append((current_title, section_content, last_pos, start))
            current_title = match.group(1).lstrip("#").strip()
            last_pos = start

        if last_pos < len(text):
            section_content = text[last_pos:].strip()
            if section_content:
                sections.append((current_title, section_content, last_pos, len(text)))

        chunk_counter = 0
        for title, content, sec_start, sec_end in sections:
            # If section fits within chunk_size, keep whole
            if len(content) <= self.chunk_size:
                chunk_id = f"{Path(source_file).stem}_c{chunk_counter:03d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=f"### {title}\n{content}",
                        source_file=source_file,
                        section_title=title,
                        char_start=sec_start,
                        char_end=sec_end,
                    )
                )
                chunk_counter += 1
            else:
                # Sliding window chunking with paragraph preservation
                paragraphs = content.split("\n\n")
                current_chunk = []
                current_len = 0
                for p in paragraphs:
                    p = p.strip()
                    if not p:
                        continue
                    if current_len + len(p) > self.chunk_size and current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunk_id = f"{Path(source_file).stem}_c{chunk_counter:03d}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                text=f"### {title}\n{chunk_text}",
                                source_file=source_file,
                                section_title=title,
                                char_start=sec_start,
                                char_end=sec_end,
                            )
                        )
                        chunk_counter += 1
                        # Keep overlap paragraph if small
                        overlap_p = current_chunk[-1] if len(current_chunk[-1]) < self.chunk_overlap else ""
                        current_chunk = [overlap_p, p] if overlap_p else [p]
                        current_len = sum(len(x) for x in current_chunk)
                    else:
                        current_chunk.append(p)
                        current_len += len(p)

                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_id = f"{Path(source_file).stem}_c{chunk_counter:03d}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=f"### {title}\n{chunk_text}",
                            source_file=source_file,
                            section_title=title,
                            char_start=sec_start,
                            char_end=sec_end,
                        )
                    )
                    chunk_counter += 1

        return chunks

    def chunk_plain_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Simple character-based sliding window chunking."""
        chunks: List[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        chunk_counter = 0

        for i in range(0, len(text), step):
            sub_text = text[i : i + self.chunk_size].strip()
            if sub_text:
                chunk_id = f"{Path(source_file).stem}_c{chunk_counter:03d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=sub_text,
                        source_file=source_file,
                        section_title="General",
                        char_start=i,
                        char_end=min(i + self.chunk_size, len(text)),
                    )
                )
                chunk_counter += 1
        return chunks

    def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """Reads file and applies appropriate chunking."""
        logger.info(f"Processing policy file: {file_path.name}")
        suffix = file_path.suffix.lower()

        if suffix in [".md", ".markdown"]:
            text = file_path.read_text(encoding="utf-8")
            return self.chunk_markdown(text, file_path.name)
        elif suffix in [".txt"]:
            text = file_path.read_text(encoding="utf-8")
            return self.chunk_markdown(text, file_path.name) if "##" in text else self.chunk_plain_text(text, file_path.name)
        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n\n".join(pages)
                return self.chunk_plain_text(full_text, file_path.name)
            except Exception as e:
                logger.error(f"Failed to read PDF {file_path.name}: {e}")
                return []
        else:
            logger.warning(f"Unsupported file format {suffix} for {file_path.name}")
            return []


class GeminiEmbedder:
    """
    Computes text embeddings using Google Gemini (text-embedding-004)
    via official google-genai SDK, with deterministic fallback for offline dev.
    """
    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL):
        self.model = model
        self.client = None
        self.api_key = get_google_api_key()

        if self.api_key:
            try:
                self.client = get_gemini_client()
                logger.info(f"Gemini client initialized with model '{self.model}'.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client: {e}. Falling back to mock embeddings.")
        else:
            logger.warning("No GOOGLE_API_KEY found. Running in offline/mock embedding mode.")

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Embeds a list of strings, returning a float32 numpy array of shape (N, dim).
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        if self.client is not None:
            try:
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                    )
                    # response.embeddings is a list of ContentEmbedding objects
                    for emb in response.embeddings:
                        all_embeddings.append(emb.values)

                embeddings_arr = np.array(all_embeddings, dtype=np.float32)
                # L2 Normalize vectors for cosine similarity via dot product
                norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                return embeddings_arr / norms
            except Exception as e:
                logger.error(f"Gemini API embed call failed: {e}. Using deterministic fallback.")

        # Deterministic offline fallback embedding (768-dim pseudo-embeddings for testing)
        dim = 768
        offline_embeddings = []
        for text in texts:
            # Deterministic hash-based representation
            np.random.seed(abs(hash(text)) % (2**32))
            vec = np.random.randn(dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            offline_embeddings.append(vec / (norm if norm > 0 else 1.0))
        return np.array(offline_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string, returning shape (1, dim)."""
        return self.embed_texts([query])


class FAISSPolicyVectorStore:
    """
    FAISS-based vector store for policy documents.
    Supports index building, persistent storage, and cosine similarity search.
    """
    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        storage_dir: Path = VECTOR_STORE_DIR,
    ):
        self.embedder = embedder or GeminiEmbedder()
        self.storage_dir = Path(storage_dir)
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dim: int = 768

    @property
    def index_path(self) -> Path:
        return self.storage_dir / "policy_faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.storage_dir / "policy_metadata.json"

    def build_from_directory(self, policies_dir: Path = POLICIES_DIR) -> int:
        """
        Scans policies directory, chunks all documents, computes embeddings,
        and constructs the FAISS index.
        """
        policies_dir = Path(policies_dir)
        if not policies_dir.exists():
            logger.warning(f"Policies dir {policies_dir} does not exist.")
            return 0

        chunker = PolicyDocumentChunker()
        all_chunks: List[DocumentChunk] = []

        for file_path in sorted(policies_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in [".md", ".txt", ".pdf"]:
                chunks = chunker.process_file(file_path)
                all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning(f"No policy documents found in {policies_dir}.")
            return 0

        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.dim = embeddings.shape[1]

        if FAISS_AVAILABLE:
            # Using Inner Product (cosine similarity since vectors are L2-normalized)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
        else:
            self._raw_embeddings = embeddings

        self.chunks = all_chunks
        self.save()
        logger.info(f"FAISS index built successfully with {len(self.chunks)} chunks.")
        return len(self.chunks)

    def save(self):
        """Saves FAISS index and chunk metadata to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

        meta_list = [c.to_dict() for c in self.chunks]
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved vector store to {self.storage_dir}")

    def load(self) -> bool:
        """Loads FAISS index and metadata from disk if available."""
        if not self.metadata_path.exists():
            return False

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        self.chunks = [DocumentChunk.from_dict(d) for d in meta_list]

        if FAISS_AVAILABLE and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dim = self.index.d
            logger.info(f"Loaded FAISS index from {self.index_path} with {len(self.chunks)} items.")
            return True
        elif not FAISS_AVAILABLE:
            logger.info(f"Loaded {len(self.chunks)} chunk metadata records.")
            return True
        return False

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        Performs semantic search to find top_k most relevant policy chunks.
        """
        if not self.chunks:
            # Attempt to load or build
            if not self.load():
                self.build_from_directory()

        if not self.chunks:
            return []

        query_emb = self.embedder.embed_query(query)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "section_title": chunk.section_title,
                    "source_file": chunk.source_file,
                    "score": float(score),
                    "text": chunk.text,
                })
            return results
        else:
            # Fallback keyword + simple cosine similarity
            # Simple keyword overlap scoring
            q_words = set(query.lower().split())
            scored = []
            for c in self.chunks:
                c_words = set(c.text.lower().split())
                overlap = len(q_words.intersection(c_words))
                scored.append((overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "chunk_id": c.chunk_id,
                    "section_title": c.section_title,
                    "source_file": c.source_file,
                    "score": float(score),
                    "text": c.text,
                }
                for score, c in scored[:top_k]
            ]


# Singleton instance helper
_global_vector_store: Optional[FAISSPolicyVectorStore] = None


def get_vector_store() -> FAISSPolicyVectorStore:
    """Returns singleton instance of the policy vector store."""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = FAISSPolicyVectorStore()
        if not _global_vector_store.load():
            _global_vector_store.build_from_directory()
    return _global_vector_store
