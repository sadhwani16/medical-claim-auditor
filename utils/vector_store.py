"""FAISS-based vector store for PM-JAY rules (works on Databricks Free Edition via DBFS)."""

import os
import pickle
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_INDEX_FILENAME = "pmjay_faiss_index"

_store: Optional[FAISS] = None
_embeddings = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_index(rules_text: str, save_path: str) -> FAISS:
    """Chunk PM-JAY rules text, embed, build FAISS index, and save to disk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "—", "-"],
    )
    chunks = splitter.split_text(rules_text)
    print(f"[VectorStore] Built index with {len(chunks)} rule chunks.")

    store = FAISS.from_texts(chunks, _get_embeddings())
    _save_index(store, save_path)
    global _store
    _store = store
    return store


def load_index(index_path: str) -> Optional[FAISS]:
    """Load a previously built FAISS index from disk."""
    global _store
    index_dir = Path(index_path)
    if not index_dir.exists():
        return None
    try:
        _store = FAISS.load_local(
            str(index_dir / _INDEX_FILENAME),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return _store
    except Exception as e:
        print(f"[VectorStore] Failed to load index: {e}")
        return None


def retrieve_relevant_rules(query: str, k: int = 5) -> list[str]:
    """Return top-k rule snippets relevant to the claim query."""
    store = _store or load_index(_default_path())
    if store is None:
        return ["[Vector store not initialised — run notebook 03 first]"]
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def _save_index(store: FAISS, save_path: str):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    store.save_local(str(Path(save_path) / _INDEX_FILENAME))


def _default_path() -> str:
    from config import VECTOR_STORE_PATH
    return VECTOR_STORE_PATH
