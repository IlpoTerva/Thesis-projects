"""
Simple vector index framework for initial testing and development. This is a basic implementation that allows you to create a vector index from a set of documents and perform similarity searches.
"""

from pathlib import Path
from typing import Iterable, Optional, Any
from app.config import settings

EMBED_MODEL = settings.EMBEDDING_MODEL
CHROMA_PATH = settings.VECTOR_STORE_DIR
COLLECTION  = settings.VECTOR_COLLECTION

_model = None

RAW_DIR = Path(settings.VECTOR_STORE_DIR).resolve().parent / "raw"

# e5 prefix conventions — keep in sync with analytics/document_search.py
PASSAGE_PREFIX = "passage: "

# Approx token budget per chunk (e5 char heuristic: ~4 chars/token)
CHUNK_CHARS = 1600          # ~400 tokens
CHUNK_OVERLAP_CHARS = 240   # ~60 tokens
MIN_CHUNK_CHARS = 200       # drop fragments shorter than ~50 tokens

INDEXABLE_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".xlsx"}

SKIP_FILES = {"sample_FAQs.txt"} #NOTE in the base code not in this one


def _parse_pdf(path: Path) -> Iterable[tuple[str, Optional[int], Optional[str]]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield text, i, None

PARSERS = {
    ".pdf": _parse_pdf,
}

def _chunk(text: str) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= CHUNK_CHARS:
        return [text] if len(text) >= MIN_CHUNK_CHARS else ([text] if text else [])

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        # try to break on a sentence/space boundary near the end
        if end < n:
            window = text.rfind(" ", start + MIN_CHUNK_CHARS, end)
            if window != -1:
                end = window
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS or (chunk and not chunks):
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks



def build() -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not RAW_DIR.exists():
        raise SystemExit(f"Raw KB directory not found: {RAW_DIR}")

    files = sorted(
        p
        for p in RAW_DIR.iterdir()
        if p.suffix.lower() in INDEXABLE_SUFFIXES and p.name not in SKIP_FILES
    )
    if not files:
        raise SystemExit(f"No indexable files in {RAW_DIR} (looked for {sorted(INDEXABLE_SUFFIXES)})")

    print(f"Loading embedding model: {settings.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}

    for path in files:
        parser = PARSERS[path.suffix.lower()]
        chunk_index = 0
        try:
            records = list(parser(path))
        except Exception as exc:
            print(f"  ! skipping {path.name}: {exc}")
            continue
        for text, page, sheet in records:
            for chunk in _chunk(text):
                ids.append(f"{path.name}::{chunk_index}")
                docs.append(chunk)
                metas.append(
                    {
                        "source": path.name,
                        "page": page if page is not None else "",
                        "sheet": sheet if sheet is not None else "",
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
        per_source[path.name] = chunk_index
        print(f"  {path.name}: {chunk_index} chunks")

    if not docs:
        raise SystemExit("No chunks produced — nothing to index.")

    print(f"Embedding {len(docs)} chunks ...")
    embeddings = model.encode(
        [PASSAGE_PREFIX + d for d in docs],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    print(f"Persisting to Chroma at {settings.VECTOR_STORE_DIR} ...")
    client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
    try:
        client.delete_collection(settings.VECTOR_COLLECTION)  # idempotent rebuild
    except Exception:
        pass
    collection = client.create_collection(
        settings.VECTOR_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=docs,
        metadatas=metas,
    )

    total = sum(per_source.values())
    print("\nDone.")
    print(f"  collection: {settings.VECTOR_COLLECTION}")
    print(f"  total chunks: {total}")
    for src, n in per_source.items():
        print(f"    - {src}: {n}")
