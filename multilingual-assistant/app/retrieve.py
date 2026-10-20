
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings
#from analytics.load_data import CODE_PATTERN, load_fault_codes
import re



_collection = None
_model = None

def embed(texts, is_query=False):
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    prefix = "query: " if is_query else "passage: "
    return _model.encode([prefix + t for t in texts], normalize_embeddings=True).tolist()

def normalize_code(s: str) -> str:
    return re.sub(r"[-_ ]", "", s.upper())

# ------------------------------------------------------------------
# Retrieval  — exact code lookup + code-filtered history + semantic
# ------------------------------------------------------------------
def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
        _collection = client.get_collection(settings.VECTOR_COLLECTION) 
    return _collection


def retrieve(query, k=5):
    col = get_collection()

    ctx = {"exact_codes": [], "code_logs": [], "semantic": []}
    q_emb = embed([query], is_query=True)

    res = col.query(
        query_embeddings=q_emb, n_results=k,
        where={"$and": [{"source_type": "log"}]},
    )
    ctx["code_logs"] = res["documents"][0]

    # 3. general semantic pass — instructions + any similar cases
    res = col.query(query_embeddings=q_emb, n_results=k)
    ctx["semantic"] = res["documents"][0]
    return ctx
