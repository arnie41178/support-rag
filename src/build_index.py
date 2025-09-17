# src/build_index.py
import json, os, tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "support-rag/data/kb_chunks.jsonl"
PERSIST_DIR = "support-rag/chroma_index"   # folder will be created

def load_chunks(path):
    currDir = os.getcwd()
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        yield rec["chunk_id"], rec["text"], rec.get("title",""), rec.get("url","")

if __name__ == "__main__":
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    collection = client.get_or_create_collection(
        name="kb_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # local
    ids, docs, metas = [], [], []

    for cid, text, title, url in tqdm.tqdm(load_chunks(CHUNKS_PATH)):
        ids.append(cid)
        docs.append(text)
        metas.append({"title": title, "url": url})

        # Batch insert in chunks of 256 for speed
        if len(ids) >= 256:
            embs = model.encode(docs, batch_size=64, normalize_embeddings=True).tolist()
            collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        embs = model.encode(docs, batch_size=64, normalize_embeddings=True).tolist()
        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    print("✅ Indexed chunks into Chroma at", PERSIST_DIR)

