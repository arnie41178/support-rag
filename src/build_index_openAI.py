# src/build_index_openAI.py  (OpenAI variant)
import json, os, tqdm
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

load_dotenv()
client_oa = OpenAI()

CHUNKS_PATH = "support-rag/data/kb_chunks.jsonl"
PERSIST_DIR = "support-rag/chroma_index"

def load_chunks(path):
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        yield rec["chunk_id"], rec["text"], rec.get("title",""), rec.get("url","")

def embed_texts(texts):
    resp = client_oa.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [d.embedding for d in resp.data]

if __name__ == "__main__":
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name="kb_chunks", metadata={"hnsw:space":"cosine"})

    ids, docs, metas = [], [], []
    for cid, text, title, url in tqdm.tqdm(load_chunks(CHUNKS_PATH)):
        ids.append(cid); docs.append(text); metas.append({"title": title, "url": url})
        if len(ids) >= 128:
            embs = embed_texts(docs)
            collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
            ids, docs, metas = [], [], []
    if ids:
        embs = embed_texts(docs)
        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    print("✅ Indexed with OpenAI embeddings")
