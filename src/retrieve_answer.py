# src/retrieve_answer.py
import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings

PERSIST_DIR = "chroma_index"

def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection("kb_chunks")

def embed_query(text: str):
    resp = client_oa.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    return resp.data[0].embedding

def retrieve(query: str, k: int = 3):
    col = get_collection()
    query_embedding = embed_query(query)
    res = col.query(query_embeddings=[query_embedding], n_results=k, include=["documents","metadatas","distances"])
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "chunk_id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "url": res["metadatas"][0][i].get("url"),
            "title": res["metadatas"][0][i].get("title"),
            "score": 1 - res["distances"][0][i]  # cosine similarity proxy
        })
    return hits

from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv(".env")
client_oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
"You are a helpful ecommerce support agent. "
"Use ONLY the provided chunks. Cite chunk IDs like (id: <chunk_id>). "
"If the answer is missing, say: 'This may require escalation.'"
)

def answer_with_rag(question: str, k: int = 3) -> Dict:
    chunks = retrieve(question, k=k)
    context = "\n\n".join([f"[id:{c['chunk_id']}] {c['text']}" for c in chunks])
    prompt = f"""Customer question: {question}

Use only the 'Chunks' below. Be concise (<=120 words), empathetic, and actionable.
If uncertain, say it may require escalation.

Chunks:
{context}
"""
    resp = client_oa.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":prompt}],
        temperature=0.2
    )
    return {"answer": resp.choices[0].message.content, "chunks": chunks}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrieve answer using RAG search")
    parser.add_argument("query", help="The question/query to search for")
    parser.add_argument("-k", "--num_results", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    out = answer_with_rag(args.query, k=args.num_results)
    print("\nANSWER:\n", out["answer"])
    print("\nCHUNKS USED:")
    for c in out["chunks"]:
        print(f"- {c['chunk_id']}  {c['title']}  ({c['score']:.3f})  {c['url']}")