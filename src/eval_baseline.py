# src/eval_baseline.py
import json, numpy as np, tqdm
from sentence_transformers import SentenceTransformer
from retrieve_answer import answer_with_rag

GOLD_PATH = "support-rag/data/qa_from_kb.jsonl"

def load_gold(n_max=None):
    items = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            items.append({"q": rec["instruction"], "gold": rec["output_gold"]})
            if n_max and i+1 >= n_max: break
    return items

if __name__ == "__main__":
    gold = load_gold(n_max=100)  # sample first 100 for speed
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    gold_vecs = enc.encode([g["gold"] for g in gold], normalize_embeddings=True)
    preds = []
    for g in tqdm.tqdm(gold):
        ans = answer_with_rag(g["q"], k=3)["answer"]
        preds.append(ans)
    pred_vecs = enc.encode(preds, normalize_embeddings=True)
    sims = (gold_vecs * pred_vecs).sum(axis=1)  # cosine since normalized
    print(f"Mean cosine similarity: {float(np.mean(sims)):.3f}")
    print(f"% >= 0.80: {100*np.mean(sims>=0.80):.1f}%")
