# app.py
import os, time, csv, json, datetime as dt
import streamlit as st
from retrieve_answer import retrieve, answer_with_rag

# =========================
# 🔌 Adapter: plug YOUR code here
# - Implement these two functions (or adapt the bodies to call yours).
# - Expected return formats are documented below.
# =========================

def retrieve_top_k(query: str, k: int = 3):
    """
    Return a list[dict] of retrieved chunks sorted by relevance.
    Each dict should have: chunk_id, text, title (opt), url (opt), score (0..1 higher=better)
    Example:
    [
      {"chunk_id":"kb_001_0","text":"...", "title":"Return Policy","url":"...", "score":0.83},
      ...
    ]
    """
    return retrieve(query, k)


def generate_answer(question: str, k: int = 3, model_name: str = "gpt-4o-mini"):
    """
    Return (answer_text, chunks_used:list[dict], latency_seconds, model_name)
    - chunks_used should be the SAME shape as retrieve_top_k() output (top-k actually used).
    """
    import time
    start_time = time.time()
    result = answer_with_rag(question, k=k)
    latency = time.time() - start_time
    return result["answer"], result["chunks"], latency, model_name


# =========================
# ⚙️ App config
# =========================
st.set_page_config(page_title="Shipping & Returns — RAG Demo", page_icon="📦", layout="wide")

# Logging config
LOG_PATH = "logs.csv"              # full interaction logs (optional)
LEARNING_BUFFER_PATH = "learning_buffer.jsonl"  # approved corrections for future fine-tune

def ensure_log_header(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts","user_query","model_answer","model_name","latency_ms",
                "retrieved_ids","retrieved_scores","retrieved_titles","retrieved_urls",
                "feedback","gold_answer","notes"
            ])

def append_log(row: dict):
    ensure_log_header(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row.get("ts",""),
            row.get("user_query",""),
            row.get("model_answer",""),
            row.get("model_name",""),
            row.get("latency_ms",""),
            ";".join(row.get("retrieved_ids",[])),
            ";".join([f"{s:.3f}" for s in row.get("retrieved_scores",[])]),
            ";".join(row.get("retrieved_titles",[])),
            ";".join(row.get("retrieved_urls",[])),
            row.get("feedback",""),
            row.get("gold_answer",""),
            row.get("notes","")
        ])

def append_learning_buffer(example: dict):
    # JSONL with fields: instruction, input_context, output_gold, meta
    with open(LEARNING_BUFFER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

# =========================
# 🧭 Sidebar controls
# =========================
st.sidebar.header("Controls")
top_k = st.sidebar.slider("Top-K chunks", min_value=1, max_value=6, value=3, step=1)
model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-2024-07-18"])
logging_on = st.sidebar.toggle("Log interactions to logs.csv", value=True)
show_json = st.sidebar.toggle("Show raw JSON (debug)")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: wire your own functions in the Adapter at the top.")

# =========================
# 🖼️ Header
# =========================
st.title("📦 Shipping & Returns — RAG Demo")
st.caption("Ask a customer-style question. The app retrieves KB chunks and generates a grounded answer.")

# =========================
# 💬 Query box
# =========================
default_q = "Tracking says delivered but I can't find my package. What should I do?"

# Initialize session state for query text
if "query_text" not in st.session_state:
    st.session_state.query_text = default_q

query = st.text_input("Customer question", value=st.session_state.query_text, help="Type a support question")

col_run, col_clear = st.columns([1,1/3])
run_clicked = col_run.button("Answer", type="primary")
if col_clear.button("Clear"):
    st.session_state.query_text = ""
    st.rerun()

# State for feedback UI
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================
# ▶️ Run pipeline
# =========================
if run_clicked and query.strip():
    # Retrieve (for transparency, even if your answerer also retrieves internally)
    try:
        t0 = time.time()
        chunks = retrieve_top_k(query, k=top_k)
        t_retrieval = (time.time() - t0) * 1000
    except NotImplementedError as e:
        st.error(f"Retrieval not wired: {e}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    # Generate answer
    try:
        answer, used_chunks, latency, used_model = generate_answer(query, k=top_k, model_name=model_name)
    except NotImplementedError as e:
        st.error(f"Answer generation not wired: {e}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    # Display answer
    st.markdown("### ✅ Answer")
    st.write(answer)
    st.caption(f"Model: `{used_model}` • Latency: {latency*1000:.0f} ms • Retrieval: {t_retrieval:.0f} ms")

    # Display retrieved chunks (two tabs: used vs retrieved)
    tab1, tab2 = st.tabs(["Chunks used", "Top-K retrieved"])
    with tab1:
        if used_chunks:
            for c in used_chunks:
                with st.expander(f"[{c.get('chunk_id','?')}] {c.get('title','') or '(no title)'} — score {c.get('score',0):.3f}"):
                    st.write(c.get("text",""))
                    if c.get("url"):
                        st.write(c["url"])
        else:
            st.info("No chunks reported by the answerer. Showing retrieved top-k in the next tab.")

    with tab2:
        for c in chunks:
            with st.expander(f"[{c.get('chunk_id','?')}] {c.get('title','') or '(no title)'} — score {c.get('score',0):.3f}"):
                st.write(c.get("text",""))
                if c.get("url"):
                    st.write(c["url"])

    # Feedback & logging panel
    st.markdown("---")
    st.subheader("Feedback")
    fb_cols = st.columns([1,1,2])
    good = fb_cols[0].button("👍 Helpful")
    bad  = fb_cols[1].button("👎 Needs work")
    gold_answer = fb_cols[2].text_input("Optional: provide a corrected 'gold' answer (will be saved for learning)")

    notes = st.text_area("Notes (optional)", placeholder="e.g., Missing return date; tone too long; cite chunk kb_002_1")
    save_gold = st.button("Save corrected answer to learning buffer")

    # Save state for later
    st.session_state.last_result = {
        "query": query,
        "answer": answer,
        "model": used_model,
        "latency_ms": int(latency*1000),
        "chunks": used_chunks if used_chunks else chunks,
    }

    # Logging
    log_row = {
        "ts": dt.datetime.utcnow().isoformat(),
        "user_query": query,
        "model_answer": answer,
        "model_name": used_model,
        "latency_ms": int(latency*1000),
        "retrieved_ids": [c.get("chunk_id","") for c in (used_chunks if used_chunks else chunks)],
        "retrieved_scores": [float(c.get("score",0.0)) for c in (used_chunks if used_chunks else chunks)],
        "retrieved_titles": [c.get("title","") or "" for c in (used_chunks if used_chunks else chunks)],
        "retrieved_urls": [c.get("url","") or "" for c in (used_chunks if used_chunks else chunks)],
        "feedback": ("up" if good else ("down" if bad else "")),
        "gold_answer": gold_answer.strip() if gold_answer else "",
        "notes": notes.strip(),
    }
    if logging_on:
        append_log(log_row)

    # Learning buffer save
    if save_gold and gold_answer.strip():
        # construct JSONL gold example (grounded to the chunks used)
        input_ctx = "KB:[" + ",".join(log_row["retrieved_ids"]) + "]"
        example = {
            "instruction": query,
            "input_context": input_ctx,
            "output_gold": gold_answer.strip(),
            "meta": {
                "category": "Shipping",            # tweak as needed / add a dropdown to capture
                "source": "organic",
                "requires_escalation": False
            }
        }
        append_learning_buffer(example)
        st.success(f"Saved corrected answer to {LEARNING_BUFFER_PATH}")

    # Optional raw JSON
    if show_json:
        st.markdown("#### Debug JSON")
        st.json(log_row)

# Footer
st.markdown("---")
st.caption("Demo UI for grounded QA over KB chunks. Replace the Adapter functions with your own retriever/answerer.")
