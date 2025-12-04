# rag_query.py
import os
import json
import time
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
MONGO_URI =  "mongodb+srv://rag_user:rag123@health-rag-cluster.qnzikcs.mongodb.net/?appName=health-rag-cluster" # set this in your environment
DB_NAME = "health_rag"
COLLECTION_NAME = "patient_vectors"
VECTOR_INDEX_NAME = "vector_index"   # the index you created
TOP_K = 5                           # how many retrieved chunks to pass to LLM
NUM_CANDIDATES = 200                # how many to consider during vector search
MODEL_NAME = "all-MiniLM-L6-v2"     # same model used to build DB embeddings
DEEPSEEK_API_KEY = "sk-86a84040c3554c7cbde7a5e75423d007" # set if using DeepSeek
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"  # replace if wrong

# --- INIT ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
embed_model = SentenceTransformer(MODEL_NAME)

def embed_query(query_text):
    """Return a normalized embedding vector (list)."""
    v = embed_model.encode(query_text)
    v = v / np.linalg.norm(v)
    return v.tolist()

def retrieve_topk(query_vec, top_k=TOP_K, num_candidates=NUM_CANDIDATES):
    """Run MongoDB vectorSearch aggregation and return ordered results."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": num_candidates,
                "limit": top_k
            }
        },
        {"$project": {"_id": 0, "summary": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    results = list(collection.aggregate(pipeline))
    return results  # list of dicts with 'summary' and 'score'

def build_prompt(retrieved_chunks, user_query, max_context_chars=3500):
    """
    Build a system + user prompt for the LLM.
    Trims context if too long (simple char-based truncation).
    Returns messages compatible with chat-style LLMs.
    """
    # System message: rules and safety
    system_message = (
    "You are a retrieval-augmented assistant.\n"
    "Your answers must be strictly grounded in the provided Context. "
    "If the context does not contain an answer, say:\n"
    "'The context does not provide that information.'\n\n"

    "STRICT RULES:\n"
    "1. Do NOT invent names, facts, or medical details.\n"
    "2. Do NOT guess or hallucinate.\n"
    "3. Use only information found in the context.\n"
    "4. Cite each fact using [source #X].\n"
    "5. Do NOT give medical advice.\n\n"

    "STYLE RULES (make the answer feel human, smooth, and engaging WITHOUT breaking the strict rules):\n"
    "1. You may use light personality, warmth, and natural language.\n"
    "2. You may provide non-medical observations (e.g., patterns visible only in the context).\n"
    "3. You may phrase insights in a flowing, friendly way.\n"
    "4. Keep the tone readable and interesting, but stay grounded.\n"
    "5. Never exaggerate or add unsafe speculation.\n"
    "6. Add a subtle storytelling/witty vibe when possible.\n\n"

    "Your goal: factual + grounded + slightly fun to read.\n"
)



    # Combine top-k chunks into a single context string, with small identifiers
    assembled = []
    for i, r in enumerate(retrieved_chunks):
        # add an id so the model can cite: [source: #3]
        assembled.append(f"[source #{i+1}] {r['summary']}")
    context = "\n\n".join(assembled)

    # Trim context if it's too long (simple but effective)
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n...[TRUNCATED CONTEXT]..."

    user_message = f"Context:\n{context}\n\nUser question: {user_query}\n\nAnswer using only the Context above. Provide brief source citations like [source #1]."

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def call_deepseek(messages, max_tokens=1000, temperature=1.0):
    """
    Example POST call to DeepSeek-like chat endpoint.
    Replace with official SDK if you have one.
    """
    import requests
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",   # adapt to the model you have access to
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    # adapt path according to the API response format
    return j["choices"][0]["message"]["content"]

def run_rag_once(user_query, verbose=True):
    qvec = embed_query(user_query)
    retrieved = retrieve_topk(qvec, top_k=TOP_K, num_candidates=NUM_CANDIDATES)
    if verbose:
        print(f"Retrieved {len(retrieved)} chunks. Top scores: {[round(r['score'],4) for r in retrieved]}")
    messages = build_prompt(retrieved, user_query)
    # Danger check: if no DEEPSEEK key, just return the retrievals
    if not DEEPSEEK_API_KEY:
        return {
            "answer": None,
            "retrieved": retrieved,
            "note": "No DEEPSEEK_API_KEY - returning retrieved context only."
        }
    answer = call_deepseek(messages)
    # log the full exchange locally for auditing
    log = {
        "timestamp": time.time(),
        "query": user_query,
        "retrieved": retrieved,
        "answer": answer
    }
    with open("logs/rag_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
    return {"answer": answer, "retrieved": retrieved}

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    q = input("Ask something: ")
    out = run_rag_once(q, verbose=True)
    if out["answer"]:
        print("\n=== FINAL ANSWER ===\n")
        print(out["answer"])
    else:
        print("\n=== RETRIEVED CONTEXT (no LLM key) ===\n")
        for i, r in enumerate(out["retrieved"]):
            print(f"--- source #{i+1} | score {r['score']:.4f} ---")
            print(r["summary"])
            print()
