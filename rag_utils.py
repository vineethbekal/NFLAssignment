# rag_utils.py
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np, faiss

class RAG:
    def __init__(self):
        self.emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    def _embed(self, texts): return self.emb.encode(texts, normalize_embeddings=True)
    def summarize(self, docs, k=5):
        texts = [d["content"] for d in docs]
        embs = self._embed(texts)
        index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs.astype("float32"))
        # Use the average vector of titles as a cheap query
        q = self._embed([" ".join([d["title"] for d in docs[:k]])])[0].astype("float32").reshape(1,-1)
        _, idx = index.search(q, min(k, len(docs)))
        picked = [docs[i] for i in idx[0]]
        context = "\n\n".join(p["content"][:2000] for p in picked)[:6000]
        out = self.summarizer(context, max_length=220, min_length=120, do_sample=False)[0]["summary_text"]
        return out, picked
