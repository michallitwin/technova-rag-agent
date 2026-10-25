from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.llm import ask_ollama

DOCS_DIR = Path("data/docs")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="technova_docs", metadata={"hnsw:space": "cosine"})


def chunk_by_paragraph(text):
    """Split text into paragraphs (separated by an empty line), skipping empty pieces."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def ingest_docs():
    """Read every .txt file in data/docs, chunk it and store the chunks in Chroma."""
    for path in DOCS_DIR.glob("*.txt"):
        chunks = chunk_by_paragraph(path.read_text(encoding="utf-8"))
        collection.add(
            ids=[f"{path.stem}-{i}" for i in range(len(chunks))],          # e.g. regulamin-2
            documents=chunks,
            embeddings=model.encode(chunks).tolist(),
            metadatas=[{"section": c.split(".")[0], "source": path.name} for c in chunks],
        )


def search_db(query, k=2):
    """Return (section, source, similarity, text) for the k chunks most similar to the query."""
    res = collection.query(query_embeddings=model.encode([query]).tolist(), n_results=k)
    return [(m["section"], m["source"], round(1 - d, 2), doc)
            for doc, d, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])]


def rag_answer(question, k=2):
    """Retrieve chunks, ask the LLM to answer from them, and append the sources from metadata."""
    results = search_db(question, k=k)
    context = "\n\n".join(f"[{sec}] {doc}" for sec, src, sim, doc in results)
    system = ("Answer in one or two sentences, using only the context. "
              "If the context does not contain the answer, say you don't know.")
    answer = ask_ollama(f"Context:\n{context}\n\nQuestion: {question}", system=system)
    sources = sorted({f"{src} / {sec}" for sec, src, sim, doc in results})
    return f"{answer}\nSources: {', '.join(sources)}"


if __name__ == "__main__":
    ingest_docs()
    print("chunks in database:", collection.count())
    print(rag_answer("How much does delivery cost?"))