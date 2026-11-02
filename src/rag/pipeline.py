"""RAG over store documents: chunking, indexing in Chroma, search and answering."""

from functools import lru_cache

from src.config import load_prompt, settings
from src.llm.client import ask_ollama


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once, on first use (lazy loading)."""
    from sentence_transformers import (
        SentenceTransformer,
    )  # heavy import (PyTorch), only when needed

    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_collection():
    """Open the Chroma collection once, on first use (lazy loading)."""
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(
        name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )


def chunk_by_paragraph(text):
    """Split text into paragraphs (separated by an empty line), skipping empty pieces."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def ingest_docs():
    """Read every .txt file in the docs folder, chunk it and store the chunks in Chroma."""
    for path in settings.docs_dir.glob("*.txt"):
        chunks = chunk_by_paragraph(path.read_text(encoding="utf-8"))
        get_collection().upsert(
            ids=[f"{path.stem}-{i}" for i in range(len(chunks))],  # e.g. store_policy-2
            documents=chunks,
            embeddings=get_model().encode(chunks).tolist(),
            metadatas=[
                {"section": c.split(".")[0], "source": path.name} for c in chunks
            ],
        )


def search_db(query, k=settings.top_k):
    """Return (section, source, similarity, text) for the k chunks most similar to the query."""
    res = get_collection().query(
        query_embeddings=get_model().encode([query]).tolist(), n_results=k
    )
    return [
        (m["section"], m["source"], round(1 - d, 2), doc)
        for doc, d, m in zip(
            res["documents"][0], res["distances"][0], res["metadatas"][0], strict=True
        )
    ]


def rag_answer(question, k=settings.top_k):
    """Retrieve chunks, ask the LLM to answer from them, and append the sources from metadata."""
    results = search_db(question, k=k)
    context = "\n\n".join(f"[{sec}] {doc}" for sec, src, sim, doc in results)
    answer = ask_ollama(
        f"Context:\n{context}\n\nQuestion: {question}",
        system=load_prompt("answer_system"),
    )
    sources = sorted({f"{src} / {sec}" for sec, src, sim, doc in results})
    return f"{answer}\nSources: {', '.join(sources)}"


def main() -> None:
    """CLI entry point: (re)build the vector index from the docs folder."""
    ingest_docs()
    print(f"Chunks in database: {get_collection().count()}")


if __name__ == "__main__":
    main()
