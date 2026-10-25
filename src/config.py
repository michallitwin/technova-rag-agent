"""Central configuration. Every value can be overridden with an environment variable."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # LLM (local Ollama now, Amazon Bedrock later)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")

    # Embeddings and vector store
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_path: str = os.getenv("CHROMA_PATH", "chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "technova_docs")
    top_k: int = int(os.getenv("TOP_K", "2"))

    # Data
    docs_dir: Path = Path(os.getenv("DOCS_DIR", "data/docs"))
    db_path: Path = Path(os.getenv("DB_PATH", "data/technova.db"))

    # Request limits
    llm_timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "120"))
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "4"))
    prompts_dir: Path = Path(os.getenv("PROMPTS_DIR", "config/prompts"))

settings = Settings()


def load_prompt(name: str) -> str:
    """Read a prompt template from the prompts directory (e.g. config/prompts/tools.txt)."""
    return (settings.prompts_dir / f"{name}.txt").read_text(encoding="utf-8").strip()