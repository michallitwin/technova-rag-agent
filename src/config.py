"""Central configuration. Every value can be overridden with an environment variable."""
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # LLM (local Ollama now, Amazon Bedrock later)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")
    llm_timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "120"))

    # Embeddings and vector store
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_path: str = os.getenv("CHROMA_PATH", "chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "technova_docs")
    top_k: int = int(os.getenv("TOP_K", "2"))
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.35"))

    # Data and prompts
    docs_dir: Path = Path(os.getenv("DOCS_DIR", "data/docs"))
    db_path: Path = Path(os.getenv("DB_PATH", "data/technova.db"))
    prompts_dir: Path = Path(os.getenv("PROMPTS_DIR", "config/prompts"))
    eval_set_path: Path = Path(os.getenv("EVAL_SET_PATH", "data/eval/golden_set.json"))
    eval_results_dir: Path = Path(os.getenv("EVAL_RESULTS_DIR", "data/eval/results"))

    # Agent
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "4"))

    # Business rules (decided in code, not by the LLM)
    return_window_days: int = int(os.getenv("RETURN_WINDOW_DAYS", "30"))
    today_override: str | None = os.getenv("TODAY") or None

settings = Settings()


def load_prompt(name: str) -> str:
    """Read a prompt template from the prompts directory (e.g. config/prompts/tools.txt)."""
    return (settings.prompts_dir / f"{name}.txt").read_text(encoding="utf-8").strip()


def current_date() -> date:
    """Return the frozen date from TODAY if set, otherwise the real date."""
    if settings.today_override:
        return date.fromisoformat(settings.today_override)
    return date.today()