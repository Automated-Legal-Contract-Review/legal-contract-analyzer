"""Central configuration. Values come from environment variables or a local .env file."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Tiny .env loader (no extra dependency). Real environment variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file(BASE_DIR / ".env")


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


VERSION = "2.0.0"

DATA_DIR = Path(os.getenv("LEXIGUARD_DATA_DIR", str(BASE_DIR / "data"))).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
TEXT_DIR = DATA_DIR / "text"
DB_PATH = DATA_DIR / "lexiguard.sqlite3"
STATIC_DIR = BASE_DIR / "static"

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
USE_QDRANT = _bool("USE_QDRANT", True)

# Analysis tuning. Smaller chunks = better accuracy from small local models, but slower.
CHUNK_CHARS = _int("CHUNK_CHARS", 8000, 2000, 30000)
CHUNK_OVERLAP = min(_int("CHUNK_OVERLAP", 800, 0, 5000), CHUNK_CHARS // 3)
MAX_FINDINGS_PER_CHUNK = _int("MAX_FINDINGS_PER_CHUNK", 8, 1, 25)
NUM_CTX = _int("NUM_CTX", 8192, 2048, 131072)
NUM_PREDICT = _int("NUM_PREDICT", 2200, 400, 8000)
LLM_TIMEOUT = _int("LLM_TIMEOUT_SECONDS", 900, 30, 7200)
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 100, 1, 1000)

# Passages used for the optional "Ask the contract" feature.
PASSAGE_CHARS = 1400
PASSAGE_OVERLAP = 200

HOST = os.getenv("LEXIGUARD_HOST", "127.0.0.1")
PORT = _int("LEXIGUARD_PORT", 8000, 1, 65535)

for _folder in (DATA_DIR, UPLOAD_DIR, TEXT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)
