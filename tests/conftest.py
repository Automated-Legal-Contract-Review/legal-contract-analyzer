"""Test setup: start fake Ollama/Qdrant servers and point the app at them BEFORE the app is imported."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fakes  # noqa: E402

_tmp = tempfile.mkdtemp(prefix="lexiguard-test-")
OLLAMA = fakes.FakeOllama()
QDRANT = fakes.FakeQdrant()
_ollama_port, _qdrant_port = fakes.free_port(), fakes.free_port()
fakes.serve(OLLAMA.app, _ollama_port)
fakes.serve(QDRANT.app, _qdrant_port)

os.environ["LEXIGUARD_DATA_DIR"] = _tmp
os.environ["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{_ollama_port}"
os.environ["QDRANT_URL"] = f"http://127.0.0.1:{_qdrant_port}"
os.environ["CHUNK_CHARS"] = "4000"
os.environ["CHUNK_OVERLAP"] = "400"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def reset_fakes():
    OLLAMA.mode = "normal"
    OLLAMA.models = ["llama3.2:latest", "nomic-embed-text:latest"]
    OLLAMA.chat_calls = 0
    QDRANT.healthy = True
    yield


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
