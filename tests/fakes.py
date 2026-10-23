"""Tiny stand-ins for Ollama and Qdrant so the whole pipeline can be tested without a GPU or Docker."""
from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

TRIGGERS = [
    ("unlimited liability", "Liability", "High", 5, "Unlimited liability exposure"),
    ("automatically renew", "Renewal", "Medium", 3, "Automatic renewal"),
    ("indemnify", "Indemnity", "High", 4, "Broad indemnity"),
    ("terminate this agreement at any time", "Termination", "Medium", 3, "Unilateral termination"),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeOllama:
    def __init__(self) -> None:
        self.mode = "normal"  # normal | fail500 | badjson | truncate | schema400
        self.models = ["llama3.2:latest", "nomic-embed-text:latest"]
        self.chat_calls = 0
        self.schema_formats_seen = 0
        self.last_user_prompt = ""
        self.app = FastAPI()
        self._routes()

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * 64
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _routes(self) -> None:
        app = self.app

        @app.get("/api/tags")
        def tags():
            return {"models": [{"name": n} for n in self.models]}

        @app.post("/api/embed")
        async def embed(request: Request):
            data = await request.json()
            return {"embeddings": [self._vector(t) for t in data["input"]]}

        @app.post("/api/chat")
        async def chat(request: Request):
            data = await request.json()
            self.chat_calls += 1
            system = data["messages"][0]["content"]
            user = data["messages"][1]["content"]
            self.last_user_prompt = user
            fmt = data.get("format")
            if isinstance(fmt, dict):
                self.schema_formats_seen += 1
                if self.mode == "schema400":
                    return JSONResponse({"error": "invalid format"}, status_code=400)
            if self.mode == "fail500":
                return JSONResponse({"error": "boom"}, status_code=500)

            if "contract-review assistant" in system:
                excerpt = user.split("<contract_excerpt>")[-1].split("</contract_excerpt>")[0]
                findings = []
                for needle, category, level, severity, title in TRIGGERS:
                    match = re.search(r"[^.\n]*" + re.escape(needle) + r"[^.\n]*\.?", excerpt, flags=re.I)
                    if match:
                        findings.append({
                            "title": title, "category": category, "risk_level": level, "severity_score": severity,
                            "evidence": match.group(0).strip(),
                            "explanation": f"This wording on '{needle}' may expose a party to significant risk.",
                            "recommendation": "Ask whether this can be capped or made mutual.",
                        })
                if "unlimited liability" in excerpt.lower():  # a hallucinated quote that must be discarded
                    findings.append({
                        "title": "Invented clause", "category": "Other", "risk_level": "High", "severity_score": 5,
                        "evidence": "The supplier shall pay one million pigeons on demand.",
                        "explanation": "Made up.", "recommendation": "n/a",
                    })
                body = {"chunk_summary": "This part sets out commercial terms.", "findings": findings}
                content = json.dumps(body)
                if self.mode == "badjson":
                    content = "Sure! Here is my analysis: I could not do it."
                if self.mode == "truncate" and len(findings) >= 1:
                    content = json.dumps(body)[:-25]  # cut in the middle of the last object
                return {"message": {"content": content}, "done": True}
            if "condense notes" in system:
                return {"message": {"content": json.dumps({"summary": "Merged: commercial terms continue."})}}
            if "synthesise contract-review notes" in system:
                return {"message": {"content": json.dumps({
                    "executive_summary": "A services agreement with several points that deserve review.",
                    "key_obligations": ["Supplier delivers services", {"obligation": "Customer pays invoices"}],
                    "reviewer_checklist": ["Negotiate a liability cap"],
                    "limitations": ["Not legal advice"],
                })}}
            if "answer questions about a contract" in system:
                return {"message": {"content": "The contract allows automatic renewal [1]."}}
            return {"message": {"content": "{}"}}


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, list[dict]] = {}
        self.healthy = True
        self.app = FastAPI()
        app = self.app

        @app.get("/healthz")
        def healthz():
            if not self.healthy:
                return JSONResponse({"error": "down"}, status_code=503)
            return "healthz check passed"

        @app.put("/collections/{name}")
        def create(name: str):
            self.collections[name] = []
            return {"result": True}

        @app.delete("/collections/{name}")
        def delete(name: str):
            self.collections.pop(name, None)
            return {"result": True}

        @app.put("/collections/{name}/points")
        async def upsert(name: str, request: Request):
            data = await request.json()
            self.collections[name].extend(data["points"])
            return {"result": {"status": "completed"}}

        @app.post("/collections/{name}/points/search")
        async def search(name: str, request: Request):
            data = await request.json()
            query = data["vector"]
            scored = []
            for point in self.collections.get(name, []):
                score = sum(a * b for a, b in zip(query, point["vector"]))
                scored.append({"id": point["id"], "score": score, "payload": point["payload"]})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return {"result": scored[: data.get("limit", 5)]}


def serve(app: FastAPI, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server
