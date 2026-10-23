"""Optional semantic index (Qdrant) with a keyword fallback so questions work even without Docker."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import requests

from . import config, llm

BATCH = 16


def collection_name(contract_id: str) -> str:
    return "lg_" + contract_id.replace("-", "")


def qdrant_alive() -> tuple[bool, str | None]:
    if not config.USE_QDRANT:
        return False, "Disabled (USE_QDRANT=false)"
    try:
        response = requests.get(f"{config.QDRANT_URL}/healthz", timeout=3)
        if response.ok:
            return True, None
        return False, f"HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def index_passages(contract_id: str, passages: list[dict[str, Any]]) -> None:
    """Embed passages and store them in a per-contract Qdrant collection. Raises on any failure."""
    alive, error = qdrant_alive()
    if not alive:
        raise RuntimeError(f"Qdrant is not available ({error}).")
    vectors: list[list[float]] = []
    for offset in range(0, len(passages), BATCH):
        vectors.extend(llm.embed([p["text"] for p in passages[offset:offset + BATCH]]))
    if not vectors:
        raise RuntimeError("No vectors were generated.")

    name = collection_name(contract_id)
    url = f"{config.QDRANT_URL}/collections/{name}"
    requests.delete(url, timeout=10)  # start clean; ignore if it does not exist
    create = requests.put(url, json={"vectors": {"size": len(vectors[0]), "distance": "Cosine"}}, timeout=15)
    create.raise_for_status()

    points = [
        {
            "id": passage["i"],
            "vector": vector,
            "payload": {
                "contract_id": contract_id,
                "i": passage["i"],
                "page_start": passage["page_start"],
                "page_end": passage["page_end"],
                "text": passage["text"],
            },
        }
        for passage, vector in zip(passages, vectors)
    ]
    for offset in range(0, len(points), 64):
        response = requests.put(f"{url}/points?wait=true", json={"points": points[offset:offset + 64]}, timeout=60)
        response.raise_for_status()


def delete_collection(contract_id: str) -> None:
    if not config.USE_QDRANT:
        return
    try:
        requests.delete(f"{config.QDRANT_URL}/collections/{collection_name(contract_id)}", timeout=5)
    except Exception:
        pass


def search_vectors(contract_id: str, question: str, limit: int = 6) -> list[dict[str, Any]]:
    vector = llm.embed([question])[0]
    response = requests.post(
        f"{config.QDRANT_URL}/collections/{collection_name(contract_id)}/points/search",
        json={"vector": vector, "limit": limit, "with_payload": True},
        timeout=20,
    )
    response.raise_for_status()
    results = []
    for hit in response.json().get("result", []):
        payload = hit.get("payload") or {}
        results.append({
            "i": payload.get("i", hit.get("id")),
            "text": payload.get("text", ""),
            "page_start": payload.get("page_start", 1),
            "page_end": payload.get("page_end", 1),
            "score": float(hit.get("score", 0.0)),
        })
    return results


# ----------------------------------------------------------------------------- keyword fallback
_STOP = set(
    "the a an and or of to in on for with by is are was were be been being this that these those it its as at from "
    "what which who whom whose when where why how do does did can could should would may might shall will not no "
    "any all each per into than then there their they them our we you your i me my about under over between "
    "contract agreement clause section".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP]


def search_keywords(passages: list[dict[str, Any]], question: str, limit: int = 6) -> list[dict[str, Any]]:
    query_terms = set(_tokens(question))
    if not query_terms or not passages:
        return []
    docs = [Counter(_tokens(p["text"])) for p in passages]
    n = len(docs)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(doc.keys())
    scored: list[tuple[float, dict[str, Any]]] = []
    for passage, doc in zip(passages, docs):
        length = sum(doc.values()) or 1
        score = 0.0
        for term in query_terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / 250))
        if score > 0:
            scored.append((score, passage))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"i": p["i"], "text": p["text"], "page_start": p["page_start"], "page_end": p["page_end"], "score": round(s, 3)}
        for s, p in scored[:limit]
    ]


def fuse(vector_hits: list[dict[str, Any]], keyword_hits: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion: passages that rank well in either search float to the top. Legal questions
    often hinge on exact words (e.g. 'indemnify'), which pure vector search can miss."""
    scores: dict[Any, float] = {}
    by_id: dict[Any, dict[str, Any]] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, hit in enumerate(hits):
            key = hit.get("i", hit["text"][:60])
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            by_id.setdefault(key, hit)
    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [by_id[k] for k in ordered[:limit]]


def best_snippet(text: str, question: str, width: int = 420) -> str:
    """A short excerpt centred on the part of the passage that best matches the question."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    terms = set(_tokens(question))
    lowered = flat.lower()
    best_pos, best_score = 0, -1
    step = max(40, width // 6)
    for pos in range(0, max(1, len(flat) - width // 2), step):
        window = lowered[pos:pos + width]
        score = sum(window.count(t) for t in terms)
        if score > best_score:
            best_pos, best_score = pos, score
    start = max(0, min(best_pos, len(flat) - width))
    prefix = "..." if start > 0 else ""
    suffix = "..." if start + width < len(flat) else ""
    return prefix + flat[start:start + width].strip() + suffix
