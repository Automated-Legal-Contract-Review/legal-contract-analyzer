"""Thin, defensive client for the local Ollama server."""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from . import config


class LLMError(RuntimeError):
    """An error whose message is safe and actionable to show to the user."""


# ----------------------------------------------------------------------------- server / model checks
def list_models() -> tuple[bool, list[str], str | None]:
    try:
        response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=4)
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json().get("models", []) if item.get("name")]
        return True, names, None
    except Exception as exc:
        return False, [], str(exc)


def model_available(models: list[str], wanted: str) -> bool:
    if ":" in wanted:
        return wanted in models
    # "llama3.2" means "llama3.2:latest" but also accept any installed tag of that family.
    return any(name == wanted or name.split(":")[0] == wanted for name in models)


def check_ready(model: str) -> None:
    """Raise LLMError with a fix-it message if analysis cannot possibly work."""
    connected, models, error = list_models()
    if not connected:
        raise LLMError(
            "Ollama is not running or not reachable at "
            f"{config.OLLAMA_URL}. Start the Ollama app (or run 'ollama serve') and try again."
        )
    if not model_available(models, model):
        raise LLMError(f"The model '{model}' is not installed. Run:  ollama pull {model}")


# ----------------------------------------------------------------------------- JSON handling
def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def salvage_json(text: str) -> dict[str, Any] | None:
    """Recover what we can from malformed or truncated JSON (common with small models)."""
    text = _strip_fences(text)
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        return None
    try:
        value, _ = decoder.raw_decode(text[start:])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    result: dict[str, Any] = {}
    summary = re.search(r'"chunk_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if summary:
        try:
            result["chunk_summary"] = json.loads(f'"{summary.group(1)}"')
        except json.JSONDecodeError:
            result["chunk_summary"] = summary.group(1)
    array_start = re.search(r'"findings"\s*:\s*\[', text)
    if array_start:
        position = array_start.end()
        items: list[Any] = []
        while position < len(text):
            while position < len(text) and text[position] in " \r\n\t,":
                position += 1
            if position >= len(text) or text[position] != "{":
                break
            try:
                item, consumed = decoder.raw_decode(text[position:])
            except json.JSONDecodeError:
                break  # the last object was cut off; keep the complete ones
            items.append(item)
            position += consumed
        result["findings"] = items
    return result or None


def parse_json_lenient(content: str) -> dict[str, Any]:
    cleaned = _strip_fences(content)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    salvaged = salvage_json(cleaned)
    if salvaged is None:
        raise LLMError("The local model did not return usable JSON.")
    return salvaged


# ----------------------------------------------------------------------------- chat calls
def _post_chat(payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    try:
        response = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=(10, timeout))
    except requests.ConnectionError as exc:
        raise LLMError(f"Lost connection to Ollama at {config.OLLAMA_URL}. Is it still running? ({exc})") from exc
    except requests.Timeout as exc:
        raise LLMError(
            f"Ollama took longer than {timeout} seconds to answer. The model may be too slow for this machine; "
            "try a smaller CHUNK_CHARS value in .env."
        ) from exc
    if response.status_code != 200:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        raise LLMError(f"Ollama returned HTTP {response.status_code}: {str(detail)[:300]}")
    try:
        return response.json()
    except ValueError as exc:
        raise LLMError("Ollama returned a response that is not valid JSON.") from exc


def chat_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    schema: dict[str, Any] | None = None,
    num_predict: int | None = None,
    timeout: int | None = None,
    attempts: int = 2,
) -> dict[str, Any]:
    """Ask the model for a JSON object. Uses a JSON schema when supported, falls back to plain JSON mode,
    and retries once if the answer cannot be parsed."""
    use_schema = schema is not None
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        payload = {
            "model": model,
            "stream": False,
            "format": schema if use_schema else "json",
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0,
                "seed": 7,
                "num_ctx": config.NUM_CTX,
                "num_predict": num_predict or config.NUM_PREDICT,
            },
        }
        try:
            body = _post_chat(payload, timeout or config.LLM_TIMEOUT)
        except LLMError as exc:
            message = str(exc)
            # Older Ollama versions reject schema-style "format"; retry once with plain JSON mode.
            if use_schema and "HTTP 400" in message:
                use_schema = False
                last_error = exc
                continue
            if "Lost connection" in message and attempt < attempts:
                time.sleep(3)
                last_error = exc
                continue
            raise
        content = (body.get("message") or {}).get("content", "")
        try:
            return parse_json_lenient(content)
        except LLMError as exc:
            last_error = exc
            continue
    raise LLMError(str(last_error) if last_error else "The local model failed to answer.")


def chat_text(system_prompt: str, user_prompt: str, model: str, num_predict: int = 700, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1, "num_ctx": config.NUM_CTX, "num_predict": num_predict},
    }
    body = _post_chat(payload, timeout)
    return ((body.get("message") or {}).get("content") or "").strip()


# ----------------------------------------------------------------------------- embeddings
def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with the local embedding model (supports old and new Ollama endpoints)."""
    if not texts:
        return []
    try:
        response = requests.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts, "truncate": True, "keep_alive": "30m"},
            timeout=(10, 300),
        )
        if response.status_code == 404 and "not found" not in response.text.lower():
            raise FileNotFoundError  # endpoint missing on very old Ollama; use the legacy endpoint
        if response.status_code != 200:
            raise LLMError(f"Embedding request failed (HTTP {response.status_code}): {response.text[:200]}")
        vectors = response.json().get("embeddings")
        if isinstance(vectors, list) and len(vectors) == len(texts):
            return vectors
        raise LLMError("Unexpected embedding response from Ollama.")
    except FileNotFoundError:
        vectors = []
        for text in texts:
            legacy = requests.post(
                f"{config.OLLAMA_URL}/api/embeddings",
                json={"model": config.EMBED_MODEL, "prompt": text},
                timeout=(10, 120),
            )
            legacy.raise_for_status()
            vectors.append(legacy.json()["embedding"])
        return vectors
    except requests.RequestException as exc:
        raise LLMError(f"Embedding request failed: {exc}") from exc
