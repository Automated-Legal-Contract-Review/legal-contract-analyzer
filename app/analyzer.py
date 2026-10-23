"""The contract-review pipeline: extract -> chunk -> review each chunk -> verify -> merge -> summarise."""
from __future__ import annotations

import json
import math
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from . import chunking, config, db, extract, llm, vectors

CATEGORIES = [
    "Payment", "Termination", "Liability", "Indemnity", "Confidentiality", "Intellectual Property",
    "Privacy and Data", "Warranties", "Governing Law", "Dispute Resolution", "Renewal", "Audit",
    "Service Levels", "Assignment", "Force Majeure", "Compliance", "Other",
]
LEVELS = ["High", "Medium", "Low", "Info"]
LEVEL_ORDER = {level: i for i, level in enumerate(LEVELS)}
SEVERITY_BAND = {"High": (4, 5), "Medium": (3, 3), "Low": (1, 2), "Info": (1, 1)}
PERSPECTIVES = {
    "neutral": "a neutral reviewer who is about to advise a party that is asked to sign this contract",
    "customer": "the customer / buyer / recipient of the services (the party paying and relying on the other side)",
    "provider": "the vendor / provider / seller of the services (the party delivering and being paid)",
}
SUMMARY_BUDGET = 6000  # characters of chunk summaries sent to the final synthesis prompt
MAX_CONSECUTIVE_FAILURES = 3

DISCLAIMER = (
    "LexiGuard is an AI-assisted issue-spotting tool, not a lawyer and not a substitute for advice from a qualified "
    "legal professional. AI can miss issues or misread context. Check every quote and recommendation against the "
    "original contract."
)

CHUNK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chunk_summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "risk_level": {"type": "string", "enum": LEVELS},
                    "severity_score": {"type": "integer"},
                    "evidence": {"type": "string"},
                    "explanation": {"type": "string"},
                    "recommendation": {"type": "string"},
                },
                "required": ["title", "category", "risk_level", "severity_score", "evidence", "explanation", "recommendation"],
            },
        },
    },
    "required": ["chunk_summary", "findings"],
}

SYNTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "key_obligations": {"type": "array", "items": {"type": "string"}},
        "reviewer_checklist": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["executive_summary", "key_obligations", "reviewer_checklist", "limitations"],
}

MERGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------------------- prompts
def chunk_system_prompt(perspective: str) -> str:
    who = PERSPECTIVES.get(perspective, PERSPECTIVES["neutral"])
    return f"""You are a careful contract-review assistant supporting a human reviewer. You review ONLY the excerpt you are given.
Review it from the viewpoint of {who}.

Return one JSON object with:
- "chunk_summary": one or two factual sentences (max 400 characters) saying what this excerpt covers.
- "findings": a list (maximum {config.MAX_FINDINGS_PER_CHUNK}) of concrete issues. Each finding has:
  "title" (short), "category" (one of: {", ".join(CATEGORIES)}), "risk_level" (High, Medium, Low or Info),
  "severity_score" (integer 1-5), "evidence" (an EXACT quote copied from the excerpt, max 300 characters),
  "explanation" (why this wording may matter, max 500 characters),
  "recommendation" (a practical question or negotiation suggestion, max 400 characters).

Rules:
- Only report issues that are visible in the excerpt: one-sided obligations, uncapped or unlimited liability, broad or one-way indemnity,
  unclear payment or acceptance terms, automatic renewal, weak or unilateral termination rights, harsh penalties, confidentiality / IP / data
  gaps, ambiguous remedies, unusual waivers, or materially ambiguous drafting.
- The evidence MUST be copied word for word from the excerpt. If you cannot quote it exactly, leave the finding out.
- Never invent wording, facts, dates, parties, laws or clause numbers.
- Do not say a clause is illegal or unenforceable. Do not claim something is missing from the whole contract based on one excerpt.
- Scale: High = 4-5 (serious exposure), Medium = 3, Low = 1-2, Info = 1.
- Do not repeat the same issue twice. If the excerpt has no concrete issue, return an empty findings list.
- This is an informational review, not legal advice."""


MERGE_SYSTEM = (
    "You condense notes about consecutive parts of one contract. Merge them, in order, into one factual summary "
    "of at most 600 characters. Use only the notes. Return JSON: {\"summary\": \"...\"}."
)

SYNTH_SYSTEM = (
    "You synthesise contract-review notes for a human reviewer. Use only the supplied data. Never invent parties, dates, "
    "jurisdictions, clauses, laws or facts, and never add risks that are not in the provided findings. Return valid JSON."
)


# ----------------------------------------------------------------------------- finding verification
def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _category(value: Any) -> str:
    raw = _text(value, 80)
    for known in CATEGORIES:
        if raw.lower() == known.lower():
            return known
    return "Other"


def grounded_findings(raw: Any, chunk: dict[str, Any], doc: chunking.Document) -> list[dict[str, Any]]:
    """Keep only findings whose quoted evidence really exists in the chunk; attach exact page numbers."""
    if not isinstance(raw, list):
        return []
    matcher = chunking.Matcher(chunk["text"])
    findings: list[dict[str, Any]] = []
    for item in raw[: config.MAX_FINDINGS_PER_CHUNK * 2]:
        if not isinstance(item, dict):
            continue
        evidence = str(item.get("evidence", "")).strip()
        located = matcher.find_loose(evidence)
        if located is None:
            continue  # unverifiable quote: intentionally dropped to avoid invented evidence
        local_start, local_end = located
        start = chunk["start"] + local_start
        end = chunk["start"] + local_end

        level_raw = _text(item.get("risk_level"), 12).lower()
        level = next((lv for lv in LEVELS if lv.lower() == level_raw), "Medium")
        try:
            severity = int(item.get("severity_score", 3))
        except (TypeError, ValueError):
            severity = 3
        low, high = SEVERITY_BAND[level]
        severity = max(low, min(high, severity))  # keep level and score consistent

        title = _text(item.get("title"), 160) or "Potential contract issue"
        explanation = _text(item.get("explanation"), 600)
        if not explanation:
            continue
        findings.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "category": _category(item.get("category")),
            "risk_level": level,
            "severity_score": severity,
            "evidence": _text(evidence, 400),
            "explanation": explanation,
            "recommendation": _text(item.get("recommendation"), 500),
            "page_start": doc.page_at(start),
            "page_end": doc.page_at(end),
            "chunk_index": chunk["index"],
            "char_start": start,
            "char_end": end,
        })
        if len(findings) >= config.MAX_FINDINGS_PER_CHUNK:
            break
    return findings


def _overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    inter = min(a["char_end"], b["char_end"]) - max(a["char_start"], b["char_start"])
    shortest = max(1, min(a["char_end"] - a["char_start"], b["char_end"] - b["char_start"]))
    return max(0.0, inter / shortest)


def _same_issue(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if _overlap_ratio(a, b) >= 0.5:
        return True
    if abs(a["char_start"] - b["char_start"]) <= 700 and a["category"] == b["category"]:
        return SequenceMatcher(None, chunking.squash(a["title"]), chunking.squash(b["title"])).ratio() >= 0.85
    return False


def dedupe(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlapping chunks see the same clause twice. Merge findings that point at the same text."""
    kept: list[dict[str, Any]] = []
    for finding in sorted(findings, key=lambda f: f["char_start"]):
        duplicate_of = None
        for existing in reversed(kept):
            if finding["char_start"] - existing["char_end"] > 1500:
                break
            if _same_issue(existing, finding):
                duplicate_of = existing
                break
        if duplicate_of is None:
            kept.append(finding)
            continue
        better = (finding["severity_score"], len(finding["explanation"])) > (duplicate_of["severity_score"], len(duplicate_of["explanation"]))
        if better:
            kept[kept.index(duplicate_of)] = finding
    return kept


def sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(findings, key=lambda f: (LEVEL_ORDER.get(f["risk_level"], 9), -f["severity_score"], f["char_start"]))


def risk_score(findings: list[dict[str, Any]], pages: int = 1) -> int:
    """A heuristic 0-100 indicator of how many/how serious the flagged issues are. Not a probability."""
    points = {"High": {4: 10, 5: 14}, "Medium": {3: 5}, "Low": {1: 1.5, 2: 2.5}, "Info": {1: 0}}
    raw = 0.0
    for finding in findings:
        level = finding.get("risk_level", "Info")
        table = points.get(level, {})
        raw += table.get(int(finding.get("severity_score", 1) or 1), max(table.values()) if table else 0)
    scale = 40 + 0.6 * max(1, pages)  # longer contracts naturally contain more flags
    return int(round(100 * (1 - math.exp(-raw / scale))))


def risk_label(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Elevated"
    if score >= 25:
        return "Moderate"
    return "Low"


# ----------------------------------------------------------------------------- summarising
def _str_list(value: Any, max_items: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for entry in value:
        if isinstance(entry, dict):
            entry = " - ".join(str(v) for v in entry.values() if v)
        text = _text(entry, max_len)
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def reduce_summaries(summaries: list[str], model: str, budget: int = SUMMARY_BUDGET) -> list[str]:
    """Collapse many chunk summaries (long contracts) so the final prompt always fits in the model's context."""
    current = [s for s in summaries if s]
    rounds = 0
    while sum(len(s) for s in current) > budget and len(current) > 1 and rounds < 4:
        merged = []
        for start in range(0, len(current), 10):
            group = current[start:start + 10]
            try:
                result = llm.chat_json(
                    MERGE_SYSTEM, json.dumps(group, ensure_ascii=False), model,
                    schema=MERGE_SCHEMA, num_predict=450, timeout=300, attempts=1,
                )
                text = _text(result.get("summary"), 700)
            except llm.LLMError:
                text = ""
            merged.append(text or _text(" ".join(item[:100] for item in group), 700))
        current = merged
        rounds += 1
    if sum(len(s) for s in current) > budget and current:
        per_item = max(80, budget // len(current))
        current = [s[:per_item] for s in current]
    return current


def synthesize(summaries: list[str], findings: list[dict[str, Any]], filename: str, model: str, perspective: str) -> dict[str, Any]:
    reduced = reduce_summaries(summaries, model)
    compact, used = [], 0
    for f in sort_findings(findings):
        entry = {
            "title": f["title"], "category": f["category"], "risk_level": f["risk_level"],
            "evidence": f["evidence"][:160], "recommendation": f["recommendation"][:160],
        }
        size = len(json.dumps(entry, ensure_ascii=False))
        if used + size > 5000 or len(compact) >= 25:
            break
        compact.append(entry)
        used += size
    prompt = f"""Contract file name: {filename}
Reviewing from the viewpoint of: {PERSPECTIVES.get(perspective, PERSPECTIVES['neutral'])}

Notes on each consecutive part of the contract (in order):
{json.dumps(reduced, ensure_ascii=False)}

Top findings already extracted ({len(findings)} in total; do not add new ones):
{json.dumps(compact, ensure_ascii=False)}

Return JSON with:
- executive_summary: 3-5 clear, factual, cautious sentences about what the contract is and its main points of concern
- key_obligations: up to 8 short strings, each supported by the notes
- reviewer_checklist: up to 8 concrete questions or actions for a human reviewer
- limitations: 2-4 honest limitations (include that this is not legal advice)"""
    return llm.chat_json(SYNTH_SYSTEM, prompt, model, schema=SYNTH_SCHEMA, num_predict=1400, timeout=config.LLM_TIMEOUT)


def fallback_synthesis(pages: int, chunks: int, findings: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "executive_summary": (
            f"LexiGuard reviewed {pages} page(s) in {chunks} part(s) and flagged {len(findings)} potential issue(s) for human review. "
            "The written summary could not be generated, so please rely on the findings below."
        ),
        "key_obligations": [],
        "reviewer_checklist": [
            "Check each quoted passage against the original contract.",
            "Read the full document: clauses can interact in ways an excerpt-by-excerpt review misses.",
            "Confirm governing law and commercial context with a qualified reviewer.",
        ],
        "limitations": [f"Final summary step failed: {reason[:160]}", "This report is not legal advice."],
    }


# ----------------------------------------------------------------------------- pipeline
def _check_cancel(contract_id: str) -> None:
    record = db.get(contract_id)
    if record is None or record.get("cancel_requested"):
        raise Cancelled()


def run_analysis(contract_id: str) -> None:
    record = db.get(contract_id)
    # Queue entries can outlive cancellation/retry operations. Only the current queued
    # state is allowed to claim a job; otherwise an old queue item could restart a
    # cancelled contract or run a duplicate analysis after a retry.
    if not record or record.get("status") != "queued":
        return
    model = record.get("model") or config.OLLAMA_MODEL
    perspective = record.get("perspective") or "neutral"
    try:
        db.update(contract_id, status="processing", progress=1, stage="Checking the local AI model", error=None, cancel_requested=0)
        llm.check_ready(model)

        db.update(contract_id, progress=3, stage="Extracting text")
        extracted = extract.extract(Path(record["file_path"]))
        doc = chunking.build_document(extracted.pages)
        chunks = chunking.make_chunks(doc, config.CHUNK_CHARS, config.CHUNK_OVERLAP)
        text_path = config.TEXT_DIR / f"{contract_id}.txt"
        text_path.write_text(doc.full_text, encoding="utf-8")
        passages = chunking.split_passages(doc, config.PASSAGE_CHARS, config.PASSAGE_OVERLAP)
        passages_path = config.TEXT_DIR / f"{contract_id}.passages.json"
        passages_path.write_text(json.dumps(passages, ensure_ascii=False), encoding="utf-8")
        db.update(
            contract_id, text_path=str(text_path), passages_path=str(passages_path), pages=doc.page_count, progress=6,
            stage=f"Read {doc.page_count} page(s); reviewing in {len(chunks)} part(s)",
        )

        system_prompt = chunk_system_prompt(perspective)
        findings: list[dict[str, Any]] = []
        summaries: list[str] = []
        errors: list[str] = []
        consecutive = 0
        ok_chunks = 0
        for position, chunk in enumerate(chunks, start=1):
            _check_cancel(contract_id)
            pages_label = (
                f"page {chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
                else f"pages {chunk['page_start']}-{chunk['page_end']}"
            )
            db.update(
                contract_id,
                progress=min(88, 6 + int(82 * (position - 1) / len(chunks))),
                stage=f"Reviewing part {position} of {len(chunks)} ({pages_label})",
            )
            try:
                result = llm.chat_json(
                    system_prompt,
                    f"This is part {position} of {len(chunks)} of the contract (approximate pages {chunk['page_start']}-{chunk['page_end']}).\n\n"
                    f"<contract_excerpt>\n{chunk['text']}\n</contract_excerpt>",
                    model, schema=CHUNK_SCHEMA,
                )
                summary = _text(result.get("chunk_summary"), 400)
                if summary:
                    summaries.append(summary)
                findings.extend(grounded_findings(result.get("findings", []), chunk, doc))
                ok_chunks += 1
                consecutive = 0
            except llm.LLMError as exc:
                errors.append(f"Part {position}: {str(exc)[:240]}")
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    raise llm.LLMError(f"Stopped after {MAX_CONSECUTIVE_FAILURES} failures in a row. Last error: {str(exc)[:240]}") from exc

        if ok_chunks == 0:
            raise llm.LLMError("The AI model could not analyse any part of the contract. " + (errors[0] if errors else ""))

        _check_cancel(contract_id)
        findings = sort_findings(dedupe(findings))
        db.update(contract_id, progress=90, stage="Writing the summary")
        try:
            synthesis = synthesize(summaries, findings, record["filename"], model, perspective)
        except llm.LLMError as exc:
            synthesis = fallback_synthesis(doc.page_count, len(chunks), findings, str(exc))

        counts = {level: sum(1 for f in findings if f["risk_level"] == level) for level in LEVELS}
        score = risk_score(findings, doc.page_count)
        limitations = _str_list(synthesis.get("limitations"), 4, 300)
        if "not legal advice" not in " ".join(limitations).lower():
            limitations.append("This report is not legal advice.")
        report = {
            "contract_id": contract_id,
            "filename": record["filename"],
            "pages": doc.page_count,
            "page_kind": extracted.page_kind,
            "created_at": db.now_iso(),
            "model": model,
            "perspective": perspective,
            "analysis_chunks": len(chunks),
            "chunks_analyzed": ok_chunks,
            "partial": ok_chunks < len(chunks),
            "risk_indicator": score,
            "risk_label": risk_label(score),
            "risk_indicator_note": (
                "A rule-of-thumb indicator based on the issues the model flagged. It is not a probability or a legal "
                "conclusion, and a low score does not prove a contract is safe."
            ),
            "counts": counts,
            "executive_summary": _text(synthesis.get("executive_summary"), 1500) or "No summary was generated.",
            "key_obligations": _str_list(synthesis.get("key_obligations"), 8, 300),
            "reviewer_checklist": _str_list(synthesis.get("reviewer_checklist"), 8, 300),
            "findings": findings,
            "extraction_notes": extracted.notes,
            "processing_notes": errors[:10],
            "limitations": limitations,
            "disclaimer": DISCLAIMER,
        }
        db.update(
            contract_id, status="completed", progress=100, stage="Analysis complete", error=None,
            report_json=json.dumps(report, ensure_ascii=False), risk_indicator=score, findings_count=len(findings),
            index_state="pending",
        )
    except Cancelled:
        db.update(contract_id, status="cancelled", progress=0, stage="Cancelled", error=None)
        return
    except (llm.LLMError, extract.ExtractionError) as exc:
        db.update(contract_id, status="failed", progress=100, stage="Analysis failed", error=str(exc))
        return
    except Exception as exc:  # last-resort guard so a bug never leaves a job stuck in 'processing'
        db.update(contract_id, status="failed", progress=100, stage="Analysis failed", error=f"Unexpected error: {type(exc).__name__}: {exc}")
        return

    run_indexing(contract_id)


def run_indexing(contract_id: str) -> None:
    """Best-effort semantic index for the 'Ask the contract' feature. Never affects the finished report."""
    record = db.get(contract_id)
    if not record or not record.get("passages_path"):
        return
    if not config.USE_QDRANT:
        db.update(contract_id, index_state="unavailable", index_note="Semantic search is turned off; keyword search is used instead.")
        return
    try:
        db.update(contract_id, index_state="building", index_note=None)
        passages = json.loads(Path(record["passages_path"]).read_text(encoding="utf-8"))
        vectors.index_passages(contract_id, passages)
        db.update(contract_id, index_state="ready", index_note=None)
    except Exception as exc:
        db.update(
            contract_id, index_state="unavailable",
            index_note=f"Semantic search is unavailable ({str(exc)[:200]}). Questions will use keyword search instead.",
        )
