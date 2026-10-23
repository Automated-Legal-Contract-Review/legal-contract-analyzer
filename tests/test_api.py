import time

import docx
import pytest

from conftest import OLLAMA, QDRANT
from helpers import make_pdf, upload, wait_done, wait_index


@pytest.fixture()
def pdf3(tmp_path):
    return make_pdf(tmp_path / "contract.pdf", pages=6)


def test_health(client):
    data = client.get("/api/health").json()
    assert data["ollama"]["connected"] and data["ollama"]["model_available"]
    assert data["qdrant"]["connected"]
    assert data["embedding_model"]["available"]
    assert "llama3.2:latest" in data["ollama"]["installed_models"]
    assert "nomic-embed-text:latest" not in data["ollama"]["installed_models"]


def test_index_page_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "LexiGuard" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_full_pdf_flow(client, pdf3):
    response = upload(client, pdf3, perspective="customer")
    assert response.status_code == 200, response.text
    cid = response.json()["contract_id"]
    status = wait_done(client, cid)
    assert status["status"] == "completed", status
    report = client.get(f"/api/contracts/{cid}/report").json()

    assert report["pages"] == 6 and report["page_kind"] == "exact"
    assert report["perspective"] == "customer"
    assert report["chunks_analyzed"] == report["analysis_chunks"] and not report["partial"]
    titles = {f["title"] for f in report["findings"]}
    assert {"Unlimited liability exposure", "Automatic renewal", "Broad indemnity", "Unilateral termination"} <= titles
    assert "Invented clause" not in titles, "hallucinated quote must be dropped"
    # severity ordering and exact page numbers (the liability clause sits on page 3)
    levels = [f["risk_level"] for f in report["findings"]]
    assert levels == sorted(levels, key=["High", "Medium", "Low", "Info"].index)
    liability = next(f for f in report["findings"] if f["category"] == "Liability")
    assert liability["page_start"] == 3
    renewal = next(f for f in report["findings"] if f["category"] == "Renewal")
    assert renewal["page_start"] == 4
    assert 0 < report["risk_indicator"] <= 100
    assert report["counts"]["High"] == sum(1 for f in report["findings"] if f["risk_level"] == "High")
    assert report["key_obligations"] == ["Supplier delivers services", "Customer pays invoices"]  # dict item coerced
    assert report["executive_summary"] and report["disclaimer"]
    # internal offsets are fine to keep, but ids must be unique
    assert len({f["id"] for f in report["findings"]}) == len(report["findings"])

    download = client.get(f"/api/contracts/{cid}/report.json")
    assert "attachment" in download.headers["content-disposition"]
    assert client.get(f"/api/contracts/{cid}/text").json()["text"].startswith("Page 1")
    assert any(item["id"] == cid and item["status"] == "completed" for item in client.get("/api/contracts").json())


def test_semantic_index_and_questions(client, pdf3):
    cid = upload(client, pdf3).json()["contract_id"]
    wait_done(client, cid)
    assert wait_index(client, cid) == "ready"
    assert any(name.endswith(cid.replace("-", "")) for name in QDRANT.collections)
    answer = client.post(f"/api/contracts/{cid}/ask", json={"question": "Does the agreement renew automatically?"}).json()
    assert answer["method"] == "semantic+keyword"
    assert "renewal" in answer["answer"].lower() or "renew" in answer["answer"].lower()
    assert answer["sources"] and answer["sources"][0]["page_start"] >= 1
    assert "automatically renew" in " ".join(s["snippet"] for s in answer["sources"]).lower()


def test_questions_fall_back_to_keywords_without_qdrant(client, pdf3):
    QDRANT.healthy = False
    cid = upload(client, pdf3).json()["contract_id"]
    status = wait_done(client, cid)
    assert status["status"] == "completed", "analysis must not depend on Qdrant"
    assert wait_index(client, cid) == "unavailable"
    assert "keyword" in client.get(f"/api/contracts/{cid}/status").json()["index_note"].lower()
    answer = client.post(f"/api/contracts/{cid}/ask", json={"question": "Who must indemnify whom?"}).json()
    assert answer["method"] == "keyword" and answer["sources"]


def test_long_contract_120_pages(client, tmp_path):
    big = make_pdf(tmp_path / "big.pdf", pages=120)
    cid = upload(client, big).json()["contract_id"]
    status = wait_done(client, cid, timeout=120)
    assert status["status"] == "completed", status
    report = client.get(f"/api/contracts/{cid}/report").json()
    assert report["pages"] == 120
    assert report["analysis_chunks"] > 20
    assert report["chunks_analyzed"] == report["analysis_chunks"]
    pages = sorted({f["page_start"] for f in report["findings"]})
    assert pages[0] <= 6 and pages[-1] >= 115, "findings should span the whole document"
    # overlap duplicates are merged: each of the 120 pages has exactly one trigger clause
    assert len(report["findings"]) <= 125
    assert max(f["page_start"] for f in report["findings"]) <= 120


def test_docx_flow(client, tmp_path):
    path = tmp_path / "c.docx"
    document = docx.Document()
    for i in range(40):
        document.add_paragraph(f"Clause {i}. The Supplier shall have unlimited liability for any loss arising from clause {i}. " * 2)
    document.save(path)
    cid = upload(client, path).json()["contract_id"]
    assert wait_done(client, cid)["status"] == "completed"
    report = client.get(f"/api/contracts/{cid}/report").json()
    assert report["page_kind"] == "approx" and report["findings"]
    assert any("estimates" in note for note in report["extraction_notes"])


def test_upload_validation(client, tmp_path):
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    assert upload(client, bad).status_code == 415
    empty = tmp_path / "e.txt"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 400
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not a pdf " * 50)
    cid = upload(client, fake_pdf).json()["contract_id"]
    status = wait_done(client, cid)
    assert status["status"] == "failed" and "not a valid PDF" in status["error"]


def test_upload_rejected_when_ollama_model_missing(client, pdf3):
    OLLAMA.models = ["nomic-embed-text:latest"]
    response = upload(client, pdf3)
    assert response.status_code == 503
    assert "ollama pull llama3.2" in response.json()["detail"]


def test_upload_rejected_when_ollama_offline(client, pdf3, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:9")
    response = upload(client, pdf3)
    assert response.status_code == 503 and "Ollama is not running" in response.json()["detail"]


def test_model_errors_fail_loudly_instead_of_empty_report(client, pdf3):
    OLLAMA.mode = "fail500"
    cid = upload(client, pdf3).json()["contract_id"]
    status = wait_done(client, cid)
    assert status["status"] == "failed", "must NOT report 'completed with zero findings' when the AI never worked"
    assert "boom" in status["error"] or "failures" in status["error"]
    assert client.get(f"/api/contracts/{cid}/report").status_code == 409


def test_bad_json_from_model_fails_cleanly(client, pdf3):
    OLLAMA.mode = "badjson"
    cid = upload(client, pdf3).json()["contract_id"]
    assert wait_done(client, cid)["status"] == "failed"


def test_truncated_json_is_salvaged(client, pdf3):
    OLLAMA.mode = "truncate"
    cid = upload(client, pdf3).json()["contract_id"]
    assert wait_done(client, cid)["status"] == "completed"
    assert client.get(f"/api/contracts/{cid}/report").json()["findings"]


def test_old_ollama_without_schema_support_still_works(client, pdf3):
    OLLAMA.mode = "schema400"
    cid = upload(client, pdf3).json()["contract_id"]
    assert wait_done(client, cid)["status"] == "completed"
    assert client.get(f"/api/contracts/{cid}/report").json()["findings"]


def test_retry_after_failure_then_delete(client, pdf3):
    OLLAMA.mode = "fail500"
    cid = upload(client, pdf3).json()["contract_id"]
    assert wait_done(client, cid)["status"] == "failed"
    OLLAMA.mode = "normal"
    assert client.post(f"/api/contracts/{cid}/retry").status_code == 200
    assert wait_done(client, cid)["status"] == "completed"
    assert client.post(f"/api/contracts/{cid}/retry").status_code == 409
    wait_index(client, cid)
    assert client.delete(f"/api/contracts/{cid}").status_code == 200
    assert client.get(f"/api/contracts/{cid}/status").status_code == 404
    assert not any(name.endswith(cid.replace("-", "")) for name in QDRANT.collections)


def test_cancel_running_analysis(client, tmp_path, monkeypatch):
    from app import llm
    original = llm.chat_json

    def slow(*args, **kwargs):
        time.sleep(0.25)
        return original(*args, **kwargs)

    monkeypatch.setattr(llm, "chat_json", slow)
    big = make_pdf(tmp_path / "big2.pdf", pages=30)
    cid = upload(client, big).json()["contract_id"]
    for _ in range(100):
        if client.get(f"/api/contracts/{cid}/status").json()["status"] == "processing":
            break
        time.sleep(0.05)
    assert client.delete(f"/api/contracts/{cid}").status_code == 409  # cannot delete while running
    assert client.post(f"/api/contracts/{cid}/cancel").status_code == 200
    assert wait_done(client, cid)["status"] == "cancelled"


def test_not_found_and_validation(client):
    assert client.get("/api/contracts/not-a-uuid/status").status_code == 404
    assert client.get("/api/contracts/00000000-0000-0000-0000-000000000000/report").status_code == 404
    assert client.post("/api/contracts/00000000-0000-0000-0000-000000000000/ask", json={"question": "hello there"}).status_code == 404


def test_ask_requires_finished_analysis_and_valid_question(client, pdf3):
    cid = upload(client, pdf3).json()["contract_id"]
    assert client.post(f"/api/contracts/{cid}/ask", json={"question": "x"}).status_code == 422
    wait_done(client, cid)
    assert client.post(f"/api/contracts/{cid}/ask", json={"question": "What is the weather on Mars?"}).status_code == 200


def test_recovery_marks_interrupted_jobs_failed(client, pdf3):
    from app import db, main
    cid = upload(client, pdf3).json()["contract_id"]
    wait_done(client, cid)
    db.update(cid, status="processing", progress=40, stage="Reviewing part 3")
    main.recover_jobs()
    status = client.get(f"/api/contracts/{cid}/status").json()
    assert status["status"] == "failed" and "Retry" in status["error"]


def test_old_database_is_migrated(tmp_path, monkeypatch):
    import sqlite3
    from app import config, db
    old = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(old)
    conn.execute(
        "CREATE TABLE contracts (id TEXT PRIMARY KEY, filename TEXT NOT NULL, file_path TEXT NOT NULL, text_path TEXT, pages INTEGER DEFAULT 0, "
        "status TEXT NOT NULL DEFAULT 'uploaded', progress INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT 'Uploaded', error TEXT, "
        "report_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO contracts (id, filename, file_path, created_at, updated_at) VALUES ('a','f','p','t','t')")
    conn.commit(); conn.close()
    monkeypatch.setattr(config, "DB_PATH", old)
    db.init_db()
    row = db.get("a")
    assert row["index_state"] == "none" and row["cancel_requested"] == 0 and row["perspective"] is None
