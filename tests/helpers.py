import time
from pathlib import Path

CLAUSES = [
    "The Supplier shall provide the services described in Schedule A to the Customer.",
    "The Customer shall pay each invoice within thirty (30) days of receipt of the invoice.",
    "The Supplier shall have unlimited liability for any loss arising under this Agreement.",
    "This Agreement will automatically renew for successive one-year terms unless cancelled.",
    "The Customer shall indemnify the Supplier against all claims of any kind whatsoever.",
    "Either party may terminate this Agreement at any time by written notice to the other.",
]


def make_pdf(path: Path, pages: int = 3) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page()
        body = f"Page {number} of the master services agreement.\n\n"
        # one distinctive clause per page, plus filler so pages have real text
        body += CLAUSES[(number - 1) % len(CLAUSES)] + "\n\n"
        body += "General terms apply to all work performed by the parties under this agreement. " * 6
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), body, fontsize=10)
    doc.save(path)
    doc.close()
    return path


def wait_done(client, contract_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = client.get(f"/api/contracts/{contract_id}/status").json()
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.1)
    raise AssertionError(f"timed out: {status}")


def wait_index(client, contract_id: str, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    state = ""
    while time.time() < deadline:
        state = client.get(f"/api/contracts/{contract_id}/status").json()["index_state"]
        if state in ("ready", "unavailable"):
            return state
        time.sleep(0.1)
    return state


def upload(client, path: Path, **form):
    with path.open("rb") as handle:
        return client.post("/api/contracts/upload", files={"file": (path.name, handle)}, data=form)
