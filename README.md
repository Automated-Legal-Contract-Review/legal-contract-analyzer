# LexiGuard Local

Review contracts (PDF, Word, text; 100+ pages is fine) with an AI model that runs **on your own computer**.
You get one report with a risk indicator, exact quotes, page references, explanations and suggested questions,
plus an "Ask a question" box for the contract.

Stack: Python, FastAPI, Ollama (llama3.2), optional Qdrant in Docker.

## Quick start (Windows)

1. Install [Python 3.10+](https://www.python.org/downloads/) (tick **Add python.exe to PATH**) and [Ollama](https://ollama.com/download).
2. Extract this folder, for example to `C:\Users\<you>\LexiGuard-Local`.
3. **Double-click `run.bat`.** (Or in PowerShell: `.\run.ps1`.)

The script installs packages, makes sure the AI model is downloaded, starts Qdrant if Docker is running,
starts the app and opens your browser once it is ready. Keep the window open while you use it; Ctrl+C stops it.

`run.bat` avoids the "running scripts is disabled" error. Options: `.\run.ps1 -SkipDocker`, `-Model qwen2.5:7b`, `-Port 8010`.

## What is optional

| Part | Needed? | If missing |
|---|---|---|
| Ollama + model | Yes | Upload is refused with a message saying what to run |
| Docker + Qdrant | No | Reviews work; questions use keyword search |
| `nomic-embed-text` | No | Same as above |
| Tesseract OCR | No | Scanned PDFs are rejected with a clear message. Install with `winget install UB-Mannheim.TesseractOCR` to read them |

## How a review works

1. Text is extracted (PDF pages, Word paragraphs **and tables**, text in UTF-8/UTF-16/Windows encodings).
2. The text is cut into overlapping parts at paragraph boundaries and each part is reviewed by the model.
3. **Every quote is checked against your document.** Findings whose quote cannot be found are discarded, so invented evidence never reaches the report. Page numbers come from where the quote was actually found.
4. Overlap duplicates are merged, a risk indicator is computed, and a summary is written from the findings.
5. If the AI fails on all parts the review is marked **failed** with the reason. If it fails on some, the report says so.

The risk indicator is a rule of thumb from the flagged issues, not a legal conclusion.

## Improving quality

`llama3.2` (3B) is small and fast. For more accurate reviews install a larger model, for example
`ollama pull llama3.1:8b` or `ollama pull qwen2.5:7b`, then pick it in the app (a model menu appears when several are installed).
You can also lower `CHUNK_CHARS` in `.env` (copy `.env.example`) for small models, or raise it for big ones.
Long contracts take time: expect roughly 20 to 90 seconds per part on a CPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama is not running" | Open the Ollama app, or run `ollama serve` |
| "model ... is not installed" | `ollama pull llama3.2` |
| "Port 8000 is used" | `.\run.ps1 -Port 8010` |
| Analysis stops with timeouts | Use a smaller model or lower `CHUNK_CHARS`; raise `LLM_TIMEOUT_SECONDS` |
| Closed the window mid-review | Start again; the review shows as interrupted, click **Try again** |
| Reset everything | Close the app and delete the `data` folder |

## Privacy

The app listens on `127.0.0.1` only, Qdrant is bound to this computer only, and contracts are stored in `data/`.
The only downloads are software, packages and models during setup. Use the delete button next to a review to remove its files.

## Development

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
```
The tests run the whole pipeline against fake Ollama and Qdrant servers (upload, extraction, 120-page contract, failures, cancel, retry, questions).
API docs are at http://127.0.0.1:8000/docs while the app runs.

## Disclaimer

LexiGuard is an AI-assisted issue-spotting tool. It is not a lawyer and can miss or misread things.
Have important contracts reviewed by a qualified legal professional.
