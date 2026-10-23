# Changelog

## Local improvements (2026-10-22)

- Prevented stale background-queue entries from restarting cancelled analyses or running an already completed analysis a second time. This can occur when a queued analysis is cancelled or retried while its original queue entry is still present.
- Made scanned-PDF OCR more resilient: a failure to OCR one page no longer stops OCR attempts for every remaining page.
- Added regression tests for stale cancelled/completed jobs.

## Validation performed

- `pytest -q`: 39 passed.
- `python -m compileall -q app tests`: passed.
- Inline JavaScript syntax check with Node.js: passed.

The automated suite uses fake local Ollama and Qdrant services. It validates the application pipeline and fallback behavior, but does not establish the quality of legal analysis from a real model or the availability of optional local OCR/Qdrant on every machine.
