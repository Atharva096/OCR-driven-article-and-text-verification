# News extractor (large PDFs)

Extracts news articles from big PDFs and images using Tesseract OCR and a free
OpenAI-compatible LLM endpoint. Results are stored in SQLite and exported to JSON.

## Setup

1. Install Tesseract OCR and Python 3.10+.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, pick a provider and add your key.

## Run

```
python extractor.py report.pdf
python extractor.py folder_of_pdfs/ --db news.db --out extracted_news.json
```

## How it handles big files

- Reads pages one at a time; OCR runs only on pages with little native text.
- Page text is cached in SQLite, so re-runs never repeat OCR.
- Chunks overlap slightly so articles cut by a boundary are still captured; duplicates
  are removed by a title+source key.
- LLM calls are limited by `CONCURRENCY` and `REQUESTS_PER_MINUTE`, with backoff on 429/5xx.
- Every chunk's status is saved. If it crashes or hits a daily limit, run the same
  command again and only unfinished chunks are processed.

## Querying results

```
sqlite3 news.db "SELECT title, source, date, domain FROM articles LIMIT 20;"
```
