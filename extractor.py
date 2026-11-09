"""
Scalable news-article extractor for large PDFs, images, and web URLs.

Pipeline
  1. Read PDFs page by page (fast native text; OCR only for pages with little text).
  2. Fetch web URLs and strip HTML to raw text.
  3. Cache every page's text in SQLite so a re-run never repeats OCR/fetching.
  4. Split into overlapping chunks and send them to an OpenAI-compatible LLM
     endpoint with bounded concurrency, a requests-per-minute limiter and retries.
  5. Save deduplicated articles in SQLite. Failed chunks are retried on the next run.
  6. (Optional) Sync extracted articles to ChromaDB (Vector) and Neo4j (Graph).
  7. Export everything to JSON in the original schema.

Usage
  python extractor.py report.pdf
  python extractor.py folder_of_pdfs/ --out news.json --db news.db
  python extractor.py https://example.com/news --enable-vector --enable-graph
"""

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
import pymupdf  # Replaced deprecated 'fitz'
import pytesseract
from PIL import Image

from storage import Store

try:  # optional: load settings from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------- config
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "1") == "1"

CONCURRENCY = int(os.getenv("CONCURRENCY", "3"))  # simultaneous LLM calls
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "20"))  # 0 = unlimited
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "6000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))

MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "50"))  # below this a page gets OCR
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_WORKERS = int(os.getenv("OCR_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))

DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN") or None  # used when the LLM gives no sector

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif")
TEXT_EXTS = (".html", ".htm", ".txt", ".md")

log = logging.getLogger("extractor")


# --------------------------------------------------------------------------- validation
def validate_config() -> List[str]:
    """Validate configuration and return list of warnings."""
    warnings = []
    if CONCURRENCY < 1:
        raise ValueError("CONCURRENCY must be at least 1")
    if MAX_RETRIES < 0:
        raise ValueError("MAX_RETRIES cannot be negative")
    if CHUNK_CHARS < 100:
        raise ValueError("CHUNK_CHARS should be at least 100")
    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative")
    if CHUNK_OVERLAP >= CHUNK_CHARS:
        raise ValueError("CHUNK_OVERLAP must be less than CHUNK_CHARS")
    
    if not LLM_API_KEY and "localhost" not in LLM_BASE_URL and "127.0.0.1" not in LLM_BASE_URL:
        warnings.append("LLM_API_KEY is not set. API calls may fail.")
    return warnings


# --------------------------------------------------------------------------- reading
def file_sha256(path: str) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ocr_pdf_page(pdf_path: str, page_no: int) -> str:
    """Render one page and OCR it. Runs in a worker thread."""
    try:
        with pymupdf.open(pdf_path) as doc:
            if page_no >= len(doc):
                log.warning("Page %d out of range (total: %d)", page_no, len(doc))
                return ""
            pix = doc[page_no].get_pixmap(dpi=OCR_DPI)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(image)
    except Exception as e:
        log.error("OCR failed on page %d of %s: %s", page_no, pdf_path, e)
        raise


def ocr_image_file(image_path: str) -> str:
    """Extract text from an image file using OCR."""
    try:
        with Image.open(image_path) as image:
            return pytesseract.image_to_string(image)
    except Exception as e:
        log.error("OCR failed on %s: %s", image_path, e)
        raise


async def load_pdf_pages(store: Store, path: str, file_hash: str, force: bool = False) -> List[str]:
    """Return the text of every page, using the cache and OCR-ing only when needed."""
    if force:
        cached = {}  # Ignore cache completely
        log.info("Force mode enabled: ignoring page cache for %s", Path(path).name)
    else:
        cached = store.get_pages(file_hash)

    try:
        with pymupdf.open(path) as doc:
            total = len(doc)
            native = {i: doc[i].get_text() for i in range(total) if i not in cached}
    except Exception as e:
        log.error("Failed to open PDF %s: %s", path, e)
        raise

    ready = {i: t for i, t in native.items() if len(t.strip()) >= MIN_TEXT_CHARS}
    need_ocr = [i for i in native if i not in ready]
    store.save_pages(file_hash, ready)
    cached.update(ready)
    log.info(
        "%s: %d pages (%d cached, %d native text, %d need OCR)",
        Path(path).name, total, total - len(native), len(ready), len(need_ocr),
    )

    if not need_ocr:
        return [cached.get(i, "") for i in range(total)]

    sem = asyncio.Semaphore(OCR_WORKERS)
    done = 0
    failed_pages: Set[int] = set()

    async def run(i: int) -> None:
        nonlocal done
        async with sem:
            try:
                text = await asyncio.to_thread(ocr_pdf_page, path, i)
            except Exception as e:
                log.warning("OCR failed on page %d: %s", i + 1, e)
                text = native.get(i, "")
                failed_pages.add(i)
        store.save_pages(file_hash, {i: text})
        cached[i] = text
        done += 1
        if done % 10 == 0 or done == len(need_ocr):
            log.info("OCR progress: %d/%d pages", done, len(need_ocr))

    await asyncio.gather(*(run(i) for i in need_ocr))
    
    if failed_pages:
        log.warning("%d pages failed OCR: %s", len(failed_pages), sorted(failed_pages)[:10])
    
    return [cached.get(i, "") for i in range(total)]


async def load_image_pages(store: Store, path: str, file_hash: str, force: bool = False) -> List[str]:
    """Load text from an image file."""
    if force:
        cached = {}
    else:
        cached = store.get_pages(file_hash)
    if 0 not in cached:
        try:
            text = await asyncio.to_thread(ocr_image_file, path)
            store.save_pages(file_hash, {0: text})
            cached[0] = text
        except Exception as e:
            log.error("Failed to process image %s: %s", path, e)
            cached[0] = ""
    return [cached[0]]


# --------------------------------------------------------------------------- chunking
def build_chunks(pages: List[str], size: int, overlap: int) -> List[str]:
    """Split page text into chunks of about `size` characters on line boundaries."""
    if not pages:
        return []
    
    lines = []
    for page in pages:
        if not page:
            continue
        for line in page.splitlines():
            line = line.rstrip()
            if not line:
                continue
            while len(line) > size:
                lines.append(line[:size])
                line = line[size:]
            lines.append(line)

    if not lines:
        return []

    chunks, current, current_len = [], [], 0
    for line in lines:
        add = len(line) + 1
        if current and current_len + add > size:
            chunk_text = "\n".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            carry, total = [], 0
            for prev in reversed(current):
                if total + len(prev) + 1 > overlap:
                    break
                carry.append(prev)
                total += len(prev) + 1
            carry.reverse()
            current, current_len = carry, total
        
        current.append(line)
        current_len += add
    
    if current:
        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    return [c for c in chunks if len(c) > 40]


# --------------------------------------------------------------------------- LLM
SYSTEM_PROMPT = (
    "You extract news articles from messy OCR/PDF/HTML text. "
    "Reply with a single JSON object and nothing else."
)

USER_PROMPT = """Extract the news articles from the messy OCR text below.

Return exactly this JSON shape:
{{"articles": [
  {{"title": "[headline]", "content": "[coherent body text]", "source": "[publisher]",
    "date": "[date]", "domain": "[sector, e.g., Politics, Defence, Business]", 
    "region": "[geographic region, e.g., India, Middle East, Global]", 
    "event": "[specific event, e.g., 'Diplomatic Meeting', 'Policy Change']"}}
]}}

RULES:
1. IGNORE newspaper mastheads, page numbers, weather, TV listings, or fragmented garbage text. Only extract actual, coherent news articles.
2. If the OCR text is too fragmented to form a coherent article, SKIP IT. Do not force an extraction.
3. For valid articles, INFER the 'region' and 'event' based on the content if not explicitly stated. NEVER return null for these; make a logical best guess.
4. Fix obvious OCR typos in the content (e.g., 'PMmeets' -> 'PM meets') for basic readability, but do NOT summarize or rewrite the core facts.
5. If there are no valid, coherent articles in this chunk, return {{"articles": []}}.

Text:
{chunk}"""


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass


class RateLimiter:
    """Spaces requests out so we stay under a requests-per-minute cap."""
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = max(self._next, now) + self.interval


class LLMClient:
    """Client for making LLM API calls with rate limiting and retries."""
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.limiter = RateLimiter(REQUESTS_PER_MINUTE)
        self.json_mode = LLM_JSON_MODE
        self.url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            self.headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    async def complete(self, chunk: str) -> str:
        """Send a chunk to the LLM and return the response."""
        last_error = "unknown error"
        
        for attempt in range(MAX_RETRIES):
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(chunk=chunk)},
                ],
                "temperature": 0.1,
                "max_tokens": MAX_TOKENS,
            }
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}

            delay = min(60.0, (2 ** attempt) + random.random())
            
            async with self.sem:
                await self.limiter.wait()
                try:
                    async with self.session.post(
                        self.url, json=payload, headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if content:
                                return content
                            last_error = "Empty response from LLM"
                        else:
                            body = await resp.text()
                            last_error = f"HTTP {resp.status}: {body[:300]}"
                            
                            if resp.status == 400 and self.json_mode and "response_format" in body:
                                log.warning("Endpoint rejects JSON mode; continuing without it.")
                                self.json_mode = False
                                delay = 0
                            elif resp.status in (408, 409, 425, 429, 500, 502, 503, 504):
                                try:
                                    retry_after = float(resp.headers.get("Retry-After", 0))
                                    delay = max(delay, retry_after)
                                except (ValueError, TypeError):
                                    pass
                            elif resp.status in (401, 403):
                                raise LLMError(f"Authentication failed: {last_error}")
                            elif resp.status == 422:
                                raise LLMError(f"Invalid request: {last_error}")
                            else:
                                raise LLMError(last_error)
                                
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = f"{type(e).__name__}: {e}"
                except json.JSONDecodeError as e:
                    last_error = f"Invalid JSON response: {e}"
                    raise LLMError(last_error)
                    
            if attempt < MAX_RETRIES - 1:
                log.info("LLM retry %d/%d in %.1fs (%s)", attempt + 1, MAX_RETRIES, delay, last_error[:120])
                await asyncio.sleep(delay)
                
        raise LLMError(f"gave up after {MAX_RETRIES} attempts: {last_error}")


def parse_articles(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a list of articles."""
    if not raw:
        return []
    
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    if not raw:
        return []
    
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("Failed to parse JSON: %s. Raw: %s", e, raw[:200])
        match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        if not match:
            log.error("No JSON found in response")
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e2:
            log.error("Failed to extract JSON: %s", e2)
            return []
    
    if isinstance(data, dict):
        if isinstance(data.get("articles"), list):
            data = data["articles"]
        elif "title" in data or "article_title" in data:
            data = [data]
        else:
            log.warning("Unexpected JSON structure: %s", list(data.keys()))
            data = []
    elif isinstance(data, list):
        pass
    else:
        log.warning("Unexpected data type: %s", type(data))
        return []
    
    return [a for a in data if isinstance(a, dict)]


def format_article(a: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format an article into the standard schema, blocking null/garbage."""
    if not isinstance(a, dict):
        return None
        
    title = str(a.get("title") or a.get("article_title") or "").strip()
    content = str(a.get("content") or a.get("article_content") or "").strip()
    
    # Block mastheads or fragments that are too short to be real articles
    if len(title) < 10 or len(content) < 30:
        return None
    
    # Handle domain extraction safely
    domain = a.get("domain")
    if not domain:
        domains = a.get("domains")
        if isinstance(domains, list) and domains:
            first_domain = domains[0]
            if isinstance(first_domain, dict):
                domain = first_domain.get("domain")
    
    # Use 'or' to catch BOTH missing keys AND explicit 'null' values from the LLM
    region = a.get("region") or "Unknown"
    event = a.get("event") or "General News"
    source = a.get("source") or a.get("article_source") or "Unknown"
    date = a.get("date") or "Unknown"
    
    return {
        "article_title": title,
        "article_id": str(uuid.uuid4()),
        "article_content": content,
        "article_source": source,
        "date": date,
        "region": region,
        "event": event,
        "domains": [
            {
                "companies": [
                    {
                        "company": None,
                        "company_id": None,
                        "company_desc": None,
                        "isIndependent": False,
                    }
                ],
                "domain": domain or DEFAULT_DOMAIN or "General",
                "domain_id": str(uuid.uuid4()),
                "domain_desc": None,
                "isIndependent": True,
            }
        ],
    }


# --------------------------------------------------------------------------- pipeline
async def process_chunk(client: LLMClient, store: Store, file_hash: str, idx: int, chunk: str) -> int:
    """Process a single chunk and store extracted articles."""
    try:
        raw = await client.complete(chunk)
        articles = [x for x in map(format_article, parse_articles(raw)) if x]
        added = store.add_articles(file_hash, idx, articles)
        store.mark_chunk(file_hash, idx, "done")
        return added
    except LLMError as e:
        log.error("Chunk %d failed (LLM): %s", idx, e)
        store.mark_chunk(file_hash, idx, "failed", str(e)[:500])
        return -1
    except Exception as e:
        log.exception("Chunk %d failed (unexpected): %s", idx, e)
        store.mark_chunk(file_hash, idx, "failed", str(e)[:500])
        return -1


async def process_source(store: Store, client: LLMClient, session: aiohttp.ClientSession, source: str, force: bool = False) -> str:
    """Process one file or URL. Returns its hash. Safe to re-run: finished work is skipped."""
    is_url = source.startswith("http://") or source.startswith("https://")
    
    if is_url:
        file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        store.register_file(file_hash, source)
        log.info("Fetching web content: %s", source)
        
        try:
            async with session.get(source, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                html = await resp.text()
                html = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html, flags=re.IGNORECASE)
                html = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', html, flags=re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text).strip()
                pages = [text]
        except Exception as e:
            log.error("Failed to fetch URL %s: %s", source, e)
            store.finish_file(file_hash, "failed", 0)
            return file_hash
    else:
        file_hash = file_sha256(source)
        store.register_file(file_hash, source)
        log.info("Processing %s", Path(source).name)

        if source.lower().endswith(".pdf"):
            pages = await load_pdf_pages(store, source, file_hash, force=force)
        elif source.lower().endswith(IMAGE_EXTS):
            pages = await load_image_pages(store, source, file_hash, force=force)
        else:
            # Handle local text/HTML files
            try:
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                # Strip HTML tags if it's an HTML file
                if source.lower().endswith((".html", ".htm")):
                    text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                pages = [text]
            except Exception as e:
                log.error("Failed to read text file %s: %s", source, e)
                store.finish_file(file_hash, "failed", 0)
                return file_hash

    chunks = build_chunks(pages, CHUNK_CHARS, CHUNK_OVERLAP)
    
    if not chunks:
        log.warning("No chunks generated from %s", source)
        store.finish_file(file_hash, "done", 0)
        return file_hash

    status = store.chunk_status(file_hash)
    pending = [i for i in range(len(chunks)) if status.get(i) != "done"]
    log.info("%s: %d chunks, %d to process", source, len(chunks), len(pending))

    if not pending:
        log.info("%s: all chunks already processed", source)
        store.finish_file(file_hash, "done", len(chunks))
        return file_hash

    tasks = [process_chunk(client, store, file_hash, i, chunks[i]) for i in pending]
    failed = new_articles = 0
    
    for n, fut in enumerate(asyncio.as_completed(tasks), 1):
        try:
            added = await fut
            if added < 0:
                failed += 1
            else:
                new_articles += added
        except Exception as e:
            log.error("Task failed: %s", e)
            failed += 1
            
        if n % 5 == 0 or n == len(tasks):
            log.info("LLM progress: %d/%d chunks, %d new articles, %d failed", n, len(tasks), new_articles, failed)

    final_status = "partial" if failed else "done"
    store.finish_file(file_hash, final_status, len(chunks))
    
    if failed:
        log.warning("%s: %d chunks failed; re-run the same command to retry them.", source, failed)
    
    return file_hash


def collect_inputs(inputs: List[str]) -> List[str]:
    """Collect all valid input files or URLs from the provided paths."""
    items = []
    for item in inputs:
        if item.startswith("http://") or item.startswith("https://"):
            items.append(item)
        else:
            p = Path(item)
            if p.is_dir():
                found = [
                    str(f) for f in sorted(p.rglob("*"))
                    if f.is_file() and (
                        f.suffix.lower() == ".pdf" or 
                        f.suffix.lower() in IMAGE_EXTS or 
                        f.suffix.lower() in TEXT_EXTS
                    )
                ]
                log.info("Found %d files in %s", len(found), p)
                items.extend(found)
            elif p.exists():
                if p.is_file() and (
                    p.suffix.lower() == ".pdf" or 
                    p.suffix.lower() in IMAGE_EXTS or 
                    p.suffix.lower() in TEXT_EXTS
                ):
                    items.append(str(p))
                else:
                    log.warning("Skipping unsupported file: %s", item)
            else:
                log.warning("Not found: %s", item)
    return items


async def run(inputs: List[str], db_path: str, out_path: str, force: bool = False, enable_vector: bool = False, enable_graph: bool = False) -> None:
    """Main processing pipeline."""
    sources = collect_inputs(inputs)
    if not sources:
        log.error("No supported files or URLs found.")
        return

    log.info("Processing %d sources", len(sources))
    
    store = Store(db_path)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=max(CONCURRENCY, 1) * 2, ttl_dns_cache=300)
    hashes = []
    
    # Initialize optional Phase 1 stores
    vector_store = None
    graph_store = None
    
    if enable_vector:
        try:
            from vector_store import VectorStore
            vector_store = VectorStore()
            log.info("Vector store (ChromaDB) initialized.")
        except ImportError:
            log.warning("Vector store enabled but 'vector_store.py' not found. Skipping.")
            
    if enable_graph:
        try:
            from graph_store import GraphStore
            graph_store = GraphStore(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password")
            )
            log.info("Graph store (Neo4j) initialized.")
        except ImportError:
            log.warning("Graph store enabled but 'graph_store.py' not found. Skipping.")
    
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            client = LLMClient(session)
            for i, source in enumerate(sources, 1):
                log.info("Source %d/%d: %s", i, len(sources), source)
                try:
                    file_hash = await process_source(store, client, session, source, force=force)
                    hashes.append(file_hash)
                except Exception as e:
                    log.exception("Failed to process %s: %s", source, e)
    finally:
        await connector.close()

    # Export to JSON
    count = store.export_json(out_path, hashes)
    log.info("Exported %d articles to %s (database: %s)", count, out_path, db_path)
    
    # Export to Vector and Graph stores
    if vector_store or graph_store:
        log.info("Fetching processed articles for advanced storage...")
        all_articles = store.get_all_articles(hashes)
        
        if vector_store and all_articles:
            v_count = vector_store.add_articles(all_articles)
            log.info("Added %d articles to Vector Store", v_count)
            
        if graph_store and all_articles:
            g_count = graph_store.ingest_articles(all_articles)
            log.info("Ingested %d articles to Graph Store", g_count)
            graph_store.close()

    store.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Extract news articles from PDFs/images/URLs.")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-process all pages/chunks")
    parser.add_argument("inputs", nargs="+", help="PDF/image files, folders, or HTTP URLs")
    parser.add_argument("--db", default="news.db", help="SQLite database (default: news.db)")
    parser.add_argument("--out", default="extracted_news.json", help="JSON export path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--enable-vector", action="store_true", help="Enable ChromaDB vector storage")
    parser.add_argument("--enable-graph", action="store_true", help="Enable Neo4j graph storage")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    
    try:
        warnings = validate_config()
        for warning in warnings:
            log.warning(warning)
    except ValueError as e:
        log.error("Configuration error: %s", e)
        sys.exit(1)

    if not LLM_API_KEY and "localhost" not in LLM_BASE_URL and "127.0.0.1" not in LLM_BASE_URL:
        log.error("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    try:
        asyncio.run(run(args.inputs, args.db, args.out, args.force, args.enable_vector, args.enable_graph))
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()