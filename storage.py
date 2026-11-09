"""SQLite storage: page cache, per-chunk progress (for resume), deduplicated articles."""

import hashlib
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_hash    TEXT PRIMARY KEY,
    path         TEXT,
    status       TEXT DEFAULT 'processing',
    total_chunks INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Extracted/OCR'd text per page, so a re-run never repeats OCR.
CREATE TABLE IF NOT EXISTS pages (
    file_hash TEXT,
    page_no   INTEGER,
    text      TEXT,
    PRIMARY KEY (file_hash, page_no)
);

-- One row per LLM chunk; only 'done' chunks are skipped on resume.
CREATE TABLE IF NOT EXISTS chunks (
    file_hash TEXT,
    idx       INTEGER,
    status    TEXT DEFAULT 'pending',
    error     TEXT,
    PRIMARY KEY (file_hash, idx)
);

CREATE TABLE IF NOT EXISTS articles (
    article_id  TEXT PRIMARY KEY,
    dedup_key   TEXT UNIQUE,
    file_hash   TEXT,
    chunk_idx   INTEGER,
    title       TEXT,
    source      TEXT,
    date        TEXT,
    domain      TEXT,
    record_json TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_articles_file ON articles(file_hash);
CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title);
CREATE INDEX IF NOT EXISTS idx_articles_dedup ON articles(dedup_key);
"""


def dedup_key(article: Dict[str, Any]) -> str:
    """Generate a stable key so overlap between chunks and re-uploads don't create duplicates."""
    title = str(article.get("article_title") or "").strip().lower()
    source = str(article.get("article_source") or "").strip().lower()
    
    if title:
        basis = f"{title}|{source}"
    else:
        content = str(article.get("article_content") or "")
        basis = content[:300].lower()
        
    basis = re.sub(r"\s+", " ", basis)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str = "news.db"):
        self.path = path
        # check_same_thread=False is safe here because we use 'with self.conn:' 
        # for all writes, which serializes transactions.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        
        # Optimize for concurrent writes and performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=10000")  # Wait up to 10s if DB is locked
        self.conn.execute("PRAGMA cache_size=-10000")   # 10MB cache
        
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ---- files -------------------------------------------------------------
    def register_file(self, file_hash: str, path: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO files (file_hash, path) VALUES (?, ?)",
                (file_hash, path),
            )

    def finish_file(self, file_hash: str, status: str, total_chunks: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE files SET status = ?, total_chunks = ? WHERE file_hash = ?",
                (status, total_chunks, file_hash),
            )

    # ---- pages -------------------------------------------------------------
    def get_pages(self, file_hash: str) -> Dict[int, str]:
        rows = self.conn.execute(
            "SELECT page_no, text FROM pages WHERE file_hash = ?", (file_hash,)
        ).fetchall()
        return {n: t for n, t in rows}

    def save_pages(self, file_hash: str, pages: Dict[int, str]) -> None:
        if not pages:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO pages (file_hash, page_no, text) VALUES (?, ?, ?)",
                [(file_hash, n, t) for n, t in pages.items()],
            )

    # ---- chunks ------------------------------------------------------------
    def chunk_status(self, file_hash: str) -> Dict[int, str]:
        rows = self.conn.execute(
            "SELECT idx, status FROM chunks WHERE file_hash = ?", (file_hash,)
        ).fetchall()
        return {i: s for i, s in rows}

    def mark_chunk(self, file_hash: str, idx: int, status: str, error: Optional[str] = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks (file_hash, idx, status, error) VALUES (?, ?, ?, ?)",
                (file_hash, idx, status, error),
            )

    # ---- articles ----------------------------------------------------------
    def add_articles(self, file_hash: str, chunk_idx: int, articles: List[Dict[str, Any]]) -> int:
        """Insert articles in bulk, silently skipping duplicates. Returns number inserted."""
        if not articles:
            return 0
        
        data_to_insert = []
        for a in articles:
            # Safely extract domain from the nested structure
            domains = a.get("domains")
            domain_val = None
            if isinstance(domains, list) and len(domains) > 0:
                first_domain = domains[0]
                if isinstance(first_domain, dict):
                    domain_val = first_domain.get("domain")
            
            data_to_insert.append((
                a.get("article_id"),
                dedup_key(a),
                file_hash,
                chunk_idx,
                a.get("article_title"),
                a.get("article_source"),
                a.get("date"),
                domain_val,
                json.dumps(a, ensure_ascii=False),
            ))
        
        with self.conn:
            cur = self.conn.executemany(
                """INSERT OR IGNORE INTO articles
                   (article_id, dedup_key, file_hash, chunk_idx, title, source, date, domain, record_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                data_to_insert
            )
        # executemany rowcount returns the total number of rows affected (inserted)
        return cur.rowcount

    def get_all_articles(self, file_hashes: List[str]) -> List[Dict[str, Any]]:
        """Retrieve all processed articles for the given file hashes (for Vector/Graph sync)."""
        if not file_hashes:
            return []
        marks = ",".join("?" * len(file_hashes))
        rows = self.conn.execute(
            f"SELECT record_json FROM articles WHERE file_hash IN ({marks})",
            file_hashes,
        ).fetchall()
        return [json.loads(r[0]) for r in rows if r[0]]

    def export_json(self, out_path: str, file_hashes: Optional[List[str]] = None) -> int:
        if file_hashes is not None:
            if not file_hashes:
                return 0  # Avoid invalid SQL: "IN ()"
            marks = ",".join("?" * len(file_hashes))
            rows = self.conn.execute(
                f"SELECT record_json FROM articles WHERE file_hash IN ({marks}) ORDER BY created_at, rowid",
                file_hashes,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT record_json FROM articles ORDER BY created_at, rowid"
            ).fetchall()
        
        # Filter out any potential nulls and parse JSON
        records = [json.loads(r[0]) for r in rows if r[0]]
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            
        return len(records)

    def close(self) -> None:
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()