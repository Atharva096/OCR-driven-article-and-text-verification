"""Vector storage for semantic search using ChromaDB."""
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger("vector_store")

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="news_articles",
            metadata={"hnsw:space": "cosine"}
        )
        # Load local embedding model (downloads on first run)
        log.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def add_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Embed and store articles. Skips if article_id already exists."""
        if not articles:
            return 0

        ids = [a["article_id"] for a in articles]
        # Check existing to avoid redundant embedding computations
        existing = self.collection.get(ids=ids)["ids"]
        new_ids = [i for i in ids if i not in existing]
        
        if not new_ids:
            return 0

        new_articles = [a for a in articles if a["article_id"] in new_ids]
        
        documents = [
            f"Title: {a.get('article_title')}\nSource: {a.get('article_source')}\nContent: {a.get('article_content')}"
            for a in new_articles
        ]
        embeddings = self.model.encode(documents, show_progress_bar=False).tolist()
        
        metadatas = [
            {
                "source": a.get("article_source", "unknown"),
                "date": a.get("date", "unknown"),
                "domain": a.get("domains", [{}])[0].get("domain", "unknown")
            }
            for a in new_articles
        ]

        self.collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        log.info("Added %d new articles to Vector Store", len(new_ids))
        return len(new_ids)

    def search(self, query: str, n_results: int = 5, filter_metadata: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_metadata
        )
        
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "article_id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        return formatted_results