"""Graph storage for knowledge representation using Neo4j."""
import logging
from typing import List, Dict, Any

from neo4j import GraphDatabase

log = logging.getLogger("graph_store")

class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connection
        self.driver.verify_connectivity()
        log.info("Connected to Neo4j at %s", uri)

    def close(self):
        self.driver.close()

    def ingest_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Ingest articles, creating nodes and relationships."""
        ingested_count = 0
        with self.driver.session() as session:
            for a in articles:
                try:
                    session.execute_write(self._create_article_graph, a)
                    ingested_count += 1
                except Exception as e:
                    log.warning("Failed to ingest article %s to Neo4j: %s", a.get("article_id"), e)
        return ingested_count

    @staticmethod
    def _create_article_graph(tx, article: Dict[str, Any]):
        """Cypher query to merge article and its entities."""
        article_id = article.get("article_id")
        title = article.get("article_title", "Unknown")
        source = article.get("article_source", "Unknown")
        date = article.get("date", "Unknown")
        content = article.get("article_content", "")[:1000]  # Truncate for graph preview
        
        # Extract domain/company info (adapt based on your LLM output)
        domains = article.get("domains", [])
        domain_name = domains[0].get("domain", "General") if domains else "General"
        companies = [c.get("company") for d in domains for c in d.get("companies", []) if c.get("company")]

        # Extract new fields if your LLM prompt is updated to include them
        region = article.get("region", "Unknown")
        event = article.get("event", "General News")

        query = """
        MERGE (a:Article {id: $article_id})
        ON CREATE SET a.title = $title, a.source = $source, a.date = $date, a.content_preview = $content
        
        MERGE (d:Domain {name: $domain})
        MERGE (a)-[:BELONGS_TO]->(d)
        
        MERGE (r:Region {name: $region})
        MERGE (a)-[:OCCURRED_IN]->(r)
        
        MERGE (e:Event {name: $event})
        MERGE (a)-[:REPORTS_ON]->(e)
        
        WITH a, $companies AS comps
        UNWIND comps AS compName
        MERGE (c:Company {name: compName})
        MERGE (a)-[:MENTIONS]->(c)
        MERGE (c)-[:OPERATES_IN]->(d)
        """
        tx.run(query, 
               article_id=article_id, title=title, source=source, date=date, content=content,
               domain=domain_name, region=region, event=event, companies=companies)