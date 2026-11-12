"""Graph storage for knowledge representation using Neo4j."""
import logging
from typing import List, Dict, Any, Tuple

from neo4j import GraphDatabase

log = logging.getLogger("graph_store")

class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connection
        self.driver.verify_connectivity()
        log.info("Connected to Neo4j at %s", uri)

    def get_schema(self):
        """Return node labels, relationship types, AND the real connection patterns."""
        with self.driver.session() as session:
            labels = [r[0] for r in session.run("CALL db.labels()")]
            rels = [r[0] for r in session.run("CALL db.relationshipTypes()")]
            patterns = [
                f"({p['src']})-[:{p['rel']}]->({p['tgt']})"
                for p in session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN DISTINCT labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt
                    LIMIT 50
                """)
            ]
            return labels, rels, patterns

    def execute_query(self, cypher_query: str):
        """Execute a raw Cypher query and return results as a list of dictionaries."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                return [dict(record) for record in result]
        except Exception as e:
            return [{"error": str(e)}]

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
        
        # Extract domain/company info
        domains = article.get("domains", [])
        domain_name = domains[0].get("domain", "General") if domains and isinstance(domains[0], dict) else "General"
        companies = [
            c.get("company") 
            for d in domains if isinstance(d, dict) 
            for c in d.get("companies", []) if c.get("company")
        ]

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

    def get_graph_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Fetch all nodes and relationships for the UI visualization."""
        nodes = []
        edges = []
        seen_nodes = set()
        
        with self.driver.session() as session:
            # Fetch relationships and nodes. 
            # Note: Article uses 'id', while others use 'name' as their unique identifier.
            result = session.run("""
                MATCH (a:Article)-[r]->(b)
                RETURN a.id AS source_id, a.title AS source_name, labels(a)[0] AS source_type,
                       type(r) AS rel_type,
                       b.name AS target_name, labels(b)[0] AS target_type
            """)
            
            for record in result:
                source_id = record["source_id"]
                source_name = record["source_name"]
                source_type = record["source_type"]
                rel_type = record["rel_type"]
                target_name = record["target_name"]
                target_type = record["target_type"]
                
                # Create a unique ID for target nodes based on type and name
                target_id = f"{target_type}:{target_name}"
                
                # Add Source Node (Article)
                if source_id not in seen_nodes:
                    nodes.append({
                        "id": source_id, 
                        "label": source_name or source_id[:8], 
                        "group": source_type
                    })
                    seen_nodes.add(source_id)
                
                # Add Target Node (Domain, Region, Event, Company)
                if target_id not in seen_nodes:
                    nodes.append({
                        "id": target_id, 
                        "label": target_name, 
                        "group": target_type
                    })
                    seen_nodes.add(target_id)
                
                # Add Edge
                edges.append({
                    "source": source_id, 
                    "target": target_id, 
                    "label": rel_type
                })
                
        return nodes, edges