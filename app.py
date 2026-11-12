import streamlit as st
import pandas as pd
import subprocess
import os
import sys
import tempfile
import json
from pathlib import Path

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(__file__))
from storage import Store
from vector_store import VectorStore
from graph_store import GraphStore

# --------------------------------------------------------------------------- Page Config
st.set_page_config(page_title="News Intelligence Dashboard", layout="wide", page_icon="📰")
st.title("📰 News Intelligence Dashboard")

# --------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Extraction Settings")
    
    uploaded_files = st.file_uploader("Upload PDFs/Images", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])
    url_input = st.text_input("Or enter a URL (RSS/Web)")
    
    st.divider()
    enable_vector = st.checkbox("Enable Vector DB (ChromaDB)", value=True)
    enable_graph = st.checkbox("Enable Graph DB (Neo4j)", value=True)
    
    st.divider()
    run_button = st.button("🚀 Run Extraction", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- Helper Functions
@st.cache_resource
def get_stores():
    """Initialize database connections safely."""
    stores = {
        "sqlite": Store("news.db"),
        "vector": VectorStore() if os.path.exists("vector_store.py") else None,
        "graph": None
    }
    
    # Safely attempt to connect to Neo4j
    if os.path.exists("graph_store.py"):
        try:
            stores["graph"] = GraphStore(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password")
            )
        except Exception:
            # If it fails, we just leave it as None
            pass
            
    return stores

# --------------------------------------------------------------------------- Main Tabs
tab_pipeline, tab_search, tab_graph, tab_data, tab_chat = st.tabs(["📥 Pipeline", "🔍 Semantic Search", "🕸️ Knowledge Graph", "📊 Data Explorer", "💬 Graph Chat"])

# --- TAB 1: PIPELINE ---
with tab_pipeline:
    st.header("Extraction Pipeline")
    if run_button:
        if not uploaded_files and not url_input:
            st.error("Please upload a file or enter a URL.")
        else:
            # Save uploaded files to temp directory
            temp_files = []
            if uploaded_files:
                temp_dir = tempfile.mkdtemp()
                for f in uploaded_files:
                    file_path = os.path.join(temp_dir, f.name)
                    with open(file_path, "wb") as file:
                        file.write(f.getbuffer())
                    temp_files.append(file_path)
            
            # Build command
            cmd = [sys.executable, "extractor.py"]
            cmd.extend(temp_files)
            if url_input:
                cmd.append(url_input)
            cmd.extend(["--db", "news.db", "--out", "extracted_news.json"])
            if enable_vector: cmd.append("--enable-vector")
            if enable_graph: cmd.append("--enable-graph")
            
            st.info("Starting extraction pipeline...")
            
            # Run subprocess and stream logs
            log_container = st.empty()
            full_log = ""
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            for line in process.stdout:
                full_log += line
                log_container.code(full_log, language="bash")
            
            process.wait()
            if process.returncode == 0:
                st.success("✅ Extraction completed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Extraction failed with code {process.returncode}")

# --- TAB 2: SEMANTIC SEARCH ---
with tab_search:
    st.header("🔍 Semantic Search (ChromaDB)")
    search_query = st.text_input("Ask a question about the news...", placeholder="e.g., 'Find all articles about nuclear deals in India'")
    
    if search_query:
        stores = get_stores()
        if stores["vector"]:
            with st.spinner("Searching vector database..."):
                results = stores["vector"].search(search_query, n_results=3)
            
            for res in results:
                with st.expander(f"📄 {res['metadata'].get('source', 'Unknown Source')} (Similarity: {1 - res['distance']:.2%})"):
                    st.write(f"**Date:** {res['metadata'].get('date', 'Unknown')}")
                    st.write(f"**Region:** {res['metadata'].get('region', 'Unknown')}")
                    st.write(res['content'])
        else:
            st.warning("Vector store not initialized. Run extraction with '--enable-vector' first.")

# --- TAB 3: KNOWLEDGE GRAPH ---
with tab_graph:
    st.header("🕸️ Knowledge Graph (Neo4j)")

    stores = get_stores()

    if stores["graph"] is None:
        st.warning("⚠️ Neo4j is not running. Start Neo4j Desktop and set NEO4J_URI=neo4j://127.0.0.1:7687 in .env.")
    else:
        try:
            nodes, edges = stores["graph"].get_graph_data()

            if nodes:
                import pyvis
                import tempfile
                import streamlit.components.v1 as components
                import json  # <--- ADDED THIS

                # Consistent color + size per entity type
                GROUP_COLORS = {
                    "Article": "#4A90D9",   # blue
                    "Region":  "#E15759",   # red
                    "Domain":  "#F2C14E",   # yellow
                    "Event":   "#59A14F",   # green
                    "Company": "#B07AA1",   # purple
                }
                GROUP_SIZES = {
                    "Article": 38,
                    "Region":  24,
                    "Domain":  24,
                    "Event":   24,
                    "Company": 22,
                }

                net = pyvis.network.Network(
                    height="650px", width="100%",
                    directed=True, notebook=False
                )

                # Add Nodes (casting to str to prevent type errors)
                for n in nodes:
                    grp = str(n.get("group", "Unknown"))
                    lbl = str(n.get("label", ""))
                    nid = str(n["id"])
                    
                    net.add_node(
                        nid,
                        label=lbl,
                        color=GROUP_COLORS.get(grp, "#999999"),
                        size=GROUP_SIZES.get(grp, 18),
                        shape="dot",
                        title=f"<b>{grp}</b><br>{lbl}",
                    )

                # Add Edges (using string color instead of dict to be safe)
                for e in edges:
                    net.add_edge(
                        str(e["source"]), str(e["target"]),
                        label=str(e.get("label", "")),
                        title=str(e.get("label", "")),
                        color="#aaaaaa", 
                    )

                # Configure Physics and Layout
                # CRITICAL FIX: pyvis requires a JSON STRING, not a dictionary!
                options = {
                    "physics": {
                        "enabled": True,
                        "barnesHut": {
                            "gravitationalConstant": -3000,
                            "centralGravity": 0.3,
                            "springLength": 220,
                            "springConstant": 0.02,
                            "damping": 0.10,
                            "avoidOverlap": 0.5
                        }
                    },
                    "edges": {
                        "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.25},
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
                        "font": {"size": 12, "color": "#666666", "strokeWidth": 0, "align": "middle"},
                        "width": 2,
                    },
                    "nodes": {
                        "font": {"size": 14, "color": "#222222", "face": "arial"},
                        "borderWidth": 2,
                        "borderWidthSelected": 4,
                        "shadow": {"enabled": True, "color": "#cccccc"},
                    },
                    "interaction": {"hover": True, "tooltipDelay": 100, "dragNodes": True, "zoomView": True},
                }

                # Pass the options as a JSON string
                net.set_options(json.dumps(options))

                html_path = tempfile.gettempdir() + "/news_graph.html"
                net.write_html(html_path)
                with open(html_path, "r", encoding="utf-8") as f:
                    html = f.read()

                components.html(html, height=690, scrolling=False)

                # Legend
                st.caption(f"🔗 {len(nodes)} nodes · {len(edges)} relationships  —  drag to move, scroll to zoom, hover for details")
                lc = st.columns(5)
                lc[0].markdown("🔵 Article")
                lc[1].markdown("🔴 Region")
                lc[2].markdown("🟡 Domain")
                lc[3].markdown("🟢 Event")
                lc[4].markdown("🟣 Company")
            else:
                st.info("Graph is empty. Run an extraction with '--enable-graph' to populate nodes.")
        except Exception as e:
            st.error(f"Failed to render graph: {e}")

# --- TAB 4: DATA EXPLORER ---
with tab_data:
    st.header("📊 Data Explorer (SQLite)")
    stores = get_stores()
    
    # Fetch articles - only query columns that actually exist
    rows = stores["sqlite"].conn.execute(
        "SELECT article_id, title, source, date, domain, record_json FROM articles"
    ).fetchall()
    
    if rows:
        # Parse the data
        data = []
        for row in rows:
            article_id, title, source, date, domain, record_json = row
            record = json.loads(record_json) if record_json else {}
            
            data.append({
                "ID": article_id,
                "Title": title,
                "Source": source,
                "Date": date,
                "Domain": domain,
                "Region": record.get("region", "Unknown"),
                "Event": record.get("event", "Unknown")
            })
        
        df = pd.DataFrame(data)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            source_filter = st.multiselect("Filter by Source", options=df["Source"].unique(), default=[])
        with col2:
            region_filter = st.multiselect("Filter by Region", options=df["Region"].unique(), default=[])
            
        filtered_df = df
        if source_filter:
            filtered_df = filtered_df[filtered_df["Source"].isin(source_filter)]
        if region_filter:
            filtered_df = filtered_df[filtered_df["Region"].isin(region_filter)]
            
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "articles.csv", "text/csv")
    else:
        st.info("No articles found in the database yet.")

# --- TAB 5: GRAPH CHAT ---
with tab_chat:
    st.header("💬 Chat with your Knowledge Graph")
    st.markdown("Ask natural language questions. The AI will write the Neo4j code to find the answer.")

    chat_question = st.text_input("Ask a question...", placeholder="e.g., What regions are mentioned in the articles?")
    ask_button = st.button("Ask Graph", type="secondary", use_container_width=True)

    if ask_button and chat_question:
        stores = get_stores()
        if stores["graph"] is None:
            st.error("Neo4j is not running.")
        else:
            import aiohttp
            import asyncio
            from extractor import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

            # 1. Build a rich schema string with the REAL connection patterns
            labels, rels, patterns = stores["graph"].get_schema()
            schema_text = (
                f"Node Labels: {labels}\n"
                f"Relationship Types: {rels}\n"
                f"Valid Connection Patterns (use ONLY these):\n" + "\n".join(patterns)
            )

            prompt = f"""You are a Neo4j Cypher expert.

{schema_text}

Rules:
- Use ONLY the connection patterns listed above. Do NOT invent multi-hop paths.
- Always RETURN named columns (e.g. RETURN r.name AS region), never a bare node.
- Return ONLY the raw Cypher query. No markdown, no explanation.

Examples:
Q: What regions are mentioned?
A: MATCH (a:Article)-[:OCCURRED_IN]->(r:Region) RETURN DISTINCT r.name AS region

Q: Which domains do articles belong to?
A: MATCH (a:Article)-[:BELONGS_TO]->(d:Domain) RETURN DISTINCT d.name AS domain

Q: {chat_question}
A:"""

            async def get_cypher():
                url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
                headers = {"Content-Type": "application/json"}
                if LLM_API_KEY:
                    headers["Authorization"] = f"Bearer {LLM_API_KEY}"
                payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a Neo4j Cypher expert. Return ONLY the raw Cypher query. No markdown, no explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]

            # 2. Generate the query (separate try/except)
            try:
                raw_response = asyncio.run(get_cypher()).strip()
            except Exception as e:
                st.error(f"Failed to generate query: {e}")
                st.stop()

            # Clean markdown fences if present
            cypher_query = raw_response
            if cypher_query.startswith("```cypher"):
                cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            elif cypher_query.startswith("```"):
                cypher_query = cypher_query.replace("```", "").strip()

            st.code(cypher_query, language="cypher")

            # 3. Execute the query (separate handling)
            results = stores["graph"].execute_query(cypher_query)

            if not results:
                st.warning("The query ran but returned no rows. That pattern may not exist in your data.")
            elif "error" in results[0]:
                st.error(f"Query failed: {results[0]['error']}")
            else:
                st.success("Query executed successfully!")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)