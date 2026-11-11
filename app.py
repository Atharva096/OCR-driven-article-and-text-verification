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
tab_pipeline, tab_search, tab_graph, tab_data = st.tabs(["📥 Pipeline", "🔍 Semantic Search", "🕸️ Knowledge Graph", "📊 Data Explorer"])

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
    
    # Check if Neo4j is actually connected
    if stores["graph"] is None:
        st.warning("⚠️ Neo4j is not running or not configured. Please start your Neo4j Desktop database to see the visualization.")
    else:
        try:
            nodes, edges = stores["graph"].get_graph_data()
            
            if nodes:
                from streamlit_agraph import agraph, Node, Edge, Config
                
                st_nodes = [Node(id=n["id"], label=n["label"], group=n["group"]) for n in nodes]
                st_edges = [Edge(source=e["source"], target=e["target"], label=e["label"]) for e in edges]
                
                config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
                agraph(nodes=st_nodes, edges=st_edges, config=config)
            else:
                st.info("Graph is empty. Run an extraction with '--enable-graph' to populate nodes.")
        except Exception as e:
            st.error(f"Failed to fetch graph data: {e}")

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