"""
Local Knowledge Graph — an agentic knowledge graph that runs on your machine.
"""

import json
import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader

from src.orchestrator import Orchestrator
from src.llm_utils import check_ollama_available
from src.config import get_config

st.set_page_config(page_title="Local Knowledge Graph", layout="wide")

# --- Initialize ---
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
    st.session_state.chat_history = []
    # Load ingested files manifest (persists across restarts)
    _manifest_path = get_config().graph_path.parent / "ingested_files.json"
    if _manifest_path.exists():
        st.session_state.ingested_files = set(json.loads(_manifest_path.read_text()))
    else:
        st.session_state.ingested_files = set()

orch = st.session_state.orchestrator
config = get_config()

# --- Sidebar ---
with st.sidebar:
    st.title("Local KG")
    st.caption("Agentic Knowledge Graph")

    # Ollama status
    ollama_ok = check_ollama_available(config.ollama_model)
    if ollama_ok:
        st.success(f"Ollama: {config.ollama_model}")
    else:
        st.error(f"Ollama not available ({config.ollama_model})")

    st.divider()

    # Graph stats
    stats = orch.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Entities", stats["node_count"])
    col2.metric("Relationships", stats["edge_count"])

    if stats.get("type_counts"):
        st.subheader("Entity Types")
        for etype, count in sorted(stats["type_counts"].items(), key=lambda x: x[1], reverse=True):
            st.text(f"  {etype}: {count}")

    if stats.get("top_entities"):
        st.subheader("Top Entities")
        for entity in stats["top_entities"][:8]:
            st.text(f"  {entity['label']} ({entity['connections']})")

    st.divider()

    if st.button("Reset Graph", type="secondary"):
        st.session_state.confirm_reset = True

    if st.session_state.get("confirm_reset"):
        st.warning("This will delete all entities, relationships, and ingestion history.")
        col_yes, col_no = st.columns(2)
        if col_yes.button("Yes, reset"):
            # Clear graph
            orch.graph.graph.clear()
            orch.graph.save()
            # Clear ingestion manifest
            manifest_path = config.graph_path.parent / "ingested_files.json"
            if manifest_path.exists():
                manifest_path.unlink()
            st.session_state.ingested_files = set()
            # Clear viz
            viz_path = config.graph_path.parent / "graph_viz.html"
            if viz_path.exists():
                viz_path.unlink()
            st.session_state.confirm_reset = False
            st.session_state.chat_history = []
            st.rerun()
        if col_no.button("Cancel"):
            st.session_state.confirm_reset = False
            st.rerun()

    st.caption("100% local - Ollama + NetworkX")

# --- Main Content ---
tab_add, tab_ask, tab_discover, tab_curate, tab_graph = st.tabs(
    ["Add Knowledge", "Ask Questions", "Discover", "Curate", "Graph View"]
)

# --- Tab: Add Knowledge ---
with tab_add:
    st.header("Add Knowledge")

    ingest_mode = st.radio(
        "Input method:",
        ["Paste text", "Ingest folder"],
        horizontal=True,
    )

    if ingest_mode == "Paste text":
        st.write("Paste notes, articles, or ideas. The Ingestor agent will extract entities and relationships.")

        text_input = st.text_area(
            "Enter text to ingest:",
            height=200,
            placeholder="e.g., Python was created by Guido van Rossum in 1991. FastAPI is a modern Python web framework..."
        )

        if st.button("Ingest", type="primary", disabled=not text_input):
            with st.spinner("Extracting entities and relationships..."):
                result = orch.ingest(text_input)

            st.success(
                f"Added **{result['entities_added']}** entities and "
                f"**{result['relationships_added']}** relationships"
            )

            if result.get("entities"):
                st.write("New entities:", ", ".join(result["entities"]))

            st.rerun()

    else:
        st.write("Point to a folder of .txt, .md, .csv, or .pdf files. Each file will be ingested separately.")

        default_samples = str(Path(__file__).parent / "samples")
        folder_path = st.text_input("Folder path:", value=default_samples)

        if folder_path:
            folder = Path(folder_path)
            if folder.exists() and folder.is_dir():
                all_files = sorted(
                    f for f in folder.iterdir()
                    if f.is_file() and f.suffix in (".txt", ".md", ".csv", ".pdf")
                )
                new_files = [f for f in all_files if f.name not in st.session_state.ingested_files]
                already = len(all_files) - len(new_files)

                st.write(f"Found **{len(all_files)}** files ({already} already ingested, **{len(new_files)}** new):")
                for f in all_files:
                    marker = " (ingested)" if f.name in st.session_state.ingested_files else ""
                    st.text(f"  {f.name}{marker}")

                if st.button("Ingest New Files", type="primary", disabled=len(new_files) == 0):
                    total_entities = 0
                    total_rels = 0
                    progress = st.progress(0, text="Starting...")

                    for i, file_path in enumerate(new_files):
                        progress.progress(
                            (i + 1) / len(new_files),
                            text=f"Ingesting {file_path.name}..."
                        )
                        if file_path.suffix == ".pdf":
                            try:
                                reader = PdfReader(str(file_path))
                                text = "\n".join(
                                    page.extract_text() or "" for page in reader.pages
                                )
                            except Exception as e:
                                st.warning(f"Could not read {file_path.name}: {e}")
                                continue
                        else:
                            text = file_path.read_text(errors="ignore")
                        if text.strip():
                            result = orch.ingest(text)
                            total_entities += result["entities_added"]
                            total_rels += result["relationships_added"]
                            st.session_state.ingested_files.add(file_path.name)
                            st.write(
                                f"**{file_path.name}**: "
                                f"{result['entities_added']} entities, "
                                f"{result['relationships_added']} relationships"
                            )

                    # Save manifest
                    manifest_path = config.graph_path.parent / "ingested_files.json"
                    manifest_path.write_text(json.dumps(sorted(st.session_state.ingested_files)))

                    progress.progress(1.0, text="Done!")
                    st.success(
                        f"Ingested **{len(new_files)}** new files: "
                        f"**{total_entities}** entities, **{total_rels}** relationships"
                    )
                    st.rerun()
            else:
                st.error(f"Folder not found: {folder_path}")

# --- Tab: Ask Questions ---
with tab_ask:
    st.header("Ask Questions")
    st.write("Ask questions about your knowledge graph. The Query agent traverses the graph to find answers.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about your knowledge...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge graph..."):
                result = orch.query(question)
            st.write(result["answer"])
            if result.get("entities_used"):
                st.caption(f"Used {result.get('subgraph_size', 0)} nodes from graph")

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

# --- Tab: Discover ---
with tab_discover:
    st.header("Discover Connections")
    st.write("The Discovery agent analyzes your graph to find hidden patterns and non-obvious connections.")

    if st.button("Discover Insights", type="primary"):
        with st.spinner("Analyzing knowledge graph..."):
            result = orch.discover()

        if result.get("insights"):
            for i, insight in enumerate(result["insights"], 1):
                st.info(f"**Insight {i}:** {insight}")
        else:
            st.write("No insights found. Add more knowledge to discover connections.")

# --- Tab: Curate ---
with tab_curate:
    st.header("Curate Graph")
    st.write("The Curator agent maintains graph quality: merges duplicates, finds orphans, and flags weak relationships.")

    if st.button("Run Curation", type="primary"):
        with st.spinner("Curating knowledge graph..."):
            result = orch.curate()

        if result.get("actions"):
            for action in result["actions"]:
                msg = action.get("message", "")
                action_type = action.get("type", "")
                if action_type == "merge":
                    st.success(msg)
                elif action_type == "orphan":
                    st.warning(msg)
                elif action_type == "weak_edge":
                    st.warning(msg)
                else:
                    st.info(msg)
            st.rerun()
        else:
            st.success("Graph is clean - no issues found.")

# --- Tab: Graph View ---
with tab_graph:
    st.header("Knowledge Graph")

    # Legend
    st.markdown(orch.get_legend_html(), unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh Graph", type="primary"):
            pass  # Will regenerate below

    if stats["node_count"] > 0:
        with st.spinner("Rendering graph..."):
            viz_result = orch.visualize()

        if viz_result.get("html_path"):
            html_path = Path(viz_result["html_path"])
            if html_path.exists():
                html_content = html_path.read_text()
                st.components.v1.html(html_content, height=620, scrolling=True)
                st.caption(f"Showing {viz_result['node_count']} entities, {viz_result['edge_count']} relationships")
        else:
            st.info("No entities to visualize yet.")
    else:
        st.info("Add knowledge first to see the graph visualization.")
