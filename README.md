# Local Knowledge Graph

A local, agentic knowledge graph. Paste text or point it at a folder of documents (TXT, MD, PDF) and it builds a graph of entities and relationships — then lets you query it, discover hidden connections, and visualize everything. Runs entirely on your machine.

## How it works

Five agents, each a plain Python class:

| Agent | What it does |
|-------|-------------|
| **Ingestor** | Extracts entities and relationships from text using a local LLM |
| **Query** | Answers natural language questions by traversing the graph |
| **Discovery** | Surfaces hidden connections using graph algorithms (betweenness centrality, shortest paths, community detection) |
| **Curator** | Deduplicates entities, removes orphans, flags weak relationships |
| **Visualizer** | Renders interactive, color-coded graph visualizations |

Agents don't talk to each other. They coordinate through a shared knowledge graph (NetworkX). No framework — no CrewAI, no LangChain.

## Stack

Streamlit, NetworkX, pyvis, Ollama, PyPDF2, python-dotenv.

## Setup

**Prerequisites:**
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

**Install:**

```bash
# Clone the repo
git clone https://github.com/mmb967/local-kg.git
cd local-kg

# Pull the LLM model
ollama pull llama3:8b

# Verify Ollama is running
ollama list   # should show llama3:8b

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Usage

### Add knowledge

**Paste text** — Switch to the "Add Knowledge" tab, paste any text, click Ingest. The Ingestor agent extracts entities (people, concepts, technologies, events, projects, organizations) and relationships.

**Ingest a folder** — Toggle to "Ingest folder", point it at a directory of `.txt`, `.md`, `.csv`, or `.pdf` files. It processes each file and tracks what's been ingested so you can add files incrementally without reprocessing.

### Query

Ask natural language questions in the "Ask Questions" tab. The Query agent finds matching entities, pulls a subgraph, and synthesizes an answer grounded in your graph — not general LLM knowledge.

### Discover

Click "Discover Insights" to surface:
- **Bridge concepts** — entities connecting otherwise separate clusters
- **Hidden paths** — shortest connections between seemingly unrelated ideas
- **Thematic clusters** — groups of related entities

### Curate

Click "Run Curation" to clean up:
- Merge duplicate entities (fuzzy matching + LLM confirmation)
- Flag orphan nodes with no connections
- Flag weak relationships

### Visualize

The "Graph View" tab shows an interactive graph. Nodes are color-coded by type, sized by number of connections. Hover for details.

## Project structure

```
├── app.py                 # Streamlit UI
├── requirements.txt
├── .env.example
├── data/                  # Graph storage (gitignored)
├── samples/               # Drop your documents here
└── src/
    ├── config.py          # Settings
    ├── graph_store.py     # NetworkX wrapper + JSON persistence
    ├── llm_utils.py       # Ollama helpers, JSON parsing
    ├── orchestrator.py    # Agent coordinator
    └── agents/
        ├── ingestor.py    # Entity/relationship extraction
        ├── query.py       # Graph-based Q&A
        ├── discovery.py   # Connection discovery
        ├── curator.py     # Dedup + cleanup
        └── visualizer.py  # pyvis visualization
```

## Configuration

Optional. The app works with defaults. Copy `.env.example` to `.env` to customize:

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3:8b
GRAPH_PATH=data/graph.json
```

Any Ollama-compatible model works. Larger models produce better entity extraction.

## Graph data model

**Nodes** — entities with: label, type (person/concept/technology/event/project/organization), description, confidence score.

**Edges** — relationships with: type (uses/relates_to/created_by/part_of/inspired_by/contradicts/supports), description, weight.

Stored as JSON via NetworkX's `node_link_data` format.

## Reset

Use the "Reset Graph" button in the sidebar to clear all entities, relationships, and ingestion history. Or delete `data/graph.json` and `data/ingested_files.json` manually.

## Privacy

Everything runs locally. The LLM runs through Ollama on your machine. The graph is a JSON file on disk. There are no network calls, no telemetry, no cloud services.

## Built with

Co-created with [Claude Code](https://claude.ai/claude-code).

## License

MIT
