# Local Knowledge Graph

The intent of this project is to help you develop the intuition for building an agentic knowledge graph — locally, from scratch, with minimal dependencies.

Paste notes, articles, or PDFs and it builds a knowledge graph of entities and relationships you can query, explore, and discover connections across.

5 agents (Ingestor, Query, Discovery, Curator, Visualizer) — each a plain Python class, no framework. They coordinate through the graph itself, not through each other.

Stack: Streamlit UI, NetworkX graph engine, pyvis visualization, Ollama local LLM, PyPDF2 for document ingestion. JSON persistence — no database.

## Why no framework

Each agent is a Python class with a `run()` method that reads/writes to a shared NetworkX graph. No CrewAI, no LangChain, no LangGraph. The agents don't need to talk to each other — they coordinate through the graph itself. This keeps the code simple enough that each agent fits in a single file you can read in a few minutes.

## What I learned

The Discovery agent — which uses betweenness centrality and shortest-path algorithms to find bridge concepts — turned out to be the most interesting piece. It surfaces connections I genuinely didn't see.

Entity extraction from an 8B parameter local model is surprisingly usable. Not perfect — the Curator agent exists because duplicates happen — but good enough to build a useful graph over time.

The multi-agent pattern, where each agent is just a Python class with a `run()` method reading from a shared data store, is something I'd use again. It's easy to reason about and easy to extend.

## How it was built

Co-created with [Claude Code](https://claude.ai/claude-code) — Anthropic's CLI coding agent. Architecture, implementation, and iteration were done collaboratively through conversation.
