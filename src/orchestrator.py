"""Orchestrator - coordinates all agents through a shared GraphStore."""

from src.config import get_config, Config
from src.graph_store import GraphStore
from src.agents.ingestor import IngestorAgent
from src.agents.query import QueryAgent
from src.agents.discovery import DiscoveryAgent
from src.agents.curator import CuratorAgent
from src.agents.visualizer import VisualizerAgent


class Orchestrator:

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.graph = GraphStore(self.config.graph_path)
        self.agents = {
            "ingestor": IngestorAgent(self.graph, self.config),
            "query": QueryAgent(self.graph, self.config),
            "discovery": DiscoveryAgent(self.graph, self.config),
            "curator": CuratorAgent(self.graph, self.config),
            "visualizer": VisualizerAgent(self.graph, self.config),
        }

    def ingest(self, text: str) -> dict:
        """Ingest text, extract entities/relationships, save graph."""
        result = self.agents["ingestor"].run(text=text)
        self.graph.save()
        return result

    def query(self, question: str) -> dict:
        """Answer a question using the knowledge graph."""
        return self.agents["query"].run(question=question)

    def discover(self) -> dict:
        """Surface hidden connections and insights."""
        return self.agents["discovery"].run()

    def curate(self) -> dict:
        """Run graph maintenance (dedup, orphan cleanup)."""
        result = self.agents["curator"].run()
        self.graph.save()
        return result

    def visualize(self, entity_ids: list[str] | None = None) -> dict:
        """Generate interactive graph visualization."""
        return self.agents["visualizer"].run(entity_ids=entity_ids)

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return self.graph.get_stats()

    def get_legend_html(self) -> str:
        """Get color legend HTML for the visualization."""
        return self.agents["visualizer"].generate_legend_html()
