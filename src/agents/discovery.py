"""Discovery Agent - finds hidden connections and insights in the knowledge graph."""

import random

import networkx as nx

from src.config import Config
from src.graph_store import GraphStore
from src.llm_utils import call_ollama


class DiscoveryAgent:

    def __init__(self, graph: GraphStore, config: Config):
        self.graph = graph
        self.config = config

    def run(self) -> dict:
        """Analyze the graph and surface insights."""
        if self.graph.graph.number_of_nodes() < 3:
            return {"insights": ["Add more knowledge to discover connections."], "count": 0}

        insights = []

        # 1. Bridge concepts
        bridges = self._find_bridge_concepts()
        insights.extend(bridges)

        # 2. Unexpected connections
        unexpected = self._find_unexpected_connections()
        insights.extend(unexpected)

        # 3. Thematic clusters
        clusters = self._find_thematic_clusters()
        insights.extend(clusters)

        if not insights:
            insights.append("No notable patterns found yet. Keep adding knowledge!")

        return {"insights": insights, "count": len(insights)}

    def _find_bridge_concepts(self) -> list[str]:
        """Find entities that connect otherwise separate areas of knowledge."""
        bridges = self.graph.get_bridge_nodes(top_n=3)
        insights = []
        for node_id, score in bridges:
            entity = self.graph.get_entity(node_id)
            if not entity:
                continue
            label = entity.get("label", node_id)

            # Find what clusters this node connects
            neighbors = list(self.graph.graph.successors(node_id)) + list(self.graph.graph.predecessors(node_id))
            neighbor_labels = []
            for n in neighbors[:6]:
                ndata = self.graph.get_entity(n)
                if ndata:
                    neighbor_labels.append(ndata.get("label", n))

            if len(neighbor_labels) >= 2:
                connections = ", ".join(neighbor_labels)
                insight = f"**{label}** is a key connector linking: {connections}"
                insights.append(insight)

        return insights

    def _find_unexpected_connections(self) -> list[str]:
        """Find shortest paths between entities in different clusters."""
        clusters = self.graph.get_clusters()
        if len(clusters) < 2:
            # Single cluster: find long paths within it
            return self._find_long_paths()

        insights = []
        undirected = self.graph.graph.to_undirected()

        # Pick entities from different clusters and find paths
        cluster_list = [list(c) for c in clusters if len(c) >= 2]
        for i in range(min(len(cluster_list) - 1, 2)):
            for j in range(i + 1, min(len(cluster_list), i + 3)):
                src = random.choice(cluster_list[i])
                tgt = random.choice(cluster_list[j])
                try:
                    path = nx.shortest_path(undirected, src, tgt)
                    if 3 <= len(path) <= 6:
                        path_labels = []
                        for node in path:
                            entity = self.graph.get_entity(node)
                            path_labels.append(entity.get("label", node) if entity else node)
                        chain = " -> ".join(path_labels)
                        insight = self._generate_path_insight(path_labels)
                        if insight:
                            insights.append(insight)
                        else:
                            insights.append(f"Hidden connection: {chain}")
                except nx.NetworkXNoPath:
                    continue

        return insights[:3]

    def _find_long_paths(self) -> list[str]:
        """Find interesting long paths within a connected graph."""
        insights = []
        nodes = list(self.graph.graph.nodes())
        if len(nodes) < 4:
            return []

        undirected = self.graph.graph.to_undirected()
        sampled = random.sample(nodes, min(len(nodes), 10))

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                try:
                    path = nx.shortest_path(undirected, sampled[i], sampled[j])
                    if len(path) >= 3:
                        path_labels = []
                        for node in path:
                            entity = self.graph.get_entity(node)
                            path_labels.append(entity.get("label", node) if entity else node)
                        chain = " -> ".join(path_labels)
                        insights.append(f"Did you know? {path_labels[0]} connects to {path_labels[-1]} through: {chain}")
                        if len(insights) >= 3:
                            return insights
                except nx.NetworkXNoPath:
                    continue

        return insights

    def _find_thematic_clusters(self) -> list[str]:
        """Identify and summarize thematic clusters."""
        clusters = self.graph.get_clusters()
        insights = []
        for cluster in clusters[:3]:
            labels = []
            for node_id in list(cluster)[:8]:
                entity = self.graph.get_entity(node_id)
                if entity:
                    labels.append(entity.get("label", node_id))
            if len(labels) >= 2:
                members = ", ".join(labels)
                insight = self._summarize_cluster(labels)
                if insight:
                    insights.append(insight)
                else:
                    insights.append(f"Knowledge cluster ({len(cluster)} items): {members}")
        return insights

    def _generate_path_insight(self, path_labels: list[str]) -> str | None:
        """Use Ollama to generate an insight about a connection path."""
        chain = " -> ".join(path_labels)
        prompt = f"""Given this connection path in a knowledge graph, generate a brief "Did you know?" insight (one sentence).

Path: {chain}

Insight:"""
        try:
            return call_ollama(prompt, self.config.ollama_model, temperature=0.5)
        except Exception:
            return None

    def _summarize_cluster(self, labels: list[str]) -> str | None:
        """Use Ollama to summarize a thematic cluster."""
        members = ", ".join(labels)
        prompt = f"""These concepts are connected in a knowledge graph: {members}

Summarize the theme of this cluster in one sentence, starting with "Theme:":"""
        try:
            return call_ollama(prompt, self.config.ollama_model, temperature=0.3)
        except Exception:
            return None
