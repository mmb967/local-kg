"""NetworkX graph wrapper with JSON persistence."""

import json
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx


class GraphStore:
    """Thin wrapper around NetworkX DiGraph with JSON persistence."""

    def __init__(self, path: Path):
        self.path = path
        self.graph: nx.DiGraph = nx.DiGraph()
        self._load()

    # --- Core CRUD ---

    def add_entity(
        self,
        entity_id: str,
        label: str,
        entity_type: str,
        description: str = "",
        source_text: str = "",
        confidence: float = 0.9,
    ) -> str:
        """Add or update an entity node. Returns the entity_id."""
        now = datetime.now(timezone.utc).isoformat()
        if self.graph.has_node(entity_id):
            self.graph.nodes[entity_id]["updated_at"] = now
            if description:
                self.graph.nodes[entity_id]["description"] = description
        else:
            self.graph.add_node(
                entity_id,
                label=label,
                type=entity_type,
                description=description,
                source_text=source_text,
                confidence=confidence,
                created_at=now,
                updated_at=now,
            )
        return entity_id

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        description: str = "",
        weight: float = 0.8,
        source_text: str = "",
    ) -> None:
        """Add a relationship edge between two entities."""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return
        now = datetime.now(timezone.utc).isoformat()
        self.graph.add_edge(
            source_id,
            target_id,
            type=rel_type,
            description=description,
            weight=weight,
            source_text=source_text,
            created_at=now,
        )

    def get_entity(self, entity_id: str) -> dict | None:
        """Get entity data by ID."""
        if self.graph.has_node(entity_id):
            data = dict(self.graph.nodes[entity_id])
            data["id"] = entity_id
            return data
        return None

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and all its edges."""
        if self.graph.has_node(entity_id):
            self.graph.remove_node(entity_id)

    # --- Search ---

    def search_entities(self, query: str, entity_type: str | None = None) -> list[dict]:
        """Fuzzy search entities by label."""
        query_lower = query.lower()
        results = []
        for node_id, data in self.graph.nodes(data=True):
            if entity_type and data.get("type") != entity_type:
                continue
            label = data.get("label", "").lower()
            # Substring match
            if query_lower in label or label in query_lower:
                score = 1.0
            else:
                score = SequenceMatcher(None, query_lower, label).ratio()
            if score > 0.4:
                entry = dict(data)
                entry["id"] = node_id
                entry["match_score"] = score
                results.append(entry)
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

    def find_similar_entities(self, label: str, threshold: float = 0.85) -> list[dict]:
        """Find entities with similar labels for deduplication."""
        label_lower = label.lower()
        similar = []
        for node_id, data in self.graph.nodes(data=True):
            existing_label = data.get("label", "").lower()
            ratio = SequenceMatcher(None, label_lower, existing_label).ratio()
            if ratio >= threshold:
                entry = dict(data)
                entry["id"] = node_id
                entry["similarity"] = ratio
                similar.append(entry)
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar

    # --- Graph traversal ---

    def get_neighbors(self, entity_id: str, depth: int = 1) -> nx.DiGraph:
        """Get subgraph around an entity up to given depth."""
        if not self.graph.has_node(entity_id):
            return nx.DiGraph()
        nodes = {entity_id}
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            nodes.update(next_frontier)
            frontier = next_frontier
        return self.graph.subgraph(nodes).copy()

    def get_neighbors_multi(self, entity_ids: list[str], depth: int = 2) -> nx.DiGraph:
        """Get subgraph around multiple entities."""
        all_nodes = set()
        for eid in entity_ids:
            sub = self.get_neighbors(eid, depth)
            all_nodes.update(sub.nodes())
        if not all_nodes:
            return nx.DiGraph()
        return self.graph.subgraph(all_nodes).copy()

    def get_subgraph(self, entity_ids: list[str]) -> nx.DiGraph:
        """Get subgraph containing exactly the given entities and edges between them."""
        valid = [eid for eid in entity_ids if self.graph.has_node(eid)]
        return self.graph.subgraph(valid).copy()

    # --- Analysis ---

    def get_clusters(self) -> list[set[str]]:
        """Get connected components as clusters."""
        undirected = self.graph.to_undirected()
        return [c for c in nx.connected_components(undirected) if len(c) > 1]

    def get_bridge_nodes(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Get nodes with highest betweenness centrality."""
        if len(self.graph) < 3:
            return []
        centrality = nx.betweenness_centrality(self.graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [(nid, score) for nid, score in sorted_nodes[:top_n] if score > 0]

    def get_stats(self) -> dict:
        """Get graph statistics."""
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        # Top entities by connection count
        top_entities = []
        for node_id, data in self.graph.nodes(data=True):
            degree = self.graph.degree(node_id)
            top_entities.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "type": data.get("type", "unknown"),
                "connections": degree,
            })
        top_entities.sort(key=lambda x: x["connections"], reverse=True)

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "type_counts": type_counts,
            "top_entities": top_entities[:10],
            "cluster_count": len(self.get_clusters()),
        }

    # --- Dedup ---

    def merge_entities(self, keep_id: str, remove_id: str) -> None:
        """Merge remove_id into keep_id, transferring all edges."""
        if not self.graph.has_node(keep_id) or not self.graph.has_node(remove_id):
            return
        # Transfer incoming edges
        for pred in list(self.graph.predecessors(remove_id)):
            if pred != keep_id:
                edge_data = dict(self.graph.edges[pred, remove_id])
                self.graph.add_edge(pred, keep_id, **edge_data)
        # Transfer outgoing edges
        for succ in list(self.graph.successors(remove_id)):
            if succ != keep_id:
                edge_data = dict(self.graph.edges[remove_id, succ])
                self.graph.add_edge(keep_id, succ, **edge_data)
        self.graph.remove_node(remove_id)

    def get_orphans(self) -> list[str]:
        """Find nodes with no connections."""
        return [n for n in self.graph.nodes() if self.graph.degree(n) == 0]

    # --- Persistence ---

    def save(self) -> None:
        """Save graph to JSON."""
        data = nx.node_link_data(self.graph)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        """Load graph from JSON if exists."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, directed=True)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading graph: {e}")
                self.graph = nx.DiGraph()
