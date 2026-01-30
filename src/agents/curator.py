"""Curator Agent - maintains graph quality through deduplication and cleanup."""

from src.config import Config
from src.graph_store import GraphStore
from src.llm_utils import call_ollama, parse_json_response


class CuratorAgent:

    def __init__(self, graph: GraphStore, config: Config):
        self.graph = graph
        self.config = config

    def run(self) -> dict:
        """Run graph maintenance: dedup, orphan removal, quality checks."""
        actions = []

        # 1. Find and merge duplicates
        merges = self._deduplicate()
        actions.extend(merges)

        # 2. Find orphan nodes
        orphans = self._find_orphans()
        actions.extend(orphans)

        # 3. Find weak edges
        weak = self._find_weak_relationships()
        actions.extend(weak)

        return {"actions": actions, "count": len(actions)}

    def _deduplicate(self) -> list[dict]:
        """Find and merge duplicate entities."""
        actions = []
        nodes = list(self.graph.graph.nodes(data=True))
        merged_ids = set()

        for i in range(len(nodes)):
            node_id_a, data_a = nodes[i]
            if node_id_a in merged_ids:
                continue
            label_a = data_a.get("label", "")

            for j in range(i + 1, len(nodes)):
                node_id_b, data_b = nodes[j]
                if node_id_b in merged_ids:
                    continue
                label_b = data_b.get("label", "")

                similar = self.graph.find_similar_entities(label_a, self.config.similarity_threshold)
                match_ids = {s["id"] for s in similar}

                if node_id_b in match_ids:
                    # Confirm with Ollama for ambiguous cases
                    should_merge = self._confirm_merge(data_a, data_b)
                    if should_merge:
                        # Keep the one with more connections
                        degree_a = self.graph.graph.degree(node_id_a)
                        degree_b = self.graph.graph.degree(node_id_b)
                        if degree_a >= degree_b:
                            self.graph.merge_entities(node_id_a, node_id_b)
                            merged_ids.add(node_id_b)
                            actions.append({
                                "type": "merge",
                                "message": f"Merged '{label_b}' into '{label_a}'",
                            })
                        else:
                            self.graph.merge_entities(node_id_b, node_id_a)
                            merged_ids.add(node_id_a)
                            actions.append({
                                "type": "merge",
                                "message": f"Merged '{label_a}' into '{label_b}'",
                            })
                            break  # node_a is gone

        return actions

    def _confirm_merge(self, data_a: dict, data_b: dict) -> bool:
        """Ask Ollama to confirm whether two entities should be merged."""
        label_a = data_a.get("label", "")
        type_a = data_a.get("type", "")
        desc_a = data_a.get("description", "")
        label_b = data_b.get("label", "")
        type_b = data_b.get("type", "")
        desc_b = data_b.get("description", "")

        prompt = f"""Are these two entities the same thing? Answer only YES or NO.

Entity A: {label_a} ({type_a}) - {desc_a}
Entity B: {label_b} ({type_b}) - {desc_b}

Answer:"""

        try:
            response = call_ollama(prompt, self.config.ollama_model, temperature=0.1)
            return "yes" in response.lower()
        except Exception:
            # Default to not merging if unsure
            return False

    def _find_orphans(self) -> list[dict]:
        """Find and report orphan nodes."""
        orphans = self.graph.get_orphans()
        actions = []
        for orphan_id in orphans:
            entity = self.graph.get_entity(orphan_id)
            label = entity.get("label", orphan_id) if entity else orphan_id
            actions.append({
                "type": "orphan",
                "message": f"Orphan entity: '{label}' has no connections",
                "entity_id": orphan_id,
            })
        return actions

    def _find_weak_relationships(self) -> list[dict]:
        """Find edges with low weight or confidence."""
        actions = []
        for u, v, data in self.graph.graph.edges(data=True):
            weight = data.get("weight", 1.0)
            if weight < 0.3:
                src = self.graph.get_entity(u)
                tgt = self.graph.get_entity(v)
                src_label = src.get("label", u) if src else u
                tgt_label = tgt.get("label", v) if tgt else v
                actions.append({
                    "type": "weak_edge",
                    "message": f"Weak relationship: '{src_label}' -> '{tgt_label}' (weight: {weight:.2f})",
                })
        return actions
