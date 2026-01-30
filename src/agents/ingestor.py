"""Ingestor Agent - extracts entities and relationships from text."""

from src.config import Config
from src.graph_store import GraphStore
from src.llm_utils import call_ollama, parse_json_response, slugify


class IngestorAgent:

    def __init__(self, graph: GraphStore, config: Config):
        self.graph = graph
        self.config = config

    def run(self, text: str) -> dict:
        """Extract entities and relationships from text and add to graph."""
        entities = self._extract_entities(text)
        if not entities:
            return {"entities_added": 0, "relationships_added": 0, "entities": []}

        relationships = self._extract_relationships(text, entities)

        # Add entities to graph, checking for duplicates
        added = []
        id_map = {}  # label -> entity_id mapping for relationship resolution
        for entity in entities:
            label = entity.get("label", "").strip()
            if not label:
                continue
            entity_type = entity.get("type", "concept")
            if entity_type not in self.config.entity_types:
                entity_type = "concept"

            # Check for existing similar entity
            similar = self.graph.find_similar_entities(label, self.config.similarity_threshold)
            if similar:
                entity_id = similar[0]["id"]
            else:
                entity_id = slugify(label)
                self.graph.add_entity(
                    entity_id=entity_id,
                    label=label,
                    entity_type=entity_type,
                    description=entity.get("description", ""),
                    source_text=text[:200],
                )
                added.append(entity_id)

            id_map[label.lower()] = entity_id

        # Add relationships
        rels_added = 0
        for rel in relationships:
            src_label = rel.get("source", "").lower()
            tgt_label = rel.get("target", "").lower()
            src_id = id_map.get(src_label)
            tgt_id = id_map.get(tgt_label)

            # Try fuzzy match if exact lookup fails
            if not src_id:
                src_id = self._resolve_id(src_label, id_map)
            if not tgt_id:
                tgt_id = self._resolve_id(tgt_label, id_map)

            if src_id and tgt_id and src_id != tgt_id:
                rel_type = rel.get("type", "relates_to")
                if rel_type not in self.config.relationship_types:
                    rel_type = "relates_to"
                self.graph.add_relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    rel_type=rel_type,
                    description=rel.get("description", ""),
                    source_text=text[:200],
                )
                rels_added += 1

        return {
            "entities_added": len(added),
            "relationships_added": rels_added,
            "entities": added,
        }

    def _extract_entities(self, text: str) -> list[dict]:
        """Use Ollama to extract entities from text."""
        valid_types = ", ".join(self.config.entity_types)
        prompt = f"""Extract all notable entities from the following text.

For each entity provide:
- label: The entity name
- type: One of [{valid_types}]
- description: A brief one-sentence description

Text: {text}

Respond in valid JSON: {{"entities": [{{"label": "...", "type": "...", "description": "..."}}]}}"""

        try:
            response = call_ollama(prompt, self.config.ollama_model, as_json=True)
            parsed = parse_json_response(response)
            return parsed.get("entities", [])[:self.config.max_entities_per_extraction]
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def _extract_relationships(self, text: str, entities: list[dict]) -> list[dict]:
        """Use Ollama to extract relationships between entities."""
        if len(entities) < 2:
            return []

        entity_list = ", ".join(e.get("label", "") for e in entities)
        valid_types = ", ".join(self.config.relationship_types)
        prompt = f"""Given these entities extracted from a text, identify the relationships between them.

Entities: {entity_list}

Original text: {text}

For each relationship provide:
- source: Entity label (must be from the list above)
- target: Entity label (must be from the list above)
- type: One of [{valid_types}]
- description: Brief description of the relationship

Respond in valid JSON: {{"relationships": [{{"source": "...", "target": "...", "type": "...", "description": "..."}}]}}"""

        try:
            response = call_ollama(prompt, self.config.ollama_model, as_json=True)
            parsed = parse_json_response(response)
            return parsed.get("relationships", [])
        except Exception as e:
            print(f"Relationship extraction error: {e}")
            return []

    def _resolve_id(self, label: str, id_map: dict) -> str | None:
        """Try to resolve a label to an entity ID via fuzzy matching."""
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0
        for known_label, eid in id_map.items():
            score = SequenceMatcher(None, label, known_label).ratio()
            if score > best_score and score > 0.6:
                best_score = score
                best_match = eid
        return best_match
