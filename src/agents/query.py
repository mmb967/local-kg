"""Query Agent - answers questions by traversing the knowledge graph."""

import networkx as nx

from src.config import Config
from src.graph_store import GraphStore
from src.llm_utils import call_ollama, parse_json_response


class QueryAgent:

    def __init__(self, graph: GraphStore, config: Config):
        self.graph = graph
        self.config = config

    def run(self, question: str) -> dict:
        """Answer a question using the knowledge graph."""
        # Extract key terms from the question
        key_terms = self._extract_query_terms(question)
        if not key_terms:
            return {"answer": "I couldn't identify what you're asking about.", "entities_used": []}

        # Search for matching entities
        relevant = []
        for term in key_terms:
            matches = self.graph.search_entities(term)
            relevant.extend(matches)

        # Deduplicate by ID
        seen = set()
        unique = []
        for entity in relevant:
            if entity["id"] not in seen:
                seen.add(entity["id"])
                unique.append(entity)
        relevant = unique[:10]  # Cap to avoid huge context

        if not relevant:
            return {
                "answer": "I don't have any information about that in my knowledge graph yet.",
                "entities_used": [],
            }

        # Get subgraph around matching entities
        entity_ids = [e["id"] for e in relevant]
        subgraph = self.graph.get_neighbors_multi(entity_ids, depth=2)

        # Format as context
        context = self._format_subgraph(subgraph)

        # Synthesize answer
        answer = self._synthesize_answer(question, context)

        return {
            "answer": answer,
            "entities_used": entity_ids,
            "subgraph_size": subgraph.number_of_nodes(),
        }

    def _extract_query_terms(self, question: str) -> list[str]:
        """Use Ollama to extract key search terms from a question."""
        prompt = f"""Extract the key search terms from this question. Return only the most important nouns/concepts to search for.

Question: {question}

Respond in valid JSON: {{"terms": ["term1", "term2"]}}"""

        try:
            response = call_ollama(prompt, self.config.ollama_model, as_json=True)
            parsed = parse_json_response(response)
            return parsed.get("terms", [])
        except Exception:
            # Fallback: split question into words, filter short ones
            words = question.lower().split()
            stop_words = {"what", "who", "where", "when", "why", "how", "is", "are", "was", "were",
                          "do", "does", "did", "the", "a", "an", "in", "on", "at", "to", "for",
                          "of", "with", "by", "from", "about", "and", "or", "not", "can", "has", "have"}
            return [w.strip("?.,!") for w in words if w not in stop_words and len(w) > 2]

    def _format_subgraph(self, subgraph: nx.DiGraph) -> str:
        """Format a subgraph as readable text for LLM context."""
        lines = []
        for node_id, data in subgraph.nodes(data=True):
            label = data.get("label", node_id)
            ntype = data.get("type", "unknown")
            desc = data.get("description", "")
            lines.append(f"Entity: {label} ({ntype}){f' - {desc}' if desc else ''}")

            # Outgoing relationships
            for _, target, edge_data in subgraph.out_edges(node_id, data=True):
                target_label = subgraph.nodes[target].get("label", target)
                rel_type = edge_data.get("type", "relates_to")
                rel_desc = edge_data.get("description", "")
                lines.append(f"  -> {rel_type}: {target_label}{f' ({rel_desc})' if rel_desc else ''}")

        return "\n".join(lines)

    def _synthesize_answer(self, question: str, context: str) -> str:
        """Use Ollama to synthesize an answer from graph context."""
        prompt = f"""You are a knowledge assistant. Answer the question using ONLY the knowledge graph context provided.
If the context doesn't contain enough information, say so honestly.
Be concise and direct.

Knowledge Graph Context:
{context}

Question: {question}

Answer:"""

        try:
            return call_ollama(prompt, self.config.ollama_model, temperature=0.3)
        except Exception as e:
            return f"Error generating answer: {e}"
