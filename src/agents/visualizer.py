"""Visualizer Agent - generates interactive graph visualizations with pyvis."""

from pathlib import Path

from pyvis.network import Network

from src.config import Config
from src.graph_store import GraphStore


# Color map for entity types
TYPE_COLORS = {
    "person": "#e74c3c",
    "concept": "#3498db",
    "technology": "#2ecc71",
    "event": "#f39c12",
    "project": "#9b59b6",
    "organization": "#1abc9c",
}
DEFAULT_COLOR = "#95a5a6"


class VisualizerAgent:

    def __init__(self, graph: GraphStore, config: Config):
        self.graph = graph
        self.config = config

    def run(self, entity_ids: list[str] | None = None) -> dict:
        """Generate an interactive graph visualization.

        Args:
            entity_ids: If provided, visualize only the subgraph around these entities.
                       If None, visualize the full graph (capped at 200 nodes).

        Returns:
            Dict with 'html_path' pointing to the generated file.
        """
        if entity_ids:
            subgraph = self.graph.get_neighbors_multi(entity_ids, depth=2)
        else:
            subgraph = self.graph.graph

        # Cap visualization to avoid browser performance issues
        if subgraph.number_of_nodes() > 200:
            # Keep top 200 nodes by degree
            degrees = sorted(subgraph.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, _ in degrees[:200]]
            subgraph = subgraph.subgraph(top_nodes).copy()

        if subgraph.number_of_nodes() == 0:
            return {"html_path": None, "node_count": 0, "edge_count": 0}

        # Create pyvis network
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333",
            directed=True,
        )
        net.toggle_physics(True)
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -3000,
                    "springLength": 150,
                    "springConstant": 0.04
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200
            }
        }
        """)

        # Add nodes
        for node_id, data in subgraph.nodes(data=True):
            label = data.get("label", node_id)
            ntype = data.get("type", "unknown")
            desc = data.get("description", "")
            color = TYPE_COLORS.get(ntype, DEFAULT_COLOR)
            degree = subgraph.degree(node_id)

            title = f"<b>{label}</b><br>Type: {ntype}"
            if desc:
                title += f"<br>{desc}"
            title += f"<br>Connections: {degree}"

            net.add_node(
                node_id,
                label=label,
                color=color,
                title=title,
                size=12 + degree * 4,
                shape="dot",
            )

        # Add edges
        for u, v, data in subgraph.edges(data=True):
            rel_type = data.get("type", "relates_to")
            desc = data.get("description", "")
            title = f"{rel_type}"
            if desc:
                title += f": {desc}"

            net.add_edge(
                u, v,
                title=title,
                label=rel_type,
                arrows="to",
                color="#cccccc",
                font={"size": 10, "color": "#999999"},
            )

        # Save to HTML
        html_path = self.config.graph_path.parent / "graph_viz.html"
        net.save_graph(str(html_path))

        return {
            "html_path": str(html_path),
            "node_count": subgraph.number_of_nodes(),
            "edge_count": subgraph.number_of_edges(),
        }

    def generate_legend_html(self) -> str:
        """Generate HTML for a color legend."""
        items = []
        for entity_type, color in TYPE_COLORS.items():
            items.append(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{color};border-radius:50%;margin-right:4px;"></span>'
                f'{entity_type.title()}'
            )
        return " &nbsp;&nbsp; ".join(items)
