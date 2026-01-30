"""Configuration for Local Knowledge Graph."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3:8b")
    graph_path: Path = field(default_factory=lambda: BASE_DIR / os.getenv("GRAPH_PATH", "data/graph.json"))
    entity_types: list = field(default_factory=lambda: [
        "person", "concept", "technology", "event", "project", "organization",
    ])
    relationship_types: list = field(default_factory=lambda: [
        "uses", "relates_to", "created_by", "part_of", "inspired_by", "contradicts", "supports",
    ])
    max_entities_per_extraction: int = 20
    similarity_threshold: float = 0.85

    def __post_init__(self):
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)


_config = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
