"""Shared LLM utilities for all agents."""

import json
import re

import ollama as ollama_client


def slugify(label: str) -> str:
    """Convert label to a URL-safe identifier."""
    slug = label.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def parse_json_response(response_text: str) -> dict:
    """Parse JSON from Ollama response, handling markdown code fences."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


def call_ollama(prompt: str, model: str, as_json: bool = False, temperature: float = 0.1) -> str:
    """Call Ollama and return the response text."""
    options = {"temperature": temperature}
    kwargs = {"model": model, "prompt": prompt, "options": options}
    if as_json:
        kwargs["format"] = "json"
    response = ollama_client.generate(**kwargs)
    return response.get("response", "").strip()


def check_ollama_available(model: str) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        models = ollama_client.list()
        model_base = model.split(":")[0]
        for m in models.get("models", []):
            name = m.get("name", "") if isinstance(m, dict) else str(m)
            if model_base in name:
                return True
        return False
    except Exception:
        return False
