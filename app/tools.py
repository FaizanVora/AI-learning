"""Fake tools the LLM can request — we execute them, not the model."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

# OpenAI / OpenRouter tool schema (JSON Schema for arguments).
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a city. "
                "Use this whenever the user asks about weather, temperature, or conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. London or Mumbai",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]

# Deterministic fake data — learning tool-calling, not meteorology.
_FAKE_WEATHER: dict[str, dict[str, Any]] = {
    "london": {"city": "London", "temp_c": 14, "condition": "cloudy"},
    "mumbai": {"city": "Mumbai", "temp_c": 31, "condition": "humid"},
    "tokyo": {"city": "Tokyo", "temp_c": 22, "condition": "clear"},
    "new york": {"city": "New York", "temp_c": 18, "condition": "rainy"},
}


def get_weather(city: str) -> dict[str, Any]:
    """Return fake weather for a city (case-insensitive lookup)."""
    key = city.strip().lower()
    if key in _FAKE_WEATHER:
        return _FAKE_WEATHER[key]
    return {
        "city": city.strip(),
        "temp_c": 20,
        "condition": "unknown",
        "note": "No canned data for this city; using a default.",
    }


_HANDLERS: dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
}


def execute_tool(name: str, arguments_json: str) -> str:
    """Run a tool by name and return a JSON string for the LLM.

    The model only *suggested* this call; we are the ones who actually run it.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid tool arguments JSON"})
    result = handler(**args)
    return json.dumps(result)
