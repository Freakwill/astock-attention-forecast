"""Minimal OpenAI-compatible chat client (DeepSeek by default).

The API key is read from the environment or from a Hermes ``.env`` file; it is
never logged and never written into reports.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ENV_CANDIDATES = (
    Path("~/.hermes/.env").expanduser(),
    Path("~/.hermes/profiles/coder/.env").expanduser(),
    Path(".env"),
)


class LLMError(RuntimeError):
    """Raised when the chat endpoint fails or returns an unusable payload."""


def load_api_key(name: str = "DEEPSEEK_API_KEY") -> str:
    """Read a key from the environment, falling back to known ``.env`` files."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    for path in ENV_CANDIDATES:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.startswith(f"{name}="):
                candidate = line.split("=", 1)[1].strip().strip('"').strip("'")
                if candidate:
                    return candidate
    raise LLMError(f"{name} not found in the environment or in {[str(p) for p in ENV_CANDIDATES]}")


@dataclass(slots=True)
class ChatClient:
    """Thin wrapper over an OpenAI-compatible ``/chat/completions`` endpoint."""

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 1500
    timeout: int = 120
    max_retries: int = 2

    @classmethod
    def from_config(cls, cfg: dict) -> ChatClient:
        key = load_api_key(cfg.get("api_key_env", "DEEPSEEK_API_KEY"))
        return cls(
            model=cfg.get("model", "deepseek-chat"),
            base_url=cfg.get("base_url", "https://api.deepseek.com"),
            api_key=key,
            temperature=float(cfg.get("temperature", 0.2)),
            max_tokens=int(cfg.get("max_tokens", 1500)),
            timeout=int(cfg.get("timeout", 120)),
        )

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        """Send a chat request, retrying transient failures with backoff."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code >= 400:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
                body = response.json()
                return body["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retry then surface
                last_error = exc
                if attempt <= self.max_retries:
                    time.sleep(2.0 * attempt)
        raise LLMError(f"chat failed after {self.max_retries + 1} attempts: {last_error}")

    def chat_json(self, messages: list[dict]) -> dict:
        """Chat in JSON mode and parse the reply, tolerating code fences."""
        raw = self.chat(messages, json_mode=True)
        return parse_json_reply(raw)


def parse_json_reply(raw: str) -> dict:
    """Parse a model reply that may be wrapped in ```json fences or prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise LLMError(f"reply is not valid JSON: {raw[:200]}")
