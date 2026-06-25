#!/usr/bin/env python3
"""
A minimal chatbot that uses the Cerebras API.

It reads the user's prompt from the environment variable ``CHAT_PROMPT`` (default
provided) and the API key from ``CEREBRAS_API_KEY``. The script performs a single
POST request to the Cerebras chat completion endpoint and prints the assistant's
reply to stdout.

The script is fully self‑contained and exits with a non‑zero status on any
error, exposing the real problem (missing key, network failure, bad response,
etc.).
"""

import json
import os
import sys
from typing import Any, Dict

import requests

# ----------------------------------------------------------------------
# Configuration (do not modify unless you know what you are doing)
# ----------------------------------------------------------------------
API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"

def fatal(msg: str, exit_code: int = 1) -> None:
    """Print an error message to stderr and exit."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(exit_code)

def get_api_key() -> str:
    """Retrieve the Cerebras API key from the environment."""
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        fatal("Environment variable CEREBRAS_API_KEY is not set.")
    return key.strip()

def get_prompt() -> str:
    """Retrieve the user prompt from the environment (or use a default)."""
    return os.environ.get("CHAT_PROMPT", "Hello, how are you?").strip()

def call_cerebras_chat(api_key: str, prompt: str) -> str:
    """
    Send a chat request to the Cerebras API and return the assistant's answer.

    Raises:
        RuntimeError: on non‑200 HTTP status or malformed response.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as exc:
        fatal(f"Network request failed: {exc}")

    if response.status_code != 200:
        # Try to surface any error details returned by the API.
        try:
            err_data = response.json()
            err_msg = err_data.get("error", response.text)
        except Exception:
            err_msg = response.text
        fatal(
            f"API returned error {response.status_code}: {err_msg}",
            exit_code=response.status_code,
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        fatal(f"Failed to decode JSON response: {exc}")

    # Expected structure: {"choices": [{"message": {"content": "..."}}, ...], ...}
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        fatal(f"Unexpected response format: {exc}\nResponse body: {data}")

    return content.strip()

def main() -> None:
    api_key = get_api_key()
    prompt = get_prompt()

    answer = call_cerebras_chat(api_key, prompt)

    # Output the assistant's reply.
    print(answer)

if __name__ == "__main__":
    main()