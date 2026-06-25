#!/usr/bin/env python3
"""
A minimal autonomous chatbot that uses the Cerebras chat completion API.

The script reads the API key from the environment variable CEREBRAS_API_KEY.
Optionally, a custom prompt can be supplied via CEREBRAS_PROMPT; otherwise a
default prompt is used. The response from the model is printed to stdout.
"""

import os
import sys
import json

import requests


def main() -> None:
    # ----------------------------------------------------------------------
    # Configuration / environment
    # ----------------------------------------------------------------------
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        sys.stderr.write("Error: CEREBRAS_API_KEY environment variable not set.\n")
        sys.exit(1)

    # Prompt can be overridden via env var; a static default keeps the script
    # fully non‑interactive.
    prompt = os.getenv("CEREBRAS_PROMPT", "Hello, who are you?")

    endpoint = "https://api.cerebras.ai/v1/chat/completions"
    model = "gpt-oss-120b"

    # ----------------------------------------------------------------------
    # Build request payload (OpenAI‑compatible format)
    # ----------------------------------------------------------------------
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ----------------------------------------------------------------------
    # Send request
    # ----------------------------------------------------------------------
    try:
        response = requests.post(
            endpoint, headers=headers, json=payload, timeout=30
        )
    except requests.RequestException as exc:
        sys.stderr.write(f"Network error while contacting Cerebras API: {exc}\n")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Check HTTP status
    # ----------------------------------------------------------------------
    if response.status_code != 200:
        sys.stderr.write(
            f"API error: HTTP {response.status_code} – {response.text}\n"
        )
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Parse JSON response
    # ----------------------------------------------------------------------
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Failed to decode JSON response: {exc}\n")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Extract assistant reply (OpenAI‑style response schema)
    # ----------------------------------------------------------------------
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        sys.stderr.write(
            f"Unexpected response format: {exc}\nFull response: {data}\n"
        )
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Output the result
    # ----------------------------------------------------------------------
    print(content)


if __name__ == "__main__":
    main()