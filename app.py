#!/usr/bin/env python3
"""
Cerebras Chatbot CLI

Usage:
    python chatbot.py "Your message here"

The script sends the provided message to the Cerebras chat completions API
using the model "gpt-oss-120b" and prints the assistant's reply.

Requirements:
    - The environment variable CEREBRAS_API_KEY must be set.
    - The `requests` library must be installed.
"""

import os
import sys
import json
import traceback

import requests

API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"


def fatal(message: str, exit_code: int = 1) -> None:
    """Print an error message to stderr and exit."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(exit_code)


def main() -> None:
    # Retrieve the API key from the environment
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        fatal("environment variable CEREBRAS_API_KEY is not set.")

    # Retrieve the user prompt from command‑line arguments
    if len(sys.argv) < 2:
        fatal("no prompt provided. Usage: python chatbot.py \"Your message\"")
    user_prompt = " ".join(sys.argv[1:]).strip()
    if not user_prompt:
        fatal("the provided prompt is empty.")

    # Build request payload according to the OpenAI‑compatible schema
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        fatal(f"network error while contacting Cerebras API: {e}")

    if response.status_code != 200:
        try:
            error_detail = response.json()
        except json.JSONDecodeError:
            error_detail = response.text
        fatal(
            f"Cerebras API returned error {response.status_code}: {error_detail}"
        )

    try:
        data = response.json()
        # Expected structure: {"choices": [{"message": {"content": "..."} }], ...}
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        fatal(f"unexpected response format: {e}\nResponse body: {response.text}")

    # Output the assistant's reply
    print(content)


if __name__ == "__main__":
    main()