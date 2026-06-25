#!/usr/bin/env python3
"""
A minimal Cerebras chatbot script.

It reads the prompt from command‑line arguments, sends it to the Cerebras
Chat Completion API and prints the assistant's reply.

The script exits with a non‑zero status on any error, including a missing
CEREBRAS_API_KEY environment variable.
"""

import sys
import os
import json
import logging
from typing import List

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def fatal(message: str) -> None:
    """Print an error message to stderr and exit with status 1."""
    sys.stderr.write(f"ERROR: {message}\n")
    sys.exit(1)


def get_api_key() -> str:
    """Fetch the Cerebras API key from the environment."""
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        fatal("CEREBRAS_API_KEY environment variable not set")
    return key


def build_payload(prompt: str) -> dict:
    """Create the JSON payload expected by the Cerebras chat endpoint."""
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }


def call_cerebras_api(key: str, payload: dict) -> dict:
    """POST the payload to the Cerebras API and return the parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        fatal(f"Network error while contacting Cerebras API: {exc}")

    if response.status_code != 200:
        # Attempt to surface possible error details from the service.
        try:
            error_detail = response.json()
        except json.JSONDecodeError:
            error_detail = response.text
        fatal(
            f"Cerebras API returned HTTP {response.status_code}: {error_detail}"
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        fatal(f"Failed to decode JSON response from Cerebras API: {exc}")


def extract_reply(api_response: dict) -> str:
    """Extract the assistant's content from the Cerebras API response."""
    try:
        choices: List[dict] = api_response["choices"]
        if not choices:
            fatal("Cerebras API response contains no choices")
        # The API returns a list; we take the first entry.
        message = choices[0]["message"]
        content = message["content"]
        return content.strip()
    except (KeyError, TypeError) as exc:
        fatal(f"Unexpected response structure from Cerebras API: {exc}")


def main() -> None:
    """Entry point."""
    if len(sys.argv) < 2:
        fatal(
            "No prompt supplied. Provide the user prompt as command‑line arguments."
        )

    # Join all arguments to form the full prompt (preserves spaces).
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        fatal("Prompt is empty after stripping whitespace.")

    api_key = get_api_key()
    payload = build_payload(prompt)
    response_json = call_cerebras_api(api_key, payload)
    reply = extract_reply(response_json)

    print(reply)


if __name__ == "__main__":
    # Optional: configure basic logging (useful for debugging in CI)
    logging.basicConfig(level=logging.ERROR)
    main()