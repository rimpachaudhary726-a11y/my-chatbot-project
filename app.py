#!/usr/bin/env python3
"""
Simple chatbot using Cerebras API.

The script expects:
- Environment variable CEREBRAS_API_KEY containing a valid Cerebras API key.
- A user prompt supplied as the first command‑line argument.

It sends the prompt to the Cerebras chat completion endpoint and prints the
assistant's reply to stdout.

If any required information is missing or an error occurs, the script prints
the error message to stderr and exits with a non‑zero status code.
"""

import os
import sys
import json
import argparse

import requests  # external dependency

CEREBRAS_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = "gpt-oss-120b"


def get_api_key() -> str:
    """Retrieve the Cerebras API key from environment."""
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: environment variable CEREBRAS_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    return api_key


def build_payload(prompt: str) -> dict:
    """Create the JSON payload for the chat request."""
    return {
        "model": CEREBRAS_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }


def call_cerebras_api(api_key: str, payload: dict) -> str:
    """Send a request to the Cerebras chat completion API and return the assistant's reply."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(CEREBRAS_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        print(f"Network error while contacting Cerebras API: {exc}", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        # Attempt to surface any error details from the response body.
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        print(
            f"Error from Cerebras API (status {response.status_code}): {error_detail}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        data = response.json()
        # Expected structure: {"choices": [{"message": {"content": "..."}}, ...], ...}
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"Unexpected response format: {exc}", file=sys.stderr)
        sys.exit(1)

    return content.strip()


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Cerebras chatbot")
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text to send to the chatbot. If omitted, an empty string is sent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Join all positional arguments to form the prompt (preserves spaces)
    prompt = " ".join(args.prompt)

    api_key = get_api_key()
    payload = build_payload(prompt)
    reply = call_cerebras_api(api_key, payload)

    print(reply)


if __name__ == "__main__":
    main()