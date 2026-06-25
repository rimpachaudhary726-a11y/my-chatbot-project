#!/usr/bin/env python3
"""
Simple Cerebras chatbot script.

It sends a single user message to the Cerebras chat completion endpoint and prints
the assistant's reply.

The script is fully non‑interactive: all configuration is taken from environment
variables.

Required environment variables
------------------------------
CEREBRAS_API_KEY : your Cerebras API key (required)
CEREBRAS_PROMPT : the user message to send (optional, defaults to a greeting)

If CEREBRAS_API_KEY is missing the script exits with a clear error message.
"""

import os
import sys
import json

try:
    import requests
except ImportError as exc:
    sys.stderr.write("Missing required package 'requests'. Install it via pip.\n")
    sys.exit(1)


API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"


def get_api_key() -> str:
    """Retrieve the Cerebras API key from the environment."""
    key = os.getenv("CEREBRAS_API_KEY")
    if not key:
        sys.stderr.write("Error: CEREBRAS_API_KEY environment variable not set.\n")
        sys.exit(1)
    return key


def build_payload(user_message: str) -> dict:
    """Create the JSON payload expected by the Cerebras API."""
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }


def call_cerebras_api(api_key: str, payload: dict) -> dict:
    """Send the request to Cerebras and return the parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        sys.stderr.write(f"Network error while contacting Cerebras API: {exc}\n")
        sys.exit(1)

    if not response.ok:
        # Attempt to include any error details returned by the API
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text
        sys.stderr.write(
            f"API request failed with status {response.status_code}: {error_detail}\n"
        )
        sys.exit(1)

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Failed to decode JSON response: {exc}\n")
        sys.exit(1)


def extract_reply(api_response: dict) -> str:
    """Extract the assistant's reply from the API response."""
    try:
        return api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        sys.stderr.write(f"Unexpected API response format: {exc}\n")
        sys.stderr.write(f"Full response: {json.dumps(api_response, indent=2)}\n")
        sys.exit(1)


def main() -> None:
    api_key = get_api_key()
    user_message = os.getenv("CEREBRAS_PROMPT", "Hello! Who are you?")
    payload = build_payload(user_message)
    api_response = call_cerebras_api(api_key, payload)
    reply = extract_reply(api_response)
    print(reply)


if __name__ == "__main__":
    main()