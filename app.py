#!/usr/bin/env python3
"""
Cerebras Chatbot Script

This script sends a single user message to the Cerebras chat completion API
and prints the assistant's reply.

Usage:
    python3 chatbot.py "Your message here"

The script requires the environment variable CEREBRAS_API_KEY to be set.
"""

import os
import sys
import json
import requests

# Constants for the Cerebras API
API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"


def get_api_key() -> str:
    """Retrieve the Cerebras API key from the environment."""
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return api_key


def build_payload(user_message: str) -> dict:
    """Construct the JSON payload for the API request."""
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_message}
        ]
    }


def call_cerebras_api(api_key: str, payload: dict) -> dict:
    """Make the POST request to the Cerebras chat completion endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        # Try to provide useful error details from the response body
        try:
            error_info = response.json()
            print(f"Error: API returned status {response.status_code}: {error_info}", file=sys.stderr)
        except Exception:
            print(f"Error: API returned status {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    try:
        return response.json()
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON response: {e}", file=sys.stderr)
        sys.exit(1)


def extract_reply(api_response: dict) -> str:
    """Extract the assistant's reply from the API response."""
    try:
        # The structure follows OpenAI's schema: choices[0].message.content
        return api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Error: Unexpected response structure: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 chatbot.py \"Your message here\"", file=sys.stderr)
        sys.exit(1)

    user_message = sys.argv[1]
    api_key = get_api_key()
    payload = build_payload(user_message)
    api_response = call_cerebras_api(api_key, payload)
    reply = extract_reply(api_response)

    print(reply)


if __name__ == "__main__":
    main()