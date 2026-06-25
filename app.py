#!/usr/bin/env python3
"""
Simple chatbot using Cerebras API.

The script can be run without manual input. Provide the user's message as a
command‑line argument, e.g.:

    python3 chatbot.py "Tell me a joke"

If no argument is supplied, the script falls back to the environment variable
CHATBOT_MESSAGE, and if that is also missing it uses a built‑in default message.
An API key must be supplied via the environment variable CEREBRAS_API_KEY.
"""

import os
import sys
import json
import traceback

import requests

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"
DEFAULT_MESSAGE = "Hello, how are you?"

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def fatal(msg: str, exit_code: int = 1) -> None:
    """Print an error message to stderr and exit."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(exit_code)


def get_user_message() -> str:
    """Return the message to send to the model.

    Priority:
    1. Command‑line arguments (joined into a single string)
    2. Environment variable CHATBOT_MESSAGE
    3. Built‑in default message
    """
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    env_msg = os.getenv("CHATBOT_MESSAGE")
    if env_msg:
        return env_msg
    return DEFAULT_MESSAGE


def get_api_key() -> str:
    """Retrieve the Cerebras API key from the environment."""
    key = os.getenv("CEREBRAS_API_KEY")
    if not key:
        fatal(
            "CEREBRAS_API_KEY environment variable not set. "
            "Please set it to a valid Cerebras API key."
        )
    return key


def call_cerebras_api(message: str, api_key: str) -> str:
    """Send the user message to Cerebras and return the assistant's reply."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": message}],
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        # Network‑level failure – propagate the real error
        fatal(f"Network error while contacting Cerebras API: {e}")

    if response.status_code != 200:
        # The API returned an error – show details for debugging
        try:
            err_detail = response.json()
        except json.JSONDecodeError:
            err_detail = response.text
        fatal(
            f"Cerebras API returned HTTP {response.status_code}: {err_detail}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        fatal(f"Failed to parse JSON response from Cerebras API: {e}")

    # Expected structure: {"choices": [{"message": {"content": "..."}}, ...], ...}
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        fatal(f"Unexpected API response format: {e}\nFull response: {json.dumps(data, indent=2)}")

    return content.strip()


def main() -> None:
    user_message = get_user_message()
    api_key = get_api_key()
    reply = call_cerebras_api(user_message, api_key)
    print(reply)


if __name__ == "__main__":
    main()