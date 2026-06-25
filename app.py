#!/usr/bin/env python3
"""
A minimal chatbot script that uses the Cerebras chat completion API.
It sends a fixed user prompt to the model and prints the model's reply.
No interactive input is required; the API key is read from the environment.
"""

import os
import sys
import json
import requests

API_ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"


def get_api_key() -> str:
    """Retrieve the Cerebras API key from the environment."""
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        print("Error: environment variable CEREBRAS_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    return key


def build_payload(prompt: str) -> dict:
    """Create the JSON payload for the chat completion request."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return {
        "model": MODEL_NAME,
        "messages": messages,
        # Optional parameters can be added here, e.g., temperature, max_tokens
    }


def call_cerebras_api(key: str, payload: dict) -> dict:
    """Send the request to the Cerebras API and return the parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as exc:
        print(f"Network error while contacting Cerebras API: {exc}", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        # Try to include any error message returned by the API
        try:
            err_detail = response.json()
        except Exception:
            err_detail = response.text
        print(
            f"API request failed with status {response.status_code}: {err_detail}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        print(f"Failed to parse JSON response: {exc}", file=sys.stderr)
        sys.exit(1)


def extract_reply(api_response: dict) -> str:
    """Extract the assistant's reply from the API response."""
    try:
        # OpenAI-compatible format: response["choices"][0]["message"]["content"]
        return api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        print(f"Unexpected API response structure: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    # Fixed prompt – replace or extend as needed
    user_prompt = "Hello, who are you?"
    api_key = get_api_key()
    payload = build_payload(user_prompt)
    api_response = call_cerebras_api(api_key, payload)
    reply = extract_reply(api_response)
    print(reply)


if __name__ == "__main__":
    main()