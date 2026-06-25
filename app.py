#!/usr/bin/env python3
"""
A simple autonomous chatbot script that sends a predefined sequence of messages
to the OpenAI Chat Completion API (gpt-3.5-turbo) and prints the assistant's
responses. The script requires an API key in the environment variable
`CEREBRAS_API_KEY`. No manual input is needed; it runs end‑to‑end.
"""

import os
import sys
import json
import time
from typing import List, Dict

import requests

API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_api_key() -> str:
    """Retrieve the OpenAI API key from the environment."""
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        sys.stderr.write(
            "Error: environment variable CEREBRAS_API_KEY is not set.\n"
        )
        sys.exit(1)
    return key


def chat_completion(messages: List[Dict[str, str]]) -> str:
    """
    Send a chat completion request to the OpenAI API.

    Args:
        messages: List of message dicts with "role" and "content".

    Returns:
        The assistant's reply text.
    """
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                sys.stderr.write(f"API request failed after {MAX_RETRIES} attempts: {e}\n")
                sys.exit(1)
            else:
                sys.stderr.write(f"API request error (attempt {attempt}/{MAX_RETRIES}): {e}\n")
                time.sleep(RETRY_DELAY)


def main() -> None:
    # Predefined user messages the chatbot will "receive"
    user_messages = [
        "Hello!",
        "How are you today?",
        "Can you tell me a short joke?",
        "What is the capital of France?",
        "Thanks, that's all!"
    ]

    conversation: List[Dict[str, str]] = []

    for user_msg in user_messages:
        # Append user's message
        conversation.append({"role": "user", "content": user_msg})
        # Get assistant's reply
        assistant_reply = chat_completion(conversation)
        # Append assistant's reply to the history for context
        conversation.append({"role": "assistant", "content": assistant_reply})

        # Print the exchange
        print(f"User: {user_msg}")
        print(f"Assistant: {assistant_reply}")
        print("-" * 40)


if __name__ == "__main__":
    main()