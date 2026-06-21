"""
classifier.py - Analyzes incoming user messages and classifies them into one of
three customer personas: Technical Expert, Frustrated User, or Business Executive.
"""

import json
import os
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL


def classify_customer_persona(user_message: str) -> dict:
    """
    Analyzes the user's message and classifies it into one of the three target personas.

    Returns a dict with keys:
        - persona: str  ("Technical Expert" | "Frustrated User" | "Business Executive")
        - confidence: float (0.0 – 1.0)
        - reasoning: str
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    system_instruction = (
        "You are an advanced classification engine. Your task is to analyze the "
        "sentiment, vocabulary, and tone of an incoming customer support message and "
        "classify it into exactly one of three customer personas:\n\n"
        "1. 'Technical Expert': Uses technical jargon, asks about APIs, code, configurations, "
        "error codes, logs, headers, authentication tokens, database integrations, or system internals.\n"
        "2. 'Frustrated User': Uses emotional language, exclamation marks, expresses urgency, "
        "frustration, anger, or impatience. Mentions wasted time, repeated failures, or demands.\n"
        "3. 'Business Executive': Focuses on business impact, ROI, uptime, timelines, "
        "SLAs, operational efficiency, financial implications, and brief professional summaries.\n\n"
        "Return ONLY valid JSON. No explanation outside the JSON."
    )

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "persona": {
                "type": "STRING",
                "enum": ["Technical Expert", "Frustrated User", "Business Executive"]
            },
            "confidence": {"type": "NUMBER"},
            "reasoning": {"type": "STRING"}
        },
        "required": ["persona", "confidence", "reasoning"]
    }

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1
        )
    )

    return json.loads(response.text)


if __name__ == "__main__":
    test_cases = [
        "Our production API key stopped working with a 401 Unauthorized error. Check our logs.",
        "I've been waiting for an HOUR and nothing is working! This is absolutely ridiculous!!!",
        "Our operational uptime is declining. What is the resolution timeline for billing disputes?"
    ]
    for msg in test_cases:
        result = classify_customer_persona(msg)
        print(f"Message: {msg[:60]}...")
        print(f"Result:  {json.dumps(result, indent=2)}\n")
