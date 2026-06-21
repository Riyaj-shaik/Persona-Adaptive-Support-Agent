"""
generator.py - Builds persona-specific system prompts, calls the Gemini LLM,
and returns either a grounded response or an escalation payload.
"""

import os
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL, CONFIDENCE_THRESHOLD
from src.escalator import should_escalate, generate_handoff_summary


# ── Persona System Prompt Templates ───────────────────────────────────────────

PERSONA_INSTRUCTIONS = {
    "Technical Expert": (
        "You are a Senior Systems Engineer and API Specialist. "
        "Communicate with precision and depth. Provide exact root-cause analysis, "
        "configuration parameters, HTTP header specifications, code blocks, error code "
        "interpretations, and step-by-step diagnostic pathways. "
        "Structure your answer with clear numbered steps and inline code where applicable. "
        "Assume the user has strong technical knowledge — skip basic explanations."
    ),
    "Frustrated User": (
        "You are a deeply empathetic and reassuring Customer Care Specialist. "
        "The user is stressed and needs to feel heard first. ALWAYS begin your response "
        "with a genuine acknowledgment of their frustration (1-2 sentences). "
        "Then provide a clear, simple, action-oriented solution using short bullet points. "
        "Use plain, friendly language. Avoid all technical jargon. "
        "End with a warm reassurance that their issue will be resolved."
    ),
    "Business Executive": (
        "You are a concise Client Relations Director speaking to a senior business leader. "
        "Lead with the direct answer and resolution summary in the first sentence. "
        "Include: estimated resolution timeline, business impact summary, and next steps — "
        "in that order. Keep the total response under 120 words. "
        "Omit configuration details and technical terminology entirely. "
        "Use professional, executive-ready language."
    ),
}


def generate_adaptive_response(
    user_query: str,
    persona: str,
    context_chunks: list,
    chat_history: list = None,
    frustration_turns: int = 0,
) -> dict:
    """
    Generates a persona-appropriate response grounded in retrieved context.

    Returns:
        {
            "escalated": bool,
            "persona": str,
            "response": str,
            "confidence": float,
            "handoff_summary": str | None
        }
    """
    # ── 1. Escalation Check ───────────────────────────────────────────────────
    best_score = max((c["score"] for c in context_chunks), default=0.0)

    escalation_reason = should_escalate(
        query=user_query,
        best_score=best_score,
        persona=persona,
        frustration_turns=frustration_turns,
        context_chunks=context_chunks,
    )

    if escalation_reason:
        handoff = generate_handoff_summary(user_query, persona, context_chunks, escalation_reason)
        return {
            "escalated": True,
            "persona": persona,
            "response": (
                "I sincerely apologize — I want to make sure you get the best possible help here. "
                "I'm connecting you with a live specialist who can fully resolve this for you right away. "
                "Your case details have been shared with them so you won't need to repeat yourself."
            ),
            "confidence": best_score,
            "handoff_summary": handoff,
        }

    # ── 2. Build Context-Grounded System Prompt ───────────────────────────────
    persona_instructions = PERSONA_INSTRUCTIONS.get(
        persona, PERSONA_INSTRUCTIONS["Frustrated User"]
    )

    context_text = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks]
    )

    system_prompt = (
        f"{persona_instructions}\n\n"
        "════════════════════════════════════════\n"
        "CRITICAL OPERATING RULES:\n"
        "1. Answer ONLY using information found in the CONTEXT DOCUMENTS below.\n"
        "2. Do NOT hallucinate facts, steps, or URLs not present in the documents.\n"
        "3. If the context does not contain sufficient information to answer, "
        "say: 'I don't have enough information on that specific topic — "
        "let me connect you with a specialist.'\n"
        "════════════════════════════════════════\n\n"
        f"CONTEXT DOCUMENTS:\n{context_text}"
    )

    # ── 3. Build Conversation Messages ────────────────────────────────────────
    messages = []
    if chat_history:
        for turn in chat_history[-6:]:   # Keep last 3 exchanges for context
            messages.append({"role": turn["role"], "parts": [{"text": turn["content"]}]})
    messages.append({"role": "user", "parts": [{"text": user_query}]})

    # ── 4. Call Gemini LLM ────────────────────────────────────────────────────
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=1024,
        )
    )

    return {
        "escalated": False,
        "persona": persona,
        "response": response.text,
        "confidence": best_score,
        "handoff_summary": None,
    }
