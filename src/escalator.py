"""
escalator.py - Determines whether a conversation should be escalated to a
human agent, and generates a structured JSON handoff report if so.
"""

import json
from datetime import datetime

from src.config import (
    CONFIDENCE_THRESHOLD,
    FRUSTRATION_TURN_LIMIT,
    SENSITIVE_KEYWORDS,
)


def check_sensitive_keywords(query: str) -> str | None:
    """
    Checks if the query contains any sensitive topic keywords that require
    mandatory human escalation. Returns the matched keyword or None.
    """
    query_lower = query.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in query_lower:
            return keyword
    return None


def should_escalate(
    query: str,
    best_score: float,
    persona: str,
    frustration_turns: int,
    context_chunks: list,
) -> str | None:
    """
    Evaluates all escalation conditions.

    Returns a reason string if escalation is required, else None.

    Escalation triggers:
      1. Low retrieval confidence (below threshold)
      2. Sensitive billing/legal keywords detected
      3. Repeated frustration over multiple consecutive turns
    """
    # Trigger 1: Low retrieval confidence
    if best_score < CONFIDENCE_THRESHOLD or len(context_chunks) == 0:
        return f"Low retrieval confidence (score: {best_score:.3f} < threshold: {CONFIDENCE_THRESHOLD})"

    # Trigger 2: Sensitive topic detected
    matched_keyword = check_sensitive_keywords(query)
    if matched_keyword:
        return f"Sensitive topic detected: '{matched_keyword}'"

    # Trigger 3: Prolonged frustration
    if persona == "Frustrated User" and frustration_turns >= FRUSTRATION_TURN_LIMIT:
        return f"Repeated frustration detected over {frustration_turns} consecutive turns"

    return None


def generate_handoff_summary(
    user_query: str,
    persona: str,
    context_chunks: list,
    escalation_reason: str,
) -> str:
    """
    Generates a structured JSON handoff report for the human support agent.

    The report includes:
      - Customer persona and escalation reason
      - Truncated issue description
      - Retrieved knowledge base sources consulted
      - Best confidence score from retrieval
      - Timestamp and recommended next action
    """
    best_score = max((c["score"] for c in context_chunks), default=0.0)
    retrieved_sources = list({c["source"] for c in context_chunks})

    # Build recommended action based on persona
    if persona == "Technical Expert":
        recommended_action = (
            "Review system error logs, provide API configuration details, "
            "and confirm endpoint specifications with engineering team."
        )
    elif persona == "Frustrated User":
        recommended_action = (
            "Lead with empathy and a direct apology. Resolve the issue on the first contact. "
            "Escalate to billing team if financial disputes are involved."
        )
    else:  # Business Executive
        recommended_action = (
            "Assign a senior account manager immediately. Provide an executive-level "
            "resolution summary with SLA timeline and direct phone contact."
        )

    handoff_data = {
        "handoff_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "escalation_reason": escalation_reason,
            "assigned_priority": "HIGH" if "billing" in escalation_reason.lower() or "frustrated" in escalation_reason.lower() else "MEDIUM",
        },
        "customer_profile": {
            "detected_persona": persona,
            "issue_summary": user_query[:150] + ("..." if len(user_query) > 150 else ""),
        },
        "retrieval_diagnostics": {
            "knowledge_sources_consulted": retrieved_sources,
            "best_similarity_score": round(best_score, 4),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "chunks_retrieved": len(context_chunks),
        },
        "agent_guidance": {
            "recommended_action": recommended_action,
            "context_snippets": [
                {"source": c["source"], "preview": c["text"][:100] + "..."}
                for c in context_chunks[:2]
            ],
        },
    }

    return json.dumps(handoff_data, indent=2)
