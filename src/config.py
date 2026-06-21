"""
config.py - Application configuration, thresholds, and constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Configuration ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "models/gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ─── RAG Pipeline Settings ────────────────────────────────────────────────────
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
TOP_K_RESULTS = 3
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "support_kb"

# ─── Escalation Thresholds ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40          # Below this → escalate to human
FRUSTRATION_TURN_LIMIT = 3           # Escalate after N frustrated turns in a row

# Keywords that always trigger escalation regardless of confidence
SENSITIVE_KEYWORDS = [
    "refund", "chargeback", "duplicate charge", "billing dispute",
    "legal", "lawsuit", "lawyer", "fraud", "unauthorized charge",
    "account terminated", "cancel account", "data deletion"
]

# ─── Data Directory ───────────────────────────────────────────────────────────
DATA_DIR = "./data"
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf"]

# ─── UI Configuration ─────────────────────────────────────────────────────────
APP_TITLE = "Persona-Adaptive Support Agent"
APP_ICON = "🤖"
MAX_CHAT_HISTORY = 20
