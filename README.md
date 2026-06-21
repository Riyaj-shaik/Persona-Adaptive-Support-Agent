# 🤖 Persona-Adaptive Customer Support Agent

An intelligent AI-powered customer support agent that classifies the communication style of incoming messages and adapts its tone, depth, and format accordingly — using **Google Gemini** for LLM inference, **ChromaDB** for vector search, and **Streamlit** for the interactive UI.

---

## 🏗️ Architectural Overview

```
[User Message]
      │
      ▼
[Persona Classifier] ──── Gemini LLM (structured JSON output)
      │
      │  Persona: Technical Expert / Frustrated User / Business Executive
      ▼
[RAG Pipeline]
  ├── Document Ingestion (TXT, MD, PDF → chunks → embeddings)
  ├── ChromaDB Vector Store (cosine similarity index)
  └── Semantic Retrieval (top-K chunks via cosine similarity)
      │
      ▼
[Escalation Check]
  ├── Low retrieval confidence? (score < 0.40)
  ├── Sensitive keywords? (billing, refund, legal...)
  └── Repeated frustration? (N consecutive turns)
      │
      ├── YES → [Generate Handoff JSON] → Human Agent
      └── NO  → [Adaptive Prompt Engine] → [Gemini LLM] → Response
```

---

## 📁 Project Structure

```
persona-support-agent/
│
├── data/
│   ├── api_troubleshooting.md      # API auth, rate limits, webhook, cookies
│   ├── billing_policy.txt          # Pricing, refunds, disputes, cancellations
│   └── password_reset_guide.pdf    # Password reset, 2FA, locked accounts, SSO
│
├── src/
│   ├── __init__.py
│   ├── config.py        # Centralized configuration and thresholds
│   ├── classifier.py    # Persona detection via Gemini structured output
│   ├── rag_pipeline.py  # Document parsing, chunking, embedding, ChromaDB
│   ├── generator.py     # Persona-adaptive prompt builder + LLM caller
│   └── escalator.py     # Escalation logic + structured handoff JSON generator
│
├── app.py               # Main Streamlit Web UI
├── requirements.txt     # pip dependencies
├── .env.example         # API key template (copy to .env)
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

- Python 3.11 or higher
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### 2. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/persona-support-agent.git
cd persona-support-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure API Key

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real Gemini API key
```

Or simply paste your key into the sidebar input when the app is running.

### 4. Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 How to Use

1. **Enter your Gemini API Key** in the left sidebar.
2. **Click "Load & Index Knowledge Base"** — this parses the documents in `data/`, creates embeddings, and stores them in ChromaDB. This only runs once; subsequent sessions use the persisted index.
3. **Type any support question** in the chat input or click one of the **Quick Test Prompts** in the sidebar.
4. The agent will automatically:
   - Classify your persona (Technical / Frustrated / Executive)
   - Retrieve relevant knowledge base chunks
   - Generate a tailored response — or escalate with a JSON handoff if needed

---

## 🧪 Test Scenarios

| Message | Expected Persona | Expected Behavior |
|---|---|---|
| *"Where is the guide to clear cookies? It's been an hour!"* | Frustrated User | Empathetic tone, simple bullet steps |
| *"What are the header parameter requirements for bearer token auth?"* | Technical Expert | Code blocks, precise header specs |
| *"Our uptime is declining. What is the billing dispute timeline?"* | Business Executive | Short, professional, timeline-focused |
| *"I'm getting 500 errors on my database integration."* | Technical Expert | Step-by-step diagnostic pathways |
| *"I have duplicate charges and demand an immediate refund!"* | Frustrated User + Escalation | Human handoff JSON generated |

---

## 🔧 Key Configuration (`src/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 400 | Characters per document chunk |
| `CHUNK_OVERLAP` | 40 | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | 3 | Number of chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | 0.40 | Min cosine similarity before escalation |
| `FRUSTRATION_TURN_LIMIT` | 3 | Max consecutive frustrated turns |

---

## 📐 Key Technical Concepts

### Cosine Similarity
Used to measure semantic closeness between the user query and document chunks:

$$\text{Similarity}(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|}$$

Values closer to 1.0 indicate high relevance; values below 0.40 trigger escalation.

### Retrieval-Augmented Generation (RAG)
Prevents hallucination by grounding the LLM's response in factual knowledge base content. The LLM is explicitly instructed to answer **only** from the retrieved context.

### Persona Classification
Uses Gemini's structured JSON output feature to reliably return one of three personas plus a confidence score and reasoning — eliminating unreliable free-text parsing.

---

## 📄 License

MIT License — feel free to use and adapt for your own projects.
