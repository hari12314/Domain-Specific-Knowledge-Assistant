# TicketBot — Domain-Specific Knowledge Assistant

Private chatbot for the ticket booking domain. Answers only from its knowledge base with confidence scoring, hallucination control, and automatic escalation for unknown questions.

## What Makes It Domain-Specific

| Feature | Generic RAG | TicketBot |
|---|---|---|
| Knowledge | Any topic | Ticket booking only |
| Guardrail | None | Domain keyword filter |
| Confidence | Not shown | FAISS score → high/medium/low |
| Unknown Q | Hallucination | I don't know + contact support |
| Escalation | Never | Auto-flag low confidence answers |

## Domain Coverage

- Movie bookings — formats, pricing, seat selection
- Bus tickets — operators, seat types, routes
- Concerts and sports events
- Cancellation and refund policies
- Payment methods and failure handling
- Offers, discounts, and membership
- Customer support and escalation

## Tech Stack

- Python
- Groq API (Free)
- LangChain + LangChain Text Splitters
- FAISS (local vector store)
- HuggingFace all-MiniLM-L6-v2 (local embeddings)
- Streamlit

## Files

- `app.py` — Streamlit chatbot with confidence badges and guardrails
- `TicketBot_RAG.ipynb` — Step-by-step notebook (13 sections)
- `requirements.txt` — All dependencies

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Get Free API Key
https://console.groq.com
