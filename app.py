import streamlit as st
import os
import time
import tempfile
from datetime import datetime
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="TicketBot", page_icon="T", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:     #0a0510;
    --s1:     #110a1a;
    --s2:     #1a1128;
    --s3:     #221636;
    --border: #2e2048;
    --acc:    #e040fb;
    --acc2:   #ff6d00;
    --green:  #69f0ae;
    --blue:   #40c4ff;
    --yellow: #ffd740;
    --text:   #ede7f6;
    --muted:  #7c6f9a;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.hero {
    background: linear-gradient(135deg, #110a1a 0%, #1a0d2e 60%, #200a35 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--acc), var(--blue), var(--acc2));
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(224,64,251,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 900;
    color: white;
    letter-spacing: -1px;
    margin: 0;
}

.hero-title span { color: var(--acc); }
.hero-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted); margin-top: 0.4rem; letter-spacing: 0.05em; }

.chat-area {
    height: 540px;
    overflow-y: auto;
    padding: 1.2rem;
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.msg-user { display: flex; justify-content: flex-end; }

.msg-bot { display: flex; gap: 0.7rem; align-items: flex-start; }

.bubble-user {
    background: linear-gradient(135deg, var(--acc), #9c27b0);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 72%;
    font-size: 0.9rem;
    line-height: 1.65;
    font-weight: 500;
}

.bubble-bot {
    background: var(--s2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.2rem;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.75;
}

.bot-avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--acc), var(--acc2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; font-size: 0.88rem; color: white;
    flex-shrink: 0; margin-top: 2px;
}

.conf-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    margin: 3px 3px 0 0;
    font-weight: 600;
}

.conf-high   { background: #0a3320; color: var(--green);  border: 1px solid #1a5530; }
.conf-medium { background: #2a2000; color: var(--yellow); border: 1px solid #4a3800; }
.conf-low    { background: #2a1500; color: var(--acc2);   border: 1px solid #4a2500; }
.conf-none   { background: #2a0a30; color: var(--acc);    border: 1px solid #4a1a50; }

.src-pill {
    display: inline-block;
    background: var(--s3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    margin: 2px 2px 0 0;
}

.escalate-notice {
    background: #1a0a08;
    border: 1px solid #4a1a10;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--acc2);
    margin-top: 0.4rem;
    display: inline-block;
}

.domain-card {
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.35rem;
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.kb-stat {
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}

.kb-num { font-size: 1.5rem; font-weight: 900; color: var(--acc); line-height: 1; }
.kb-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

.stTextInput input {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.92rem !important;
}

.stTextInput input:focus { border-color: var(--acc) !important; }

.stSelectbox > div > div {
    background: var(--s2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--acc)22, var(--blue)22) !important;
    color: var(--acc) !important;
    border: 1px solid var(--acc)66 !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    width: 100% !important;
}

.stButton > button:hover { background: linear-gradient(135deg, var(--acc)44, var(--blue)44) !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# Session init
for k, v in [("messages",[]),("vs",None),("kb_built",False),("chunks",0),("doc_names",[])]:
    if k not in st.session_state: st.session_state[k] = v

DOMAIN_KEYWORDS = [
    'ticket','book','booking','cancel','refund','seat','event','movie','concert',
    'bus','sport','show','payment','price','offer','discount','promo','membership',
    'venue','schedule','app','login','account','support','help','reschedule',
    'transfer','ticketapp','cinema','theatre','stadium','pass','entry','qr',
]

DOMAIN_AREAS = [
    ("Movies", "#e040fb", "movie cinema format imax"),
    ("Bus Tickets", "#40c4ff", "bus route seat operator"),
    ("Concerts", "#ffd740", "concert event live show"),
    ("Sports", "#69f0ae", "stadium sports cricket football"),
    ("Cancellation", "#ff6d00", "refund cancel policy"),
    ("Support", "#ab47bc", "help contact escalation"),
]

CONFIDENCE_HIGH   = 0.5
CONFIDENCE_MEDIUM = 1.0
CONFIDENCE_LOW    = 1.5

@st.cache_resource(show_spinner=False)
def get_emb():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

def load_file(f):
    suffix = ".pdf" if f.type == "application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.read()); path=tmp.name
    loader = PyPDFLoader(path) if suffix==".pdf" else TextLoader(path, encoding="utf-8")
    pages = loader.load()
    domain = f.name.replace(".txt","").replace(".pdf","").replace("_"," ").title()
    for p in pages:
        p.metadata["source"] = f.name
        p.metadata["domain"] = domain
    os.unlink(path)
    return pages

def build_index(docs, cs, co, emb):
    sp = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=co, separators=["\n\n","\n",". "," ",""])
    chunks = sp.split_documents(docs)
    vs = FAISS.from_documents(chunks, emb)
    return vs, len(chunks)

def is_domain_q(q):
    return any(kw in q.lower() for kw in DOMAIN_KEYWORDS)

def retrieve_conf(vs, q, k=4):
    results = vs.similarity_search_with_score(q, k=k)
    if not results: return [], "none", 9.9
    best  = results[0][1]
    chunks = [r[0] for r in results]
    if best < CONFIDENCE_HIGH:   conf = "high"
    elif best < CONFIDENCE_MEDIUM: conf = "medium"
    elif best < CONFIDENCE_LOW:    conf = "low"
    else:                          conf = "none"
    return chunks, conf, round(best, 4)

def get_answer(client, model, vs, question, history, temperature):
    t0 = time.time()
    in_domain = is_domain_q(question)
    chunks, confidence, score = retrieve_conf(vs, question)

    escalate = False

    if not in_domain or confidence == "none":
        if not in_domain:
            answer = ("I am TicketBot and can only help with ticket booking questions. "
                      "For other queries, please use a general search engine. "
                      "Do you have a booking-related question I can help with?")
        else:
            answer = ("I don't have specific information about that in my knowledge base. "
                      "Please contact support at support@ticketapp.com or call 1800-123-4567.")
        escalate = True
        return {"answer":answer,"confidence":confidence,"sources":[],"in_domain":in_domain,
                "should_escalate":escalate,"score":score,"time":round(time.time()-t0,2)}

    context = "\n\n".join(["[" + c.metadata.get("domain","?") + "] " + c.page_content for c in chunks])

    history_text = "No previous conversation."
    if history:
        lines = [("User: " if h["role"]=="user" else "Bot: ") + h["content"][:150] for h in history[-4:]]
        history_text = "\n".join(lines)

    system = (
        "You are TicketBot, expert customer support for TicketApp ticket booking platform.\n"
        "RULES:\n"
        "1. Answer ONLY from the CONTEXT provided. No outside knowledge.\n"
        "2. If not in context, say: I don't have specific information about that. "
        "Contact support@ticketapp.com or call 1800-123-4567.\n"
        "3. Never guess prices, deadlines, or policies.\n"
        "4. Be concise (2-4 sentences) unless step-by-step instructions needed.\n"
        "5. Always end with a helpful next step.\n"
    )

    user_prompt = (
        "CONTEXT:\n\n" + context +
        "\n\n---\n\nHISTORY:\n" + history_text +
        "\n\n---\n\nQUESTION: " + question + "\n\nANSWER:"
    )

    msgs = [{"role":"system","content":system}]
    if history:
        msgs += [{"role":h["role"],"content":h["content"]} for h in history[-4:]]
    msgs.append({"role":"user","content":user_prompt})

    r = client.chat.completions.create(model=model, messages=msgs, temperature=temperature, max_tokens=450)
    answer = r.choices[0].message.content.strip()
    escalate = confidence == "low"

    return {"answer":answer,"confidence":confidence,
            "sources":list(set(c.metadata.get("source","") for c in chunks)),
            "in_domain":in_domain,"should_escalate":escalate,
            "score":score,"time":round(time.time()-t0,2)}

# SIDEBAR
with st.sidebar:
    st.markdown('<div style="font-family:Nunito,sans-serif;font-size:1.3rem;font-weight:900;color:#ede7f6;">Ticket<span style="color:#e040fb;">Bot</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#7c6f9a;margin-bottom:1rem;">Domain-Specific RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**API Key**")
    st.caption("Free at: https://console.groq.com")
    api_key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")

    st.markdown("**Model**")
    model = st.selectbox("", ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"], label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.markdown("**Chunking**")
    chunk_size    = st.slider("Chunk Size",  100, 600, 300, 50)
    chunk_overlap = st.slider("Overlap",       0, 100,  40, 10)
    top_k         = st.slider("Top K",          1,   6,   4)

    st.markdown("---")
    st.markdown("**Domain Coverage**")
    for name, color, _ in DOMAIN_AREAS:
        st.markdown(
            '<div class="domain-card">'
            '<div style="width:8px;height:8px;border-radius:50%;background:' + color + ';flex-shrink:0;"></div>'
            '<span style="font-size:0.8rem;">' + name + '</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

# HEADER
st.markdown(
    '<div class="hero"><h1 class="hero-title">Ticket<span>Bot</span></h1>'
    '<p class="hero-sub">DOMAIN-SPECIFIC RAG &nbsp;|&nbsp; HALLUCINATION CONTROL &nbsp;|&nbsp; CONFIDENCE SCORING &nbsp;|&nbsp; GUARDRAILS</p></div>',
    unsafe_allow_html=True)

left, right = st.columns([1, 1.5], gap="large")

# LEFT
with left:
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#e040fb;text-transform:uppercase;letter-spacing:0.12em;">Knowledge Base</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["pdf","txt"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded:
        if st.button("Build Domain Index"):
            if not api_key:
                st.error("Enter your Groq API key.")
            else:
                with st.spinner("Loading embeddings..."):
                    emb = get_emb()
                all_pages = []
                with st.spinner("Reading domain documents..."):
                    for f in uploaded: all_pages.extend(load_file(f))
                with st.spinner("Building FAISS index..."):
                    vs, nc = build_index(all_pages, chunk_size, chunk_overlap, emb)
                st.session_state.update({"vs":vs,"kb_built":True,"chunks":nc,
                                         "doc_names":[f.name for f in uploaded],"messages":[]})
                st.success("Domain knowledge base ready!")

    if st.session_state["kb_built"]:
        c1, c2 = st.columns(2)
        with c1: st.markdown('<div class="kb-stat"><div class="kb-num">' + str(len(st.session_state["doc_names"])) + '</div><div class="kb-lbl">Docs</div></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="kb-stat"><div class="kb-num">' + str(st.session_state["chunks"]) + '</div><div class="kb-lbl">Chunks</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#e040fb;text-transform:uppercase;letter-spacing:0.12em;">Guardrails</p>', unsafe_allow_html=True)

    guardrails = [
        ("#69f0ae", "Domain filter", "Only booking questions answered"),
        ("#69f0ae", "Confidence check", "FAISS score threshold applied"),
        ("#69f0ae", "I don't know", "No hallucination on unknown facts"),
        ("#ffd740", "Low confidence", "Adds escalation suggestion"),
        ("#e040fb", "Out of domain", "Redirects to relevant topics"),
    ]
    for color, title, desc in guardrails:
        st.markdown(
            '<div style="background:var(--s2);border:1px solid var(--border);border-radius:8px;'
            'padding:0.5rem 0.8rem;margin-bottom:0.3rem;display:flex;gap:0.7rem;align-items:center;">'
            '<div style="width:7px;height:7px;border-radius:50%;background:' + color + ';flex-shrink:0;"></div>'
            '<div><strong style="font-size:0.8rem;color:#ede7f6;">' + title + '</strong>'
            '<span style="font-size:0.73rem;color:#7c6f9a;margin-left:0.5rem;">' + desc + '</span></div>'
            '</div>', unsafe_allow_html=True)

# RIGHT
with right:
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#e040fb;text-transform:uppercase;letter-spacing:0.12em;">Chat</p>', unsafe_allow_html=True)

    chat_html = '<div class="chat-area">'

    if not st.session_state["messages"]:
        kb_status = ("Knowledge base loaded! Ask me anything about booking." if st.session_state["kb_built"]
                     else "Please upload your domain knowledge base on the left to get started.")
        chat_html += (
            '<div class="msg-bot"><div class="bot-avatar">TB</div>'
            '<div class="bubble-bot">'
            'Hi! I am <strong>TicketBot</strong>, your ticket booking assistant.<br><br>'
            'I can help with:<br>'
            '&nbsp;&nbsp;- Movie, bus, concert and sports bookings<br>'
            '&nbsp;&nbsp;- Cancellation and refund policies<br>'
            '&nbsp;&nbsp;- Payment issues and technical support<br>'
            '&nbsp;&nbsp;- Offers, discounts and membership<br><br>'
            + kb_status +
            '</div></div>'
        )
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                chat_html += '<div class="msg-user"><div class="bubble-user">' + msg["content"] + '</div></div>'
            else:
                conf      = msg.get("confidence","?")
                conf_cls  = "conf-" + conf
                sources   = msg.get("sources",[])
                escalate  = msg.get("should_escalate", False)
                t         = msg.get("time","")
                score     = msg.get("score","")

                src_tags = "".join(['<span class="src-pill">' + s + '</span>' for s in sources])
                escalate_html = '<div class="escalate-notice">Contact support@ticketapp.com or 1800-123-4567</div>' if escalate else ""

                chat_html += (
                    '<div class="msg-bot"><div class="bot-avatar">TB</div>'
                    '<div style="max-width:85%;">'
                    '<div class="bubble-bot">' + msg["content"].replace("\n","<br>") + '</div>'
                    + escalate_html +
                    '<div style="margin-top:0.4rem;display:flex;flex-wrap:wrap;align-items:center;gap:4px;">'
                    '<span class="conf-badge ' + conf_cls + '">Confidence: ' + conf + '</span>'
                    '<span style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#7c6f9a;">'
                    + str(t) + 's</span>'
                    + src_tags +
                    '</div></div></div>'
                )

    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input
    c1, c2 = st.columns([6, 1])
    with c1:
        user_input = st.text_input("", placeholder="Ask about booking, cancellation, payments, offers...", label_visibility="collapsed", key="tb_input")
    with c2:
        send_btn = st.button("Send")

    if (send_btn or user_input) and user_input.strip():
        if not api_key:
            st.error("Enter your Groq API key.")
        elif not st.session_state["kb_built"]:
            st.warning("Upload and build knowledge base first.")
        else:
            groq_client = Groq(api_key=api_key)
            history = [{"role":m["role"],"content":m["content"]} for m in st.session_state["messages"]]
            with st.spinner("TicketBot is thinking..."):
                try:
                    result = get_answer(groq_client, model, st.session_state["vs"],
                                        user_input.strip(), history, temperature)
                    st.session_state["messages"].append({"role":"user","content":user_input.strip()})
                    st.session_state["messages"].append({
                        "role":"assistant","content":result["answer"],
                        "confidence":result["confidence"],"sources":result["sources"],
                        "should_escalate":result["should_escalate"],
                        "time":result["time"],"score":result["score"],
                    })
                    st.rerun()
                except Exception as e:
                    st.error("Error: " + str(e))

    # Suggested questions
    if st.session_state["kb_built"] and not st.session_state["messages"]:
        suggestions = [
            "How do I book a movie ticket?",
            "What is the bus ticket refund policy?",
            "Is there a student discount?",
            "How long does payment refund take?",
        ]
        st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#7c6f9a;margin-top:0.6rem;">Suggested questions:</p>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        for i, (sc, q) in enumerate(zip([sc1,sc2,sc1,sc2], suggestions)):
            with sc:
                if st.button(q, key="sug_"+str(i)):
                    if api_key:
                        groq_client = Groq(api_key=api_key)
                        with st.spinner("Thinking..."):
                            try:
                                r = get_answer(groq_client, model, st.session_state["vs"], q, [], temperature)
                                st.session_state["messages"].append({"role":"user","content":q})
                                st.session_state["messages"].append({
                                    "role":"assistant","content":r["answer"],
                                    "confidence":r["confidence"],"sources":r["sources"],
                                    "should_escalate":r["should_escalate"],
                                    "time":r["time"],"score":r["score"],
                                })
                                st.rerun()
                            except Exception as e:
                                st.error("Error: " + str(e))
