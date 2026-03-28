import os
import streamlit as st
from groq import Groq

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nexus Chat", page_icon="💬", layout="centered")

# ── Groq client ───────────────────────────────────────────────────────────────
# os.environ.get("GROQ_API_KEY") reads the key from Streamlit Cloud's Secrets
# instead of having it written here. This keeps your key safe on GitHub.
client = Groq(api_key=os.environ.get(api_key="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"))

# ── System prompt ─────────────────────────────────────────────────────────────
# This is the instruction we give the AI before every conversation.
# It defines how Nexus should behave — context retention, honesty, etc.
SYSTEM_PROMPT = """You are Nexus, a helpful and conversational AI assistant.

Follow these rules on every reply:
1. CONTEXT  — Always use the conversation history to understand follow-ups.
              If the user asked about Python earlier and now says "what are its pros?",
              you know "its" refers to Python — no need for them to repeat themselves.
2. HONESTY  — Only say things you are confident about. If you are unsure, say
              "I'm not 100% sure, but..." instead of guessing.
3. CLARIFY  — If a question is too vague to answer well, ask one short
              clarifying question before giving a full answer.
4. NO REPS  — Never repeat something you already said. Build on prior answers.
5. TONE     — Be warm, clear, and concise. No filler phrases."""

# ── Token management ──────────────────────────────────────────────────────────
# The AI can only process a limited amount of text at once (its "context window").
# We track roughly how many tokens (words/pieces) the history uses, and trim
# older messages when we get close to the limit.
MAX_TOKENS    = 5000
CHARS_PER_TOK = 4   # roughly 4 characters = 1 token

def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOK

def trim_history(history: list) -> list:
    # Walk backwards (newest first) keeping messages until the budget runs out.
    # We always preserve the very first message so the bot never forgets
    # what topic the conversation started on.
    total = 0
    kept  = []
    for msg in reversed(history):
        cost = estimate_tokens(msg["content"])
        if total + cost > MAX_TOKENS:
            break
        kept.insert(0, msg)
        total += cost
    if history and (not kept or kept[0] is not history[0]):
        kept.insert(0, history[0])
    return kept

# ── Session state ─────────────────────────────────────────────────────────────
# Streamlit reruns the whole script on every user action, so we use
# session_state to remember the conversation between those reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("💬 Nexus")
st.caption("Context-aware chatbot powered by Groq (free) · Llama 3")

# Show all previous messages on screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle user input ─────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything...")

if user_input:
    # 1. Save and display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Trim history to stay within the token budget
    history_to_send = trim_history(st.session_state.messages)

    # 3. System prompt goes first, then the trimmed conversation history
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history_to_send

    # 4. Call Groq and stream the reply so it appears word by word
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply  = ""

        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # free, powerful open-source model by Meta
            messages=api_messages,
            max_tokens=1000,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply += delta
                placeholder.markdown(full_reply + "▌")  # ▌ looks like a blinking cursor

        placeholder.markdown(full_reply)  # finalise — remove the cursor

    # 5. Save the reply so it becomes part of future context
    st.session_state.messages.append({"role": "assistant", "content": full_reply})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Info")
    total_msgs   = len(st.session_state.messages)
    total_tokens = sum(estimate_tokens(m["content"]) for m in st.session_state.messages)
    st.metric("Messages",     total_msgs)
    st.metric("~Tokens used", total_tokens)
    st.metric("Budget left",  max(0, MAX_TOKENS - total_tokens))
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("<small>Free tier · Groq · Llama 3.3 70B</small>", unsafe_allow_html=True)
