import streamlit as st
import requests
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Travel Recommendation Chatbot",
    layout="wide"
)

# ---------------- SECRETS ----------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- OPENROUTER CALL ----------------
def get_travel_recommendation(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.app",
        "X-Title": "Travel Recommendation Chatbot"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful travel guide."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.6
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    # SHOW REAL ERROR (important for sanity)
    return f"Error from OpenRouter: {response.status_code} - {response.text}"

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Hosted LLM using OpenRouter (Stable)")

# Chat history
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

# Input
user_input = st.chat_input("Ask about travel tips, budget, or attractions...")

if user_input:
    st.session_state.chat.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            answer = get_travel_recommendation(user_input)
            st.write(answer)

    st.session_state.chat.append(("assistant", answer))
    st.rerun()

# Clear
if st.button("Clear Conversation"):
    st.session_state.chat = []
    st.rerun()
