import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Study Planner", layout="wide")

# ---------------- SECRETS & API ----------------
# IMPORTANT: You MUST add HF_API_TOKEN to Streamlit Cloud Secrets!
try:
    HF_TOKEN = st.secrets["HF_API_TOKEN"]
except KeyError:
    st.error("Missing Secrets! Please add HF_API_TOKEN to your Streamlit Cloud App Settings.")
    st.stop()

# Using Gemma-2b-it (Active Model)
VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/google/gemma-1.1-2b-it"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state: st.session_state.notes_text = ""
if "chat" not in st.session_state: st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def process_image(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    res = requests.post(VISION_API, headers=HEADERS, data=buf.getvalue(), params={"wait_for_model": "true"})
    if res.status_code == 200:
        return res.json()[0].get("generated_text", "Image uploaded successfully.")
    return f"Vision Error: {res.status_code}"

def ask_assistant(user_query):
    context = st.session_state.notes_text if st.session_state.notes_text else "General study context."
    prompt = f"Context: {context}\nStudent: {user_query}\nAssistant: Provide a concise study plan."
    
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.7}}
    res = requests.post(LLM_API, headers=HEADERS, json=payload, params={"wait_for_model": "true"})
    
    if res.status_code == 200:
        return res.json()[0].get("generated_text", "").split("Assistant:")[-1].strip()
    return f"API Error ({res.status_code}). Model may be loading. Please try again."

# ---------------- UI ----------------
st.title("Smart Study Planner")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if img_file:
        img = Image.open(img_file)
        st.image(img, width='stretch') # Fixed the deprecation warning
        if st.button("Extract Info"):
            with st.spinner("Analyzing..."):
                st.session_state.notes_text = process_image(img)
                st.success("Extracted!")

with col2:
    st.subheader("Assistant")
    if st.session_state.notes_text:
        with st.expander("Detected Context"):
            st.info(st.session_state.notes_text)

    for role, text in st.session_state.chat:
        with st.chat_message(role): st.write(text)

    if q := st.chat_input("Ask for exam tips..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"): st.write(q)
        
        with st.chat_message("assistant"):
            ans = ask_assistant(q)
            st.write(ans)
            st.session_state.chat.append(("assistant", ans))
