import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Study Planner", layout="wide")

# ---------------- SECRETS ----------------
# Ensure HF_API_TOKEN is in your Streamlit Cloud Secrets
try:
    HF_TOKEN = st.secrets["HF_API_TOKEN"]
except KeyError:
    st.error("Missing HF_API_TOKEN in Secrets!")
    st.stop()

# UPDATED MODELS: Switched to Mistral-7B for stability (Fixes 410 Error)
VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def query_api(url, payload, is_image=False):
    params = {"wait_for_model": "true"}
    if is_image:
        return requests.post(url, headers=HEADERS, data=payload, params=params)
    return requests.post(url, headers=HEADERS, json=payload, params=params)

def extract_info(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    res = query_api(VISION_API, buf.getvalue(), is_image=True)
    if res.status_code == 200:
        return res.json()[0].get("generated_text", "Image scanned.")
    return f"Vision Error: {res.status_code}"

def generate_plan(user_input):
    context = st.session_state.notes_text if st.session_state.notes_text else "General study notes"
    # Mistral Instruction Format
    prompt = f"<s>[INST] Context: {context}\nQuestion: {user_input}\nProvide a concise study plan with 3 tips. [/INST]"
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.7}
    }
    
    res = query_api(LLM_API, payload)
    if res.status_code == 200:
        # Mistral returns a list of dicts
        result = res.json()
        return result[0].get("generated_text", "").split("[/INST]")[-1].strip()
    elif res.status_code == 503:
        return "Model is starting up. Please wait 10 seconds and try again."
    return f"LLM Error {res.status_code}: Model is currently unavailable."

# ---------------- UI ----------------
st.title("Smart Study Planner")
st.caption("Multimodal AI Assistant • MACS AIML")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        # Fix for Streamlit 1.52.2 logs
        st.image(img, width=400) 
        if st.button("Analyze Notes"):
            with st.spinner("Analyzing..."):
                st.session_state.notes_text = extract_info(img)
                st.success("Done!")

with col2:
    st.subheader("Assistant")
    if st.session_state.notes_text:
        with st.expander("Detected Context"):
            st.write(st.session_state.notes_text)

    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.write(text)

    if q := st.chat_input("Ask for a study strategy..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.write(q)
        
        with st.chat_message("assistant"):
            with st.spinner("Writing plan..."):
                response = generate_plan(q)
                st.write(response)
                st.session_state.chat.append(("assistant", response))
