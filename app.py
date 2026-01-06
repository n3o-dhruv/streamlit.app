import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Study Planner", layout="wide")

# ---------------- SECRETS ----------------
# IMPORTANT: Ensure HF_API_TOKEN is added to Streamlit Cloud Secrets
try:
    HF_TOKEN = st.secrets["HF_API_TOKEN"]
except KeyError:
    st.error("Error: 'HF_API_TOKEN' not found in Secrets. Please add it to Streamlit Settings.")
    st.stop()

# UPDATED MODELS: Switched to active endpoints to fix 410 Error
VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/google/gemma-1.1-2b-it"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def extract_info(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    try:
        res = requests.post(VISION_API, headers=HEADERS, data=buf.getvalue(), params={"wait_for_model": "true"})
        if res.status_code == 200:
            return res.json()[0].get("generated_text", "Image uploaded successfully.")
        return f"Vision Model Error ({res.status_code})"
    except:
        return "Connection to Vision Model failed."

def get_study_plan(user_query):
    context = st.session_state.notes_text if st.session_state.notes_text else "General study session"
    # Prompt format optimized for Gemma
    prompt = f"Context: {context}\nStudent Question: {user_query}\nAssistant Study Plan:"
    
    payload = {
        "inputs": prompt, 
        "parameters": {"max_new_tokens": 250, "temperature": 0.7, "return_full_text": False}
    }
    
    try:
        res = requests.post(LLM_API, headers=HEADERS, json=payload, params={"wait_for_model": "true"})
        if res.status_code == 200:
            return res.json()[0].get("generated_text", "").strip()
        elif res.status_code == 503:
            return "Model is loading on Hugging Face servers. Please wait 30 seconds and try again."
        return f"LLM Error ({res.status_code}): The model might be offline or rate-limited."
    except:
        return "Connection to Study Planner failed."

# ---------------- UI ----------------
st.title("Smart Study Planner")
st.caption("Multimodal Assistant • MACS AIML Project")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    file = st.file_uploader("Upload image (jpg/png)", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        # Fixed deprecation: using width='stretch' instead of use_container_width
        st.image(img, width='stretch')
        if st.button("Analyze Notes"):
            with st.spinner("Processing image..."):
                st.session_state.notes_text = extract_info(img)
                st.success("Notes Analyzed!")

with col2:
    st.subheader("Study Assistant")
    if st.session_state.notes_text:
        with st.expander("Detected Context"):
            st.write(st.session_state.notes_text)

    # Display Chat
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.write(text)

    if q := st.chat_input("Ask for a revision strategy..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.write(q)
        
        with st.chat_message("assistant"):
            with st.spinner("Creating your plan..."):
                ans = get_study_plan(q)
                st.write(ans)
                st.session_state.chat.append(("assistant", ans))
