import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Study Planner",
    layout="wide"
)

# ---------------- SECRETS ----------------
# Ensure HF_API_TOKEN is set in your Streamlit Secrets
HF_TOKEN = st.secrets["HF_API_TOKEN"]

# Models
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_MODEL_API = "https://api-inference.huggingface.co/models/google/flan-t5-base"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def query_hf_api(api_url, payload):
    """Generic wrapper to handle HF Inference API requests"""
    try:
        response = requests.post(
            api_url, 
            headers=HEADERS, 
            json=payload, 
            params={"wait_for_model": "true"}, 
            timeout=120
        )
        # Raise an error for bad status codes (4xx, 5xx)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def extract_notes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    
    # Payload for Vision model (raw bytes)
    try:
        response = requests.post(
            VISION_MODEL_API,
            headers=HEADERS,
            data=img_bytes.getvalue(),
            params={"wait_for_model": "true"}
        )
        if response.status_code == 200:
            return response.json()[0].get("generated_text", "Could not interpret image.")
    except:
        pass
    return "Error contacting vision model."

def generate_study_plan(user_query):
    # Constructing a very specific prompt for flan-t5-base
    context = st.session_state.notes_text if st.session_state.notes_text else "General studies"
    
    prompt = f"Context: {context}. Task: {user_query}. Provide a short study plan."
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    result = query_hf_api(LLM_MODEL_API, payload)
    
    if result and isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "No text generated.")
    return "The study planner model is currently busy. Please try again in a moment."

# ---------------- UI ----------------
st.title("Smart Study Planner")
st.caption("Transformer-based Multimodal AI using Hosted Inference Models")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Notes Image")
    image_file = st.file_uploader("Upload handwritten or printed notes", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Notes", use_container_width=True)

        if st.button("Analyze Notes"):
            with st.spinner("Extracting content..."):
                st.session_state.notes_text = extract_notes(image)
                st.success("Analysis complete!")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Study Planner Assistant")

    if st.session_state.notes_text:
        with st.expander("View Extracted Context"):
            st.write(st.session_state.notes_text)

    # Display chat history
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    # Chat input
    if user_input := st.chat_input("Ask for a revision strategy or exam tips..."):
        # Add user message to UI
        st.session_state.chat.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating plan..."):
                answer = generate_study_plan(user_input)
                st.write(answer)
                st.session_state.chat.append(("assistant", answer))

st.divider()
st.caption("MACS AIML • Multimodal Transformer Project")
