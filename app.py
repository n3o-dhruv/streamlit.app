import streamlit as st
import requests
from PIL import Image
import io
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Travel Recommendation Chatbot",
    layout="wide"
)

# ---------------- SECRETS & CONFIG ----------------
# Ensure HF_API_TOKEN is set in your Streamlit Secrets
HF_TOKEN = st.secrets["HF_API_TOKEN"]

# Models
# BLIP for Image Captioning
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
# Mistral is much better for conversational "Travel Guide" logic than flan-t5
LLM_MODEL_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def query_hf_api(api_url, payload, is_image=False):
    """Generic wrapper to handle model loading and errors"""
    for _ in range(3):  # Retry up to 3 times if model is loading
        if is_image:
            response = requests.post(api_url, headers=HEADERS, data=payload)
        else:
            response = requests.post(api_url, headers=HEADERS, json=payload)
        
        result = response.json()
        
        # Handle model loading state
        if response.status_code == 503 and "estimated_time" in result:
            wait_time = result.get("estimated_time", 10)
            time.sleep(min(wait_time, 5)) # Wait a bit and retry
            continue
        return response
    return response

def identify_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG") # JPEG is generally smaller/faster
    
    response = query_hf_api(VISION_MODEL_API, img_bytes.getvalue(), is_image=True)

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except (KeyError, IndexError):
            return "Landmark detected but text could not be parsed."
    
    return f"Vision Error: {response.status_code}"

def get_travel_recommendation(prompt):
    # Mistral/Llama models expect specific prompt formatting, but simple text works too
    payload = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = query_hf_api(LLM_MODEL_API, payload)

    if response.status_code == 200:
        try:
            output = response.json()
            # Different models return different structures; this covers the most common
            if isinstance(output, list) and len(output) > 0:
                return output[0].get("generated_text", "No text generated.")
            elif isinstance(output, dict):
                return output.get("generated_text", "No text generated.")
        except Exception as e:
            return f"Parsing Error: {str(e)}"

    return f"LLM Error: {response.status_code}. The model might be offline."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Transformer-based Multimodal AI using Hosted Inference Models")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader(
        "Upload an image of a landmark",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Landmark Image", use_container_width=True)

        if st.button("Identify Landmark"):
            with st.spinner("Identifying landmark..."):
                result = identify_landmark(image)
                st.session_state.landmark = result
                st.success(f"Result: {result}")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Travel Chatbot")

    if st.session_state.landmark:
        st.info(f"**Context:** Looking for info about: {st.session_state.landmark}")

    # Display Chat History
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input(
        "Ask about best time to visit, budget, or tips..."
    )

    if user_input:
        # Add user message to history
        st.session_state.chat.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Generate Response
        context = f"Context: The user is looking at a photo of '{st.session_state.landmark}'. Question: {user_input}. As a travel guide, provide a concise, helpful answer."
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting travel guides..."):
                answer = get_travel_recommendation(context)
                st.write(answer)
        
        st.session_state.chat.append(("assistant", answer))

st.divider()
st.caption("MACS AIML • Multimodal Transformer Project")
