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
HF_TOKEN = st.secrets["HF_API_TOKEN"]

# Models
# BLIP for Image Captioning
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
# Switching to Llama-3.1 which is highly stable on the API
LLM_MODEL_API = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def query_hf_api(api_url, payload, is_image=False):
    """Generic wrapper with retry logic for model loading"""
    for _ in range(3):
        if is_image:
            response = requests.post(api_url, headers=HEADERS, data=payload)
        else:
            response = requests.post(api_url, headers=HEADERS, json=payload)
        
        # Handle 503 (Model loading)
        if response.status_code == 503:
            time.sleep(5)
            continue
        return response
    return response

def identify_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    
    response = query_hf_api(VISION_MODEL_API, img_bytes.getvalue(), is_image=True)

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except:
            return "Landmark detected but text could not be parsed."
    
    return f"Vision Error: {response.status_code}"

def get_travel_recommendation(prompt):
    # Llama-3 format
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    response = query_hf_api(LLM_MODEL_API, payload)

    if response.status_code == 200:
        output = response.json()
        if isinstance(output, list) and len(output) > 0:
            return output[0].get("generated_text", "No response.")
        elif isinstance(output, dict):
            return output.get("generated_text", "No response.")
    
    # Specific message for 410 or other errors
    if response.status_code == 410:
        return "Error 410: This model endpoint is no longer available. Please try a different model."
    
    return f"LLM Error: {response.status_code}. (Check if your API token is correct)."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Transformer-based Multimodal AI")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, use_container_width=True)

        if st.button("Identify Landmark"):
            with st.spinner("Analyzing image..."):
                st.session_state.landmark = identify_landmark(image)
                st.success(f"Detected: {st.session_state.landmark}")

with col2:
    st.subheader("Travel Chatbot")

    if st.session_state.landmark:
        st.info(f"Context: {st.session_state.landmark}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Ask for travel tips...")

    if user_input:
        st.session_state.chat.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Better prompt engineering
        full_prompt = (
            f"You are a helpful travel guide. The user is asking about: {st.session_state.landmark if st.session_state.landmark else 'general travel'}. "
            f"User question: {user_input}"
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_travel_recommendation(full_prompt)
                st.write(answer)
        
        st.session_state.chat.append(("assistant", answer))

st.divider()
