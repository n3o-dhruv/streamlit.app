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
# Ensure your HF_API_TOKEN is in .streamlit/secrets.toml
HF_TOKEN = st.secrets["HF_API_TOKEN"]

# Stable Model Endpoints
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
# Zephyr is highly reliable and excellent for travel advice
LLM_MODEL_API = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def query_hf_api(api_url, payload, is_image=False):
    """Handles API calls with automatic retries for model loading (503)"""
    for attempt in range(5):  # Increased retries for stability
        if is_image:
            response = requests.post(api_url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, data=payload)
        else:
            response = requests.post(api_url, headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            return response
        
        # If model is loading, wait and retry
        if response.status_code == 503:
            wait_time = response.json().get("estimated_time", 10)
            time.sleep(min(wait_time, 10))
            continue
        else:
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
            return "Landmark identified but response format unknown."
    
    return f"Vision Error: {response.status_code}"

def get_travel_recommendation(prompt):
    # Zephyr prompt format (ChatML-ish)
    formatted_prompt = f"<|system|>\nYou are a professional travel guide assistant. Give concise and helpful advice.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "top_p": 0.95,
            "return_full_text": False
        }
    }

    response = query_hf_api(LLM_MODEL_API, payload)

    if response.status_code == 200:
        output = response.json()
        if isinstance(output, list) and len(output) > 0:
            return output[0].get("generated_text", "No response received.")
        elif isinstance(output, dict):
            return output.get("generated_text", "No response received.")
    
    # Specific handling for the 410 error to help debugging
    if response.status_code == 410:
        return "The chosen model is currently unavailable on Hugging Face. Please contact the developer to update the model endpoint."
    
    return f"LLM Error: {response.status_code}. The service may be busy."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Transformer-based Multimodal AI using Zephyr-7B")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader("Upload an image of a landmark", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Current Upload", use_container_width=True)

        if st.button("Identify Landmark"):
            with st.spinner("Analyzing landmark..."):
                result = identify_landmark(image)
                st.session_state.landmark = result
                st.success(f"Context: {result}")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Travel Chatbot")

    if st.session_state.landmark:
        st.info(f"📍 Talking about: {st.session_state.landmark}")

    # Display Chat
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Ask about travel tips, budget, or attractions...")

    if user_input:
        st.session_state.chat.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        # Build prompt with landmark context
        if st.session_state.landmark:
            context_prompt = f"The user is looking at a picture of {st.session_state.landmark}. {user_input}"
        else:
            context_prompt = user_input

        with st.chat_message("assistant"):
            with st.spinner("Finding recommendations..."):
                answer = get_travel_recommendation(context_prompt)
                st.write(answer)
        
        st.session_state.chat.append(("assistant", answer))

# Add a Clear Button
if st.button("Clear Conversation"):
    st.session_state.chat = []
    st.session_state.landmark = ""
    st.rerun()

st.divider()
