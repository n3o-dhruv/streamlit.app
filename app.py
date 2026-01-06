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
HF_TOKEN = st.secrets["HF_API_TOKEN"]

# UPDATED MODELS: Switched to active endpoints
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
# Gemma is a modern, active replacement for the retired Flan-T5
LLM_MODEL_API = "https://api-inference.huggingface.co/models/google/gemma-1.1-2b-it"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def query_hf(api_url, payload):
    response = requests.post(api_url, headers=HEADERS, json=payload, params={"wait_for_model": "true"})
    return response.json()

def extract_notes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    
    response = requests.post(
        VISION_MODEL_API,
        headers=HEADERS,
        data=img_bytes.getvalue(),
        params={"wait_for_model": "true"}
    )
    
    if response.status_code == 200:
        return response.json()[0].get("generated_text", "Image uploaded but no text read.")
    return "Vision model error."

def generate_study_plan(user_input):
    # Context matters for the model to give better tips
    context = st.session_state.notes_text if st.session_state.notes_text else "Student notes"
    
    # Gemma works best with a clear instruction format
    prompt = f"Context: {context}\nQuestion: {user_input}\nAssistant: Provide a clear, helpful study plan and exam tips."
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.7}
    }
    
    result = query_hf(LLM_MODEL_API, payload)
    
    # Extracting text from Gemma response format
    if isinstance(result, list) and len(result) > 0:
        full_text = result[0].get("generated_text", "")
        # Clean up the output to remove the prompt part if it's returned
        return full_text.split("Assistant:")[-1].strip()
    return "The model is currently busy. Please wait a few seconds and try again."

# ---------------- UI ----------------
st.title("Smart Study Planner")
st.caption("MACS AIML • Multimodal Transformer Project")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    image_file = st.file_uploader("Upload your schedule or notes", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, use_container_width=True)

        if st.button("Analyze Notes"):
            with st.spinner("Analyzing image content..."):
                st.session_state.notes_text = extract_notes(image)
                st.success("Analysis complete!")

with col2:
    st.subheader("Study Assistant")

    # Show what the model "saw" in the image
    if st.session_state.notes_text:
        with st.expander("Extracted Content"):
            st.write(st.session_state.notes_text)

    # Display Chat History
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    # User Input
    if user_query := st.chat_input("Ask for exam tips or a schedule..."):
        st.session_state.chat.append(("user", user_query))
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Generating plan..."):
                answer = generate_study_plan(user_query)
                st.write(answer)
                st.session_state.chat.append(("assistant", answer))
