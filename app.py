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
# Ensure your HF_API_TOKEN is correctly set in .streamlit/secrets.toml
HF_TOKEN = st.secrets["HF_API_TOKEN"]

# UPDATED MODELS: flan-t5-base (410 error) replaced with Mistral-7B
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_MODEL_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def extract_notes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    
    try:
        response = requests.post(
            VISION_MODEL_API,
            headers=HEADERS,
            data=img_bytes.getvalue(),
            params={"wait_for_model": "true"}
        )
        if response.status_code == 200:
            return response.json()[0].get("generated_text", "No text detected.")
    except Exception as e:
        return f"Vision Error: {str(e)}"
    return "Vision model currently unavailable."

def generate_study_plan(user_query):
    # Constructing a prompt that Mistral understands
    context = st.session_state.notes_text if st.session_state.notes_text else "General topics"
    
    # Prompt format for Mistral Instruct
    prompt = f"<s>[INST] Context: {context}\nQuestion: {user_query}\nProvide a concise study plan with 3 key tips. [/INST]"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(
            LLM_MODEL_API,
            headers=HEADERS,
            json=payload,
            params={"wait_for_model": "true"}
        )
        if response.status_code == 200:
            result = response.json()
            return result[0].get("generated_text", "I'm not sure how to answer that.")
        elif response.status_code == 410:
            return "Error: This model endpoint is no longer available on HF. Try 'google/gemma-1.1-2b-it'."
        else:
            return f"API Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Connection error: {str(e)}"

# ---------------- UI ----------------
st.title("Smart Study Planner")
st.caption("Multimodal AI Study Assistant")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    image_file = st.file_uploader("Upload notes image", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Reading notes..."):
                st.session_state.notes_text = extract_notes(image)
                st.success("Analysis complete!")

with col2:
    st.subheader("Study Assistant")

    if st.session_state.notes_text:
        with st.expander("Extracted Context"):
            st.info(st.session_state.notes_text)

    # Display chat
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    # Input handling
    if user_input := st.chat_input("How can I help you study?"):
        st.session_state.chat.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                answer = generate_study_plan(user_input)
                st.markdown(answer)
                st.session_state.chat.append(("assistant", answer))
