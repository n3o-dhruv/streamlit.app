import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Study Planner",
    layout="wide"
)

# ================= UI HEADER =================
st.title("Smart Study Planner")
st.caption("Multimodal AI Assistant • MACS AIML")

# ================= SECRETS =================
if "HF_API_TOKEN" not in st.secrets:
    st.error("HF_API_TOKEN missing in Streamlit Secrets")
    st.stop()

HF_TOKEN = st.secrets["HF_API_TOKEN"]

# ================= MODELS =================
VISION_MODEL = "Salesforce/blip-image-captioning-large"
TEXT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ================= CLIENTS =================
vision_client = InferenceClient(
    model=VISION_MODEL,
    token=HF_TOKEN
)

text_client = InferenceClient(
    model=TEXT_MODEL,
    token=HF_TOKEN
)

# ================= SESSION STATE =================
if "notes" not in st.session_state:
    st.session_state.notes = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= IMAGE ANALYSIS =================
def analyze_image(img):
    try:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        result = vision_client.image_to_text(buffer)
        return result["generated_text"]

    except Exception:
        return "⚠️ Image model busy or network issue."

# ================= STUDY PLAN =================
def get_study_plan(query):
    prompt = f"""
You are an expert academic study planner.
Create a clear, simple 5-step study plan.

Topic:
{query}
"""
    try:
        response = text_client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.5
        )
        return response

    except Exception:
        return "⚠️ Text model busy or network issue. Retry."

# ================= LAYOUT =================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)

        if st.button("Analyze Notes"):
            with st.spinner("Analyzing notes..."):
                st.session_state.notes = analyze_image(image)
                st.success("Notes analyzed!")

with col2:
    st.subheader("Assistant")

    if st.session_state.notes:
        st.info(st.session_state.notes)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_query = st.chat_input("Ask for a study strategy...")

    if user_query:
        st.session_state.chat.append(("user", user_query))

        with st.chat_message("assistant"):
            with st.spinner("Generating study plan..."):
                reply = get_study_plan(user_query)
                st.write(reply)
                st.session_state.chat.append(("assistant", reply))
