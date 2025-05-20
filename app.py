import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from pymongo import MongoClient
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
import io
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile

# ---------------------- CONFIG ------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GOOGLE_API_KEY or not MONGO_URI:
    st.error("Set GOOGLE_API_KEY and MONGO_URI in .env")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
collection = db["chat_history"]

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# ---------------------- CSS ---------------------------
st.set_page_config(page_title="Gemini PDF Chat", layout="centered")
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1000px;
    }
    body {
        zoom: 1.15;
    }
    .stTextInput, .stFileUploader, .stCheckbox, .stButton {
        margin-bottom: 0.8rem;
    }
    .chat-message {
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- SESSION INIT --------------------
def initialize_session_state():
    defaults = {
        "messages": [{"role": "assistant", "content": "Hi! Upload a PDF or just start chatting."}],
        "pdf_text": "",
        "tables_summary": "",
        "image_captions_str": "",
        "extracted_images": [],
        "just_chat": False,
        "audio_data": None,
        "show_chat_history": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# ------------------ PDF PROCESSING ---------------------
def extract_text_from_pdf(file_bytes):
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Text Extraction Error: {e}")
        return ""

def extract_tables_from_pdf(file):
    tables = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    except Exception as e:
        st.error(f"Table Extraction Error: {e}")
    return tables

def extract_images_from_pdf(file_bytes):
    images_data = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                for img in page.get_images(full=True):
                    xref = img[0]
                    base = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base["image"]))
                    images_data.append({"page": i + 1, "image": image})
    except Exception as e:
        print(f"Image Extraction Error: {e}")
    return images_data

# --------------------- CHAT HISTORY ---------------------
def save_chat(role, message):
    collection.insert_one({"role": role, "message": message})

def load_chat():
    return list(collection.find({}, {"_id": 0}))

# ----------------- GEMINI INFERENCE ---------------------
chat_history = [{"role": h["role"].lower(), "parts": [h["message"]]} for h in load_chat()]
flash_model = genai.GenerativeModel("gemini-1.5-flash-latest")
vision_model = genai.GenerativeModel("gemini-1.5-flash-latest")
chat = flash_model.start_chat(history=chat_history)

def get_gemini_response(context, question, images, just_chat=False):
    if just_chat:
        return chat.send_message(question).text
    prompt = f"""--- PDF Text ---\n{context['text'][:4000]}...\n\n--- Tables ---\n{context['tables_summary']}\n\n--- Image Info ---\n{context['image_captions']}\n\nUser Question: {question}"""
    images_pil = [img["image"] for img in images]
    contents = [prompt] + images_pil
    return vision_model.generate_content(contents).text

# --------------------- AUDIO ----------------------------
def transcribe_audio(audio_bytes):
    try:
        model = load_whisper_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
        result = model.transcribe(temp_path)
        os.remove(temp_path)
        return result["text"]
    except Exception as e:
        st.error(f"Whisper Transcription Error: {e}")
        return None

# ----------------------- UI -----------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1000px;
    }
    body {
        zoom: 1.15;
    }
    .stTextInput, .stFileUploader, .stCheckbox, .stButton {
        margin-bottom: 0.8rem;
    }
    .chat-message {
        margin-bottom: 0.5rem;
    }
    .mic-container {
        display: flex;
        align-items: flex-end;
        height: 100%;
        margin-top:20px;
        padding-top:7px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Chatbot")

# Input and mic perfectly inline
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input("Ask me anything:", key="text_input")

with col2:
    with st.container():
        st.markdown('<div class="mic-container">', unsafe_allow_html=True)
        audio_data = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key="mic")
        st.markdown('</div>', unsafe_allow_html=True)


# Transcription display
voice_question = None
if audio_data:
    st.audio(audio_data['bytes'])
    with st.spinner("Transcribing..."):
        voice_question = transcribe_audio(audio_data['bytes'])
    if voice_question:
        st.success(f"You said: '{voice_question}'")
        if not question:
            question = voice_question

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

# Chat history toggle
st.session_state.show_chat_history = st.checkbox("📚 Show Chat History")

# Ask button
submit = st.button("Ask")

# PDF processing
if uploaded_file:
    file_bytes = uploaded_file.read()
    st.session_state.pdf_text = extract_text_from_pdf(file_bytes)

    uploaded_file.seek(0)
    tables = extract_tables_from_pdf(uploaded_file)
    st.session_state.tables_summary = "\n".join(
        [f"Table {i+1}:\n{df.to_string(index=False)}\n" for i, df in enumerate(tables)]
    )

    st.session_state.extracted_images = extract_images_from_pdf(file_bytes)
    captions = [f"Image found on Page {img['page']}" for img in st.session_state.extracted_images]
    st.session_state.image_captions_str = "\n".join(captions)

# Generate response
if question and submit:
    with st.spinner("Thinking..."):
        ctx = {
            "text": st.session_state.pdf_text,
            "tables_summary": st.session_state.tables_summary,
            "image_captions": st.session_state.image_captions_str
        }
        answer = get_gemini_response(ctx, question, st.session_state.extracted_images)

    save_chat("User", question)
    save_chat("Model", answer)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]

# Chat display
st.subheader("💬 Chat")
for msg in reversed(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat history
if st.session_state.show_chat_history:
    st.subheader("🕘 History")
    for h in reversed(load_chat()):
        with st.chat_message(h["role"]):
            st.markdown(h["message"])

client.close()
