import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import sqlite3
import fitz
from PIL import Image
import io
import datetime
import whisper
import subprocess
import tempfile
from streamlit_mic_recorder import mic_recorder

# OPTIONAL: Add ffmpeg to PATH if not already
ffmpeg_path = r"C:\ffmpeg\ffmpeg-2025-05-19-git-c55d65ac0a-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Verify ffmpeg
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
except Exception as e:
    st.warning(f"FFmpeg not detected or failed to run: {e}")

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API key not found. Please set GOOGLE_API_KEY in .env")
    st.stop()

# Streamlit theme setup
st.set_page_config(page_title="Document Chatbot with Gemini Flash", layout="centered")
st.config.set_option("theme.base", "dark")
st.config.set_option("theme.primaryColor", "#673ab7")
st.config.set_option("theme.secondaryBackgroundColor", "#303030")
st.config.set_option("theme.textColor", "#ffffff")
st.config.set_option("theme.font", "sans serif")

@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")

model = load_gemini_model(GOOGLE_API_KEY)

# SQLite setup
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        message TEXT,
        timestamp TEXT
    )
''')
conn.commit()

@st.cache_data(show_spinner=False)
def save_chat(role, message):
    cursor.execute(
        "INSERT INTO chat_history (role, message, timestamp) VALUES (?, ?, ?)",
        (role, message, datetime.datetime.now().isoformat())
    )
    conn.commit()

@st.cache_data
def load_chat():
    cursor.execute("SELECT role, message FROM chat_history ORDER BY id ASC")
    return cursor.fetchall()

def clear_chat():
    cursor.execute("DELETE FROM chat_history")
    conn.commit()

def process_document_and_ask_gemini(uploaded_file, question):
    try:
        file_bytes = uploaded_file.read()
        mime_type = uploaded_file.type
        contents = [
            {"role": "user", "parts": [question]},
            {"role": "user", "parts": [{"mime_type": mime_type, "data": file_bytes}]},
        ]
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        return f"Error processing document: {e}"

def get_general_chat_response(prompt, chat):
    response = chat.send_message(prompt)
    return response.text

def transcribe_audio_whisper(audio_bytes):
    try:
        model = whisper.load_model("base")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_file_name = tmp_file.name

        result = model.transcribe(temp_file_name)
        os.remove(temp_file_name)
        return result["text"]
    except FileNotFoundError as e:
        st.error("❌ ffmpeg not found. Please ensure it’s installed and added to PATH.")
        return None
    except Exception as e:
        st.error(f"Whisper transcription error: {e}")
        return None

# UI Start
st.title("📄 Document Q&A with Gemini Flash")
st.caption("Chat with PDFs, images, or just talk using your voice!")

tab1, tab2 = st.tabs(["📑 Document Q&A", "💬 General Chat"])

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF or image file", type=["pdf", "png", "jpg", "jpeg"])

    audio_data = mic_recorder(start_prompt="🎤 Record", stop_prompt="🛑 Stop", key="mic_doc")
    voice_question = None

    if audio_data:
        st.audio(audio_data["bytes"])
        with st.spinner("Transcribing voice..."):
            voice_question = transcribe_audio_whisper(audio_data["bytes"])
        if voice_question:
            st.write(f"You asked (via voice): {voice_question}")

    question = st.text_input("Ask a question about the document:", value=voice_question if voice_question else "")
    submit = st.button("Ask", key="doc_ask")

    if uploaded_file and submit and question:
        with st.spinner("Analyzing document..."):
            response = process_document_and_ask_gemini(uploaded_file, question)

        save_chat("User", question)
        save_chat("Model", response)

        st.subheader("🤖 Gemini's Answer")
        st.write(response)

with tab2:
    st.subheader("💬 General Chat")

    if "general_chat_history" not in st.session_state:
        st.session_state["general_chat_history"] = [{"role": role.lower(), "parts": [msg]} for role, msg in load_chat()]
        st.session_state["general_chat"] = model.start_chat(history=st.session_state["general_chat_history"])

    chat_placeholder = st.container()
    input_placeholder = st.container()

    with chat_placeholder:
        st.markdown("### 🧾 Conversation")
        for message in st.session_state["general_chat_history"]:
            role = message["role"]
            msg = message["parts"][0]
            st.markdown(f"<div style='padding:10px;border-radius:10px;margin-bottom:5px'><strong>{'You' if role == 'user' else 'Gemini'}:</strong><br>{msg}</div>", unsafe_allow_html=True)

    with input_placeholder:
        audio_data_gen = mic_recorder(start_prompt="🎤 Record", stop_prompt="🛑 Stop", key="mic_gen")
        voice_input = None

        if audio_data_gen:
            st.audio(audio_data_gen["bytes"])
            with st.spinner("Transcribing voice..."):
                voice_input = transcribe_audio_whisper(audio_data_gen["bytes"])
            if voice_input:
                st.write(f"You asked (via voice): {voice_input}")

        chat_input = st.text_input("Type your message:", value=voice_input if voice_input else "", key="chat_input")
        send_chat = st.button("Send", key="chat_send")

        if send_chat and chat_input:
            with st.spinner("Thinking..."):
                reply = get_general_chat_response(chat_input, st.session_state["general_chat"])

            save_chat("User", chat_input)
            save_chat("Model", reply)

            st.session_state["general_chat_history"].append({"role": "user", "parts": [chat_input]})
            st.session_state["general_chat_history"].append({"role": "model", "parts": [reply]})
            st.session_state["general_chat"].history.append({"role": "user", "parts": [chat_input]})
            st.session_state["general_chat"].history.append({"role": "model", "parts": [reply]})

            st.rerun()

with st.sidebar:
    st.subheader("📜 Chat History")
    search_query = st.text_input("Search messages:")

    if st.button("🗑 Clear Chat History"):
        clear_chat()
        st.session_state["general_chat_history"] = []
        st.session_state["general_chat"] = model.start_chat(history=[])
        st.success("Chat history cleared.")

    history = load_chat()
    if search_query:
        history = [msg for msg in history if search_query.lower() in msg[1].lower()]

    for role, msg in history:
        st.markdown(f"{role}: {msg}")