import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import sqlite3
import datetime
import whisper
import subprocess
import tempfile
from streamlit_mic_recorder import mic_recorder

# Add ffmpeg to PATH if needed
ffmpeg_path = r"C:\ffmpeg\ffmpeg-2025-05-19-git-c55d65ac0a-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Check ffmpeg
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
except Exception as e:
    st.warning(f"FFmpeg not detected or failed to run: {e}")

# Load Gemini API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("API key not found. Please set GOOGLE_API_KEY in .env")
    st.stop()

# Streamlit theme
st.set_page_config(page_title="Gemini All-in-One Chat", layout="centered")
st.config.set_option("theme.base", "dark")
st.config.set_option("theme.primaryColor", "#673ab7")
st.config.set_option("theme.secondaryBackgroundColor", "#303030")
st.config.set_option("theme.textColor", "#ffffff")
st.config.set_option("theme.font", "sans serif")

# Load Gemini model
@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")

model = load_gemini_model(GOOGLE_API_KEY)

# Whisper transcription
def transcribe_audio(audio_bytes):
    try:
        whisper_model = whisper.load_model("base")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        result = whisper_model.transcribe(path)
        os.remove(path)
        return result["text"]
    except Exception as e:
        st.error(f"Whisper error: {e}")
        return None

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

def save_chat(role, message):
    cursor.execute("INSERT INTO chat_history (role, message, timestamp) VALUES (?, ?, ?)",
                   (role, message, datetime.datetime.now().isoformat()))
    conn.commit()

def load_chat():
    cursor.execute("SELECT role, message FROM chat_history ORDER BY id ASC")
    return cursor.fetchall()

def clear_chat():
    cursor.execute("DELETE FROM chat_history")
    conn.commit()

def process_with_document(file, question):
    try:
        file_bytes = file.read()
        mime_type = file.type
        contents = [
            {"role": "user", "parts": [question]},
            {"role": "user", "parts": [{"mime_type": mime_type, "data": file_bytes}]},
        ]
        return model.generate_content(contents).text
    except Exception as e:
        return f"❌ Error: {e}"

def process_general_chat(prompt, chat_session):
    return chat_session.send_message(prompt).text

# ==== UI ====
st.markdown("<h1 style='text-align:center;'>🤖 Gemini All-in-One Chat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ccc;'>Ask questions, chat casually, or analyze documents — in one place.</p>", unsafe_allow_html=True)
st.divider()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": role.lower(), "parts": [msg]} for role, msg in load_chat()]
    st.session_state["chat"] = model.start_chat(history=st.session_state["chat_history"])

# === Chat Display ===
for msg in st.session_state["chat_history"]:
    role, content = msg["role"], msg["parts"][0]
    color = "#1976d2" if role == "user" else "#673ab7"
    align = "right" if role == "user" else "left"
    st.markdown(f"""
        <div style='background-color:{color};color:white;padding:10px 15px;
                    border-radius:12px;margin:10px 0;max-width:70%;
                    float:{align};clear:both;'>
            <strong>{'You' if role == 'user' else 'Gemini'}:</strong><br>{content}
        </div>
    """, unsafe_allow_html=True)

st.divider()

# === Input Section ===
st.markdown("### 💬 Unified Chat Input")

cols = st.columns([3, 1])
with cols[0]:
    user_input = st.text_input("Type your message or ask something:")

with cols[1]:
    audio = mic_recorder(
        start_prompt="🎙️ Tap to Speak",
        stop_prompt="⏹️ Stop Recording",
        key="mic",
        use_container_width=True
    )

if audio:
    if not user_input:
        st.audio(audio["bytes"])
        with st.spinner("Transcribing your voice..."):
            transcript = transcribe_audio(audio["bytes"])
            if transcript:
                user_input = transcript
                st.success(f"Transcribed: {transcript}")
                audio = None  # delete the voice data
    # else: ignore audio if user_input is already present
        

uploaded_file = st.file_uploader("📎 Optional: Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])
send = st.button("🚀 Send")

if send and user_input:
    with st.spinner("Gemini is thinking..."):
        if uploaded_file:
            reply = process_with_document(uploaded_file, user_input)
        else:
            reply = process_general_chat(user_input, st.session_state["chat"])

    # Save and update history
    save_chat("User", user_input)
    save_chat("Model", reply)
    st.session_state["chat_history"].append({"role": "user", "parts": [user_input]})
    st.session_state["chat_history"].append({"role": "model", "parts": [reply]})
    st.session_state["chat"].history.append({"role": "user", "parts": [user_input]})
    st.session_state["chat"].history.append({"role": "model", "parts": [reply]})

    st.rerun()

# === Sidebar: History ===
with st.sidebar:
    st.markdown("### 📜 Chat History")
    if st.button("🗑 Clear History"):
        clear_chat()
        st.session_state["chat_history"] = []
        st.session_state["chat"] = model.start_chat(history=[])
        st.success("History cleared.")
    search = st.text_input("🔍 Search:")
    history = load_chat()
    if search:
        history = [m for m in history if search.lower() in m[1].lower()]
    with st.expander("🧾 View All"):
        for r, m in history:
            st.markdown(f"**{r}**: {m}")