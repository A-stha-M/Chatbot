import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import sqlite3
import fitz  # PyMuPDF for PDF reading
import datetime

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check API Key
if not GOOGLE_API_KEY:
    st.error("API key not found. Please set GOOGLE_API_KEY in .env")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# SQLite DB setup
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

# Helper functions
def save_chat(role, message):
    cursor.execute(
        "INSERT INTO chat_history (role, message, timestamp) VALUES (?, ?, ?)",
        (role, message, datetime.datetime.now().isoformat())
    )
    conn.commit()

def load_chat():
    cursor.execute("SELECT role, message FROM chat_history ORDER BY id ASC")
    return cursor.fetchall()

def clear_chat():
    cursor.execute("DELETE FROM chat_history")
    conn.commit()

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Load previous history into Gemini format
chat_history = [{"role": role.lower(), "parts": [msg]} for role, msg in load_chat()]
model = genai.GenerativeModel("gemini-1.5-flash-latest")
chat = model.start_chat(history=chat_history)

# Get response from Gemini
def get_gemini_response(prompt):
    response = chat.send_message(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="PDF + Chatbot with Gemini Flash 2.0", layout="centered")
st.title("📄 Gemini Flash Q&A Chatbot")
st.caption("Upload a PDF or just ask questions. History is stored in SQLite.")

# Tabbed layout
tab1, tab2 = st.tabs(["📕 PDF Q&A", "💬 General Chat"])

# --- PDF Q&A TAB ---
with tab1:
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    question = st.text_input("Ask something about the document:")
    submit = st.button("Ask", key="pdf_ask")

    if uploaded_file and submit and question:
        with st.spinner("Reading and analyzing PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)

        prompt = f"PDF Content:\n{pdf_text[:8000]}\n\nQuestion: {question}"

        with st.spinner("Generating response..."):
            response = get_gemini_response(prompt)

        save_chat("User", question)
        save_chat("Model", response)

        st.subheader("🤖 Gemini's Answer")
        st.write(response)

# --- GENERAL CHAT TAB ---
with tab2:
    st.subheader("💬 Chat with Gemini")

    # Right column layout with chat on top, input at bottom
    chat_placeholder = st.container()
    input_placeholder = st.container()

    with chat_placeholder:
        st.markdown("### 🧾 Conversation")
        general_history = load_chat()
        for role, msg in general_history:
            if role.lower() == "user":
                st.markdown(f"<div style='background-color:#e0f7fa;padding:10px;border-radius:10px;margin-bottom:5px'><strong>You:</strong><br>{msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color:#f3e5f5;padding:10px;border-radius:10px;margin-bottom:15px'><strong>Gemini:</strong><br>{msg}</div>", unsafe_allow_html=True)

    with input_placeholder:
        chat_input = st.text_input("Type your message:", key="chat_input")
        send_chat = st.button("Send", key="chat_send")

        if send_chat and chat_input:
            with st.spinner("Thinking..."):
                reply = get_gemini_response(chat_input)

            save_chat("User", chat_input)
            save_chat("Model", reply)
            st.rerun()

# --- CHAT HISTORY ---
st.subheader("📜 Chat History")
search_query = st.text_input("Search messages:")

if st.button("🗑️ Clear Chat History"):
    clear_chat()
    st.success("Chat history cleared.")

history = load_chat()
if search_query:
    history = [msg for msg in history if search_query.lower() in msg[1].lower()]

for role, msg in history:
    st.markdown(f"**{role}**: {msg}")
