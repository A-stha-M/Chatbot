import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from pymongo import MongoClient
import fitz  # PyMuPDF for PDF reading

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GOOGLE_API_KEY:
    st.error("API key not found. Please set GOOGLE_API_KEY in .env")
    st.stop()

if not MONGO_URI:
    st.error("Mongo URI not found. Please set MONGO_URI in .env")
    st.stop()

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
collection = db["chat_history"]

# Save chat message to MongoDB
def save_chat(role, message):
    collection.insert_one({"role": role, "message": message})

# Load chat history from MongoDB
def load_chat():
    return list(collection.find({}, {"_id": 0}))

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Load previous chat history
chat_history = [
    {"role": h["role"].lower(), "parts": [h["message"]]} for h in load_chat()
]

# Initialize Gemini Flash model with previous chat history
model = genai.GenerativeModel("gemini-1.5-flash-latest")
chat = model.start_chat(history=chat_history)

# Get response from Gemini Flash model
def get_gemini_response(prompt):
    response = chat.send_message(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="Q&A with Gemini")
st.title("Chatbot")
st.caption("Upload a PDF and chat with it using Gemini Flash. History is saved in MongoDB.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question:")
submit = st.button("Ask")

if uploaded_file and submit and question:
    with st.spinner("Reading PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Thinking..."):
        prompt = f"PDF Content:\n{pdf_text[:8000]}\n\nQuestion: {question}"
        response = get_gemini_response(prompt)

    # Save to MongoDB
    save_chat("User", question)
    save_chat("Model", response)

    st.subheader("Response")
    st.write(response)

st.subheader("Chat History")
for msg in load_chat():
    st.markdown(f"**{msg['role']}**: {msg['message']}")
