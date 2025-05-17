import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from pymongo import MongoClient

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

# Load history for Gemini model
gemini_history = [
    {"role": h["role"].lower(), "parts": [h["message"]]}
    for h in load_chat()
]

# Create chat session with previous history
model = genai.GenerativeModel("gemini-1.5-flash-latest")
chat = model.start_chat(history=gemini_history)

# Function to get response
def get_gemini_response(question):
    response = chat.send_message(question)
    return response.text

# Streamlit UI
st.set_page_config(page_title="Q&A with Gemini")
st.title("💬 Gemini Flash Chatbot")
st.caption("Using Gemini 1.5 Flash with MongoDB for chat history")

input_text = st.text_input("Your question:", key="input")
submit = st.button("Ask")

if submit and input_text:
    save_chat("User", input_text)
    response = get_gemini_response(input_text)
    save_chat("Model", response)

    st.subheader("Response")
    st.write(response)

st.subheader("Chat History")

for msg in load_chat():
    st.markdown(f"**{msg['role']}**: {msg['message']}")
