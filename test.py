import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from pymongo import MongoClient
import fitz  # PyMuPDF
import pdfplumber  # For table extraction
import pandas as pd
from PIL import Image
import io
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile

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

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I assist you today? You can upload a PDF or chat with me."}]
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "tables_summary" not in st.session_state:
        st.session_state.tables_summary = ""
    if "image_captions_str" not in st.session_state:
        st.session_state.image_captions_str = ""
    if "extracted_images" not in st.session_state:
        st.session_state.extracted_images = []
    if "just_chat" not in st.session_state:
        st.session_state.just_chat = False
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None

initialize_session_state()

# Save chat message to MongoDB
def save_chat(role, message):
    collection.insert_one({"role": role, "message": message})

# Load chat history from MongoDB
def load_chat():
    return list(collection.find({}, {"_id": 0}))

# Extract text from PDF
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    return text

# Extract tables
def extract_tables_from_pdf(file):
    tables = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    except Exception as e:
        st.error(f"Error extracting tables from PDF: {e}")
        return []
    return tables

# Extract images from PDF
def extract_images_from_pdf(file_bytes):
    images_data = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                for img in page.get_images(full=True):
                    xref = img[0]
                    base = doc.extract_image(xref)
                    image_bytes = base["image"]
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        images_data.append({"page": i + 1, "image": image})
                    except Exception as e:
                        print(f"Error opening image on page {i+1}: {e}")
    except Exception as e:
        print(f"Error opening PDF for image extraction: {e}")
        return []
    return images_data

# Transcribe audio using Whisper
def transcribe_audio_whisper(audio_bytes):
    try:
        model = whisper.load_model("base")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_file_name = tmp_file.name

        result = model.transcribe(temp_file_name)
        os.remove(temp_file_name)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio with Whisper: {e}")
        return None

# Gemini model setup
chat_history = [{"role": h["role"].lower(), "parts": [h["message"]]} for h in load_chat()]
flash_model = genai.GenerativeModel("gemini-1.5-flash-latest")
vision_model = genai.GenerativeModel("gemini-1.5-flash-latest")
chat = flash_model.start_chat(history=chat_history)

def get_gemini_response(context, question, images, just_chat=False):
    if just_chat:
        response = chat.send_message(question)
        return response.text
    else:
        prompt_parts = [
            f"""The following is extracted from a PDF document. Use it to answer the user's question.

            --- Start of PDF Text ---
            {context['text'][:4000]}...
            --- End of PDF Text ---

            --- Tables Extracted ---
            {context['tables_summary']}

            --- Image Information ---
            {context['image_captions']}

            User Question: {question}

            If the question is about an image, refer to the provided images. Indicate which page the image is from if relevant.
            """
        ]
        image_parts_pil = [img_data["image"] for img_data in images]
        contents = prompt_parts + image_parts_pil
        response = vision_model.generate_content(contents)
        return response.text

# Streamlit UI
st.set_page_config(page_title="PDF Assistant with Gemini Vision")
st.title("PDF Chat Assistant with Image Understanding")
st.caption("Upload a PDF and ask questions about its content, or just chat!")

# Chat mode selector
st.session_state.just_chat = st.radio("Chat Mode", ('PDF Chat', 'General Chat'), index=0) == 'General Chat'

# Show PDF uploader only in PDF mode
uploaded_file = None
if not st.session_state.just_chat:
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Microphone input
audio_data = mic_recorder(start_prompt="Click to Record", stop_prompt="Click to Stop", key="audio_recorder")
st.session_state.audio_data = audio_data

voice_question = None
if st.session_state.audio_data:
    st.audio(st.session_state.audio_data['bytes'])
    with st.spinner("Transcribing audio..."):
        voice_question = transcribe_audio_whisper(st.session_state.audio_data['bytes'])
    if voice_question:
        st.write(f"You asked: '{voice_question}'")

question = st.text_input("Ask a question about the uploaded PDF or just chat:", value=voice_question if voice_question else "")
submit = st.button("Ask")

if uploaded_file:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    # Extract text
    st.session_state.pdf_text = extract_text_from_pdf(file_bytes)

    # Extract tables
    uploaded_file.seek(0)
    tables = extract_tables_from_pdf(uploaded_file)
    st.session_state.tables_summary = ""
    if tables:
        for i, df in enumerate(tables):
            st.subheader(f"Table {i+1}")
            st.dataframe(df)
            st.session_state.tables_summary += f"Table {i+1}:\n{df.to_string(index=False)}\n\n"
    else:
        st.session_state.tables_summary = "No tables found."

    # Extract images
    uploaded_file.seek(0)
    st.session_state.extracted_images = extract_images_from_pdf(file_bytes)
    image_captions = []
    if st.session_state.extracted_images:
        for img_data in st.session_state.extracted_images:
            image_captions.append(f"Image found on Page {img_data['page']}")
    else:
        st.info("No images found in the PDF.")
    st.session_state.image_captions_str = "\n".join(image_captions)

# Handle Question
if question and submit:
    with st.spinner("Processing your request..."):
        if st.session_state.just_chat or not uploaded_file:
            answer = get_gemini_response("", question, [], just_chat=True)
        else:
            context = {
                "text": st.session_state.pdf_text,
                "tables_summary": st.session_state.tables_summary,
                "image_captions": st.session_state.image_captions_str
            }
            answer = get_gemini_response(context, question, st.session_state.extracted_images)

        save_chat("User", question)
        save_chat("Model", answer)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.session_state.audio_data = None

        st.subheader("Gemini Response")
        st.write(answer)

st.subheader("Chat History")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Close MongoDB connection
client.close()
