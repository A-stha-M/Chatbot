import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from pymongo import MongoClient, ASCENDING, DESCENDING
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
import io
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import hashlib
import subprocess # For ffmpeg check
import datetime # For MongoDB timestamp

# ---------------------- CONFIG ------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GOOGLE_API_KEY or not MONGO_URI:
    st.error("Set GOOGLE_API_KEY and MONGO_URI in .env")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# MongoDB setup
try:
    client = MongoClient(MONGO_URI)
    db = client["chatbot_db"]
    general_chat_collection = db["general_chat_history"]
    pdf_chat_collection = db["pdf_chat_history"]
    print("MongoDB connection successful!")
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}. Please check your MONGO_URI.")
    print(f"MongoDB connection error: {e}")
    st.stop()

# OPTIONAL: Add ffmpeg to PATH if not already (for Windows/macOS where it's not global)
ffmpeg_path = os.getenv("FFMPEG_PATH", None) # Get from .env if set
if ffmpeg_path and ffmpeg_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + ffmpeg_path
    print(f"Added FFmpeg path to PATH: {ffmpeg_path}")

# Verify ffmpeg
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    print("FFmpeg detected and working.")
except (FileNotFoundError, subprocess.CalledProcessError) as e:
    st.warning(f"FFmpeg not detected or failed to run: {e}. Voice transcription may not work.")
    print(f"FFmpeg check failed: {e}")

@st.cache_resource
def load_whisper_model():
    """Loads the Whisper base model. Cached for efficiency."""
    try:
        print("Attempting to load Whisper model 'base'...")
        model = whisper.load_model("base")
        print("Whisper model 'base' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}. Ensure 'ffmpeg' is installed and your environment is set up correctly.")
        print(f"Whisper model load error: {e}")
        st.stop()

# ---------------------- CSS ---------------------------
st.set_page_config(page_title="Gemini Chatbot", layout="centered")
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

# -------------------- SESSION INIT --------------------
def initialize_session_state():
    """Initializes Streamlit session state variables with default values."""
    defaults = {
        "general_messages": [{"role": "assistant", "content": "Hi! How can I help you today?"}],
        "pdf_messages": [],
        "pdf_text": "",
        "tables_summary": "",
        "image_captions_str": "",
        "extracted_images_for_gemini": [],
        "extracted_images_info": [],
        "current_question": "", # For the text input widget's value (typed or pre-filled by voice)
        "show_chat_history": False,
        "active_chat_mode": "general",
        "uploaded_pdf_content": None,
        "uploaded_pdf_hash": None,
        "should_process_question": False, # ONLY process when this is True (triggered by Send button)
        "last_mic_audio_bytes": None # To prevent re-transcribing same audio
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    print("Session state initialized.")

initialize_session_state()

# ------------------ PDF PROCESSING ---------------------
def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Text Extraction Error: {e}")
        print(f"Text Extraction Error: {e}")
        return ""

def extract_tables_from_pdf(file_bytes):
    """Extracts tables from a PDF file using pdfplumber."""
    tables = [] # Initialize tables as an empty list
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if table: # Ensure the extracted table is not empty
                        # Check if table has headers before creating DataFrame
                        if len(table) > 1 and table[0]: # If there's more than one row and the first row is not empty
                            df = pd.DataFrame(table[1:], columns=table[0])
                        else: # If no explicit header row, treat all as data
                            df = pd.DataFrame(table)
                        tables.append(df)
    except Exception as e:
        st.error(f"Table Extraction Error: {e}. This might happen if tables are images or have complex layouts.")
        print(f"Table Extraction Error: {e}")
        return [] # Ensure it returns an empty list on error
    return tables # Ensure it always returns the list, which might be empty

def extract_images_from_pdf(file_bytes):
    """
    Extracts images from a PDF.
    Returns a list of PIL Images for Gemini and a list of info for captions.
    """
    images_for_gemini = []
    images_info = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                for img_data in page.get_images(full=True):
                    xref = img_data[0]
                    base = doc.extract_image(xref)
                    if base and "image" in base:
                        try:
                            image = Image.open(io.BytesIO(base["image"]))
                            images_for_gemini.append(image)
                            images_info.append({"page": i + 1, "status": "processed"})
                        except Exception as img_e:
                            print(f"Warning: Could not open image on page {i+1}, xref {xref}: {img_e}")
                            images_info.append({"page": i + 1, "status": "failed", "error": str(img_e)})
    except Exception as e:
        st.error(f"Overall Image Extraction Error: {e}")
        print(f"Overall Image Extraction Error: {e}")
    return images_for_gemini, images_info

# --------------------- CHAT HISTORY ---------------------
def save_chat_to_db(collection, role, message):
    """Saves a chat message to the specified MongoDB collection."""
    try:
        collection.insert_one({"role": role, "message": message, "timestamp": datetime.datetime.now()})
        print(f"Saved message to DB: {role}: {message[:50]}...")
    except Exception as e:
        st.error(f"Failed to save chat to DB: {e}")
        print(f"Failed to save chat to DB: {e}")

def load_chat_from_db(collection):
    """Loads chat history from the specified MongoDB collection."""
    try:
        history = list(collection.find({}, {"_id": 0}).sort("timestamp", ASCENDING))
        print(f"Loaded {len(history)} messages from DB.")
        return history
    except Exception as e:
        st.error(f"Failed to load chat history from DB: {e}")
        print(f"Failed to load chat history from DB: {e}")
        return []

# ----------------- GEMINI INFERENCE ---------------------
# These are loaded at the beginning of each rerun, outside the chat functions
# This is necessary because Streamlit reruns the whole script.
# We need to ensure the chat history is up-to-date for the model's session.
general_chat_history_for_model = [{"role": h["role"].lower(), "parts": [h["message"]]} for h in load_chat_from_db(general_chat_collection)]
pdf_chat_history_for_model = [{"role": h["role"].lower(), "parts": [h["message"]]} for h in load_chat_from_db(pdf_chat_collection)]

general_model = genai.GenerativeModel("gemini-1.5-flash-latest")
pdf_vision_model = genai.GenerativeModel("gemini-1.5-flash-latest")

general_chat_session = general_model.start_chat(history=general_chat_history_for_model)
pdf_chat_session = pdf_vision_model.start_chat(history=pdf_chat_history_for_model)


def get_gemini_general_response(question):
    """Sends a general question to the Gemini model."""
    print(f"Sending general question to Gemini: {question[:50]}...")
    try:
        response = general_chat_session.send_message(question)
        print(f"Received general response from Gemini: {response.text[:50]}...")
        return response.text
    except Exception as e:
        st.error(f"Gemini General Chat Error: {e}. Please try again.")
        print(f"Gemini General Chat Error: {e}")
        return "An error occurred during general chat."

def get_gemini_pdf_response(context, question, images):
    """Sends a PDF-related question and context to the Gemini Vision model."""
    print(f"Sending PDF question to Gemini: {question[:50]}...")
    prompt_parts = []
    prompt_parts.append("You are an AI assistant specialized in analyzing PDF documents.")
    prompt_parts.append("Your primary goal is to answer the user's question **ONLY** using the provided PDF content.")
    prompt_parts.append("Do not use outside knowledge or make assumptions. If the answer or relevant information is not explicitly present within the provided PDF content, you **MUST** clearly state: 'I cannot find the answer to that question in the provided document.'")

    if images:
        prompt_parts.append("If the user's question relates to visual content, or if it helps in understanding the document, please analyze and describe the provided images. Mention the page number if known from the 'IMAGE INFORMATION' section.")
    else:
        prompt_parts.append("No images were extracted from this document, so I cannot answer questions requiring visual analysis.")

    prompt_parts.append("\n--- START OF PROVIDED PDF CONTENT ---")

    if context['text']:
        text_preview = context['text'][:10000]
        if len(context['text']) > 10000:
            text_preview += "\n\n... (Document text truncated for processing. More content may exist.)"
        prompt_parts.append(f"\n### DOCUMENT TEXT EXCERPT:\n{text_preview}")
    else:
        prompt_parts.append("\n### DOCUMENT TEXT EXCERPT: (No significant text extracted from the PDF.)")

    if context['tables_summary']:
        prompt_parts.append(f"\n### EXTRACTED TABLES:\n{context['tables_summary']}")
    else:
        prompt_parts.append("\n### EXTRACTED TABLES: (No tables extracted from the PDF.)")

    if context['image_captions']:
        prompt_parts.append(f"\n### IMAGE INFORMATION FROM PDF (presence and page):\n{context['image_captions']}")
    else:
        prompt_parts.append("\n### IMAGE INFORMATION: (No images or image information extracted from the PDF.)")

    prompt_parts.append("\n--- END OF PROVIDED PDF CONTENT ---")
    prompt_parts.append(f"\nBased on the above PDF content, answer the following question:")
    prompt_parts.append(f"User Question: {question}")
    prompt_text = "\n".join(prompt_parts)

    try:
        contents = [prompt_text]
        if images:
            contents.extend(images)
        response = pdf_chat_session.send_message(contents)
        print(f"Received PDF response from Gemini: {response.text[:50]}...")
        return response.text
    except Exception as e:
        st.error(f"Gemini PDF Chat Error: {e}. This could be due to context window limits, model issues, or a very large/complex request. Please try a simpler question or a smaller PDF.")
        print(f"Gemini PDF Chat Error: {e}")
        return "I encountered an error while processing the PDF-related question. The model might have been overloaded, or the question was too complex given the document's content. Please try rephrasing or a different question, or try a smaller PDF."

# --------------------- AUDIO ----------------------------
def transcribe_audio(audio_bytes):
    """Transcribes audio bytes to text using Whisper."""
    print("Starting audio transcription...")
    try:
        model = load_whisper_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
        print(f"Temporary audio file saved to: {temp_path}")
        result = model.transcribe(temp_path)
        os.remove(temp_path)
        print(f"Audio transcription complete: '{result['text'][:50]}...'")
        return result["text"]
    except Exception as e:
        st.error(f"Whisper Transcription Error: {e}. Please ensure 'ffmpeg' is installed and accessible in your PATH.")
        print(f"Whisper Transcription Error: {e}")
        return None

# ----------------------- UI -----------------------------
st.title("🤖 Gemini Chatbot")

# --- PDF Uploader and Processing Logic ---
newly_uploaded_file = st.file_uploader(
    "📄 Upload PDF (will automatically switch to PDF chat)",
    type=["pdf"],
    key="pdf_uploader"
)

if newly_uploaded_file:
    new_file_bytes = newly_uploaded_file.read()
    new_file_hash = hashlib.md5(new_file_bytes).hexdigest()

    if (new_file_hash != st.session_state.uploaded_pdf_hash) or (st.session_state.uploaded_pdf_content is None):
        st.session_state.uploaded_pdf_content = new_file_bytes
        st.session_state.uploaded_pdf_hash = new_file_hash

        st.session_state.pdf_messages = []
        st.session_state.current_question = ""
        print("New PDF uploaded, clearing PDF chat state.")

        with st.spinner("Processing PDF... This may take a moment."):
            st.session_state.pdf_text = extract_text_from_pdf(st.session_state.uploaded_pdf_content)
            
            # --- START OF FIX FOR TYPEDERROR ---
            extracted_tables = extract_tables_from_pdf(st.session_state.uploaded_pdf_content)
            
            if extracted_tables is not None and isinstance(extracted_tables, list):
                st.session_state.tables_summary = "\n".join(
                    [f"Table {i+1}:\n{df.to_string(index=False)}\n" for i, df in enumerate(extracted_tables)]
                )
            else:
                st.session_state.tables_summary = "" # Ensure it's an empty string if no tables or error
                print("Warning: extract_tables_from_pdf did not return a list or returned None.")
            # --- END OF FIX FOR TYPEDERROR ---

            st.session_state.extracted_images_for_gemini, st.session_state.extracted_images_info = extract_images_from_pdf(st.session_state.uploaded_pdf_content)

            captions = []
            for img_info in st.session_state.extracted_images_info:
                caption_text = f"Image found on Page {img_info['page']}"
                if img_info.get("status") == "failed":
                    caption_text += f" (Could not process: {img_info.get('error', 'Unknown error')})"
                captions.append(caption_text)
            st.session_state.image_captions_str = "\n".join(captions)
        st.success("PDF processed successfully! You can now ask questions about its content.")
        st.rerun() # Rerun to update chat mode
    else:
        st.info("Same PDF already loaded.")
        print("Same PDF loaded again.")

elif newly_uploaded_file is None and st.session_state.uploaded_pdf_content is not None:
    st.info("PDF cleared. Reverting to general chat mode.")
    st.session_state.uploaded_pdf_content = None
    st.session_state.uploaded_pdf_hash = None
    st.session_state.pdf_text = ""
    st.session_state.tables_summary = ""
    st.session_state.image_captions_str = ""
    st.session_state.extracted_images_for_gemini = []
    st.session_state.extracted_images_info = []
    st.session_state.pdf_messages = []
    print("PDF explicitly cleared by user.")
    st.rerun() # Rerun to update chat mode

if st.session_state.uploaded_pdf_content:
    st.session_state.active_chat_mode = "pdf"
    st.markdown("---")
    st.markdown(f"**Current Mode: PDF Chat** (Answering questions about the uploaded PDF)")
else:
    st.session_state.active_chat_mode = "general"
    st.markdown("---")
    st.markdown(f"**Current Mode: General Chat** (Answering general questions)")


# --- Chat Input (Text and Mic) ---
col1, col2 = st.columns([5, 1])
with col1:
    question_label = "Ask a question about the PDF:" if st.session_state.active_chat_mode == "pdf" else "Ask me anything:"
    
    # Text input for user's question. This widget's value will be what's in session state.
    # We use a key, and then we manually update st.session_state.current_question from it
    # to maintain consistency and allow for editing.
    input_text_widget_value = st.text_input(
        question_label,
        value=st.session_state.current_question,
        key="text_input_main" # Unique key for this widget
    )
    
    # IMPORTANT: Update st.session_state.current_question if the user types into the widget
    # This ensures that any manual edits are reflected in the session state.
    if input_text_widget_value != st.session_state.current_question:
        st.session_state.current_question = input_text_widget_value
        print(f"User typed/edited input. current_question updated to: {st.session_state.current_question[:50]}")


with col2:
    with st.container():
        st.markdown('<div class="mic-container">', unsafe_allow_html=True)
        # Unique key for mic_recorder to avoid conflicts
        audio_data = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key="main_mic_recorder")
        st.markdown('</div>', unsafe_allow_html=True)

# Process audio data if captured and not already processed
if audio_data and audio_data['bytes'] and audio_data['bytes'] != st.session_state.last_mic_audio_bytes:
    print("New audio data captured and bytes available. Starting transcription flow.")
    st.session_state.last_mic_audio_bytes = audio_data['bytes'] # Store the audio bytes to avoid re-processing

    st.audio(audio_data['bytes'], format="audio/wav") # Display the recorded audio
    st.info("Audio recorded. Transcribing...")
    with st.spinner("Transcribing..."):
        transcribed_text = transcribe_audio(audio_data['bytes'])
    
    if transcribed_text:
        st.success(f"You said: '{transcribed_text}'")
        st.session_state.current_question = transcribed_text # Set the session state. This will pre-fill the text input on next rerun.
        print(f"Voice transcription complete. Set current_question to: {st.session_state.current_question[:50]}.")
        # No st.rerun() here, and no setting should_process_question=True.
        # Streamlit will naturally rerun to update the text input due to state change or mic interaction.
    else:
        st.warning("Audio transcription failed or returned empty.")
        print("Voice transcription failed or returned empty string.")
elif audio_data is not None and audio_data['bytes'] == st.session_state.last_mic_audio_bytes:
    print("Mic recorder returned same audio bytes. Avoiding re-transcription.")
elif audio_data is not None: # Means mic_recorder was interacted with, but no bytes (e.g., just stopped without recording)
    print("Mic recorder was interacted with, but no new audio bytes.")


# --- Send Message Button ---
send_button = st.button("Send Message", key="send_message_button")

# Main logic for processing the question (triggered by send_button)
if send_button and st.session_state.current_question:
    st.session_state.should_process_question = True # Set flag to trigger processing
    print(f"Send button clicked and current_question is present. Setting should_process_question=True. Rerunning.")
    st.rerun() # Rerun to enter the processing block below
elif send_button and not st.session_state.current_question:
    st.warning("Please enter a question or speak into the microphone.")


# This block only executes if the flag is set (by clicking send button)
if st.session_state.should_process_question:
    user_message = st.session_state.current_question
    
    # IMPORTANT: Reset flag *before* the API call to prevent re-execution
    st.session_state.should_process_question = False 
    st.session_state.current_question = "" # Clear input field immediately
    st.session_state.last_mic_audio_bytes = None # Clear last audio to allow new recordings
    
    print(f"DEBUG: Processing question: '{user_message[:50]}'...")

    # Add user's message to immediate chat display (optional, but good UX)
    if st.session_state.active_chat_mode == "pdf":
        st.session_state.pdf_messages.append({"role": "user", "content": user_message})
    else:
        st.session_state.general_messages.append({"role": "user", "content": user_message})

    # Now, call Gemini API
    if st.session_state.active_chat_mode == "pdf":
        print("Calling get_gemini_pdf_response...")
        with st.spinner("Analyzing PDF and generating response..."):
            ctx = {
                "text": st.session_state.pdf_text,
                "tables_summary": st.session_state.tables_summary,
                "image_captions": st.session_state.image_captions_str
            }
            answer = get_gemini_pdf_response(ctx, user_message, st.session_state.extracted_images_for_gemini)
        
        save_chat_to_db(pdf_chat_collection, "user", user_message) 
        save_chat_to_db(pdf_chat_collection, "assistant", answer)
        st.session_state.pdf_messages.append({"role": "assistant", "content": answer})
    else: # General chat mode
        print("Calling get_gemini_general_response...")
        with st.spinner("Thinking and generating response..."):
            answer = get_gemini_general_response(user_message)
        
        save_chat_to_db(general_chat_collection, "user", user_message)
        save_chat_to_db(general_chat_collection, "assistant", answer)
        st.session_state.general_messages.append({"role": "assistant", "content": answer})

    print("Gemini response received. Rerunning to display assistant response and clear input.")
    st.rerun() # Rerun one last time to display the new assistant message


# Chat display
st.subheader("💬 Current Conversation")

if st.session_state.active_chat_mode == "pdf":
    current_messages = st.session_state.pdf_messages
else:
    current_messages = st.session_state.general_messages

# Display messages (newest at bottom)
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat history display
st.session_state.show_chat_history = st.checkbox("📚 Show Chat History")
if st.session_state.show_chat_history:
    st.subheader("🕘 Full Chat History")
    st.markdown("---")
    st.write("### General Chat History")
    general_history = load_chat_from_db(general_chat_collection)
    if general_history:
        for h in general_history: # Display in chronological order for history
            with st.chat_message(h["role"]):
                st.markdown(h["message"])
    else:
        st.info("No general chat history yet.")

    st.markdown("---")
    st.write("### PDF Chat History")
    pdf_history = load_chat_from_db(pdf_chat_collection)
    if pdf_history:
        for h in pdf_history: # Display in chronological order for history
            with st.chat_message(h["role"]):
                st.markdown(h["message"])
    else:
        st.info("No PDF chat history yet.")

client.close()