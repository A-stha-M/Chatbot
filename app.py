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

# Extract text from PDF
def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract tables
def extract_tables_from_pdf(file):
    tables = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
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
    return images_data

# Gemini model setup
chat_history = [{"role": h["role"].lower(), "parts": [h["message"]]} for h in load_chat()]
flash_model = genai.GenerativeModel("gemini-1.5-flash-latest")
vision_model = genai.GenerativeModel("gemini-1.5-flash-latest")
chat = flash_model.start_chat(history=chat_history)

def get_gemini_response(context, question, images):
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
st.caption("Upload a PDF and ask questions about its content including text, tables, and images.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question about the uploaded PDF:")
submit = st.button("Ask")

if uploaded_file:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    # Extract text
    pdf_text = extract_text_from_pdf(file_bytes)

    # Extract tables
    uploaded_file.seek(0)
    tables = extract_tables_from_pdf(uploaded_file)
    tables_summary = ""
    if tables:
        for i, df in enumerate(tables):
            st.subheader(f"Table {i+1}")
            st.dataframe(df)
            tables_summary += f"Table {i+1}:\n{df.to_string(index=False)}\n\n"
    else:
        tables_summary = "No tables found."

    # Extract images
    uploaded_file.seek(0)
    extracted_images = extract_images_from_pdf(file_bytes)
    image_captions = []
    if extracted_images:
        for img_data in extracted_images:
            image_captions.append(f"Image found on Page {img_data['page']}")
    else:
        st.info("No images found in the PDF.")
    image_captions_str = "\n".join(image_captions)

    # Handle Question
    if question and submit:
        with st.spinner("Reading PDF and querying Gemini Vision..."):
            context = {
                "text": pdf_text,
                "tables_summary": tables_summary,
                "image_captions": image_captions_str
            }

            answer = get_gemini_response(context, question, extracted_images)

            # Save chat
            save_chat("User", question)
            save_chat("Model", answer)

        st.subheader("Gemini Response")
        st.write(answer)

st.subheader("Chat History")
for msg in load_chat():
    st.markdown(f"**{msg['role']}**: {msg['message']}")

# Close MongoDB connection
client.close()