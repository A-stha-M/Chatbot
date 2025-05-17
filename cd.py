from dotenv import load_dotenv
load_dotenv()  # Load all the environment variables

import streamlit as st
import os
import google.generativeai as genai

# Set up Google Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Streamlit UI
st.title("Clora: Your Friendly AI Explainer")

user_input = st.text_input("Ask a simple question:", "What is an apple?")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        response = model.generate_content(
            f"""
            Answer the following question in a very understandable way. Keep the answer simple and concise. Respond in only two paragraphs.
            Question: {user_input}
            """,
        )
        st.markdown("---")
        st.subheader("Clora says:")
        st.write(response.text)