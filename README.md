📚 PDF Chatbot with Gemini API
A lightweight chatbot built with Streamlit that lets users ask questions based on uploaded PDF content using Google Gemini Pro (free tier).

🚀 Features
Upload PDF files
Extract text using pdfplumber
Ask questions in natural language
Powered by Gemini Pro API (free version)
Maintains chat history using Streamlit session state

🛠 Tech Stack
🧠 LLM: Gemini Pro via google-generativeai
🖥 Frontend: Streamlit
📄 PDF Parsing: pdfplumber
🐍 Language: Python

🔧 Setup Instructions
1.Clone the repository
git clone https://github.com/your-username/Chatbot.git
cd Chatbot

2.Create virtual environment
python -m venv venv

source venv/bin/activate  
# Windows: 
GitBash: source venv/Scripts/activate
Command Prompt: venv\Scripts\activate
PowerShell: .\venv\Scripts\Activate.ps1

3.Install dependencies
pip install -r requirements.txt

4.Add your Gemini API Key
Create a .env file in the project root:
GEMINI_API_KEY=your_api_key_here

5.Run the app
streamlit run app.py
