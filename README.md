# 💬 CustomRAG – Gemini + LangChain + Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that uses **Google Gemini** for LLM responses, **LangChain** for orchestration, **ChromaDB** for persistent vector storage, and **Streamlit** for the UI.

## 🚀 Features
- Multi-file document ingestion (PDF, TXT)
- Persistent Chroma vector store
- Conversational memory using `ConversationSummaryBufferMemory`
- Live chat UI built with Streamlit
- Uses **Google Generative AI (Gemini)** as backend LLM

## 🧩 Setup

```bash
# Clone repo
git clone https://github.com/<your-username>/CustomRAG.git
cd CustomRAG

# Create virtual env and install deps
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
GOOGLE_API_KEY=your_api_key_here

# Run app
streamlit run app.py