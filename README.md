# 📈 Stock Trend Analyzer (News Research Tool)

A Streamlit app that analyzes stock research articles from **Moneycontrol** (and other sites) using **LangChain**, **OpenAI embeddings**, and **FAISS vector store**.  
The tool allows you to load news article URLs, split the text, create embeddings, and query insights with source references.  

---

## 🚀 Features
- Fetches news articles directly from URLs  
- Splits long articles into manageable chunks  
- Creates **vector embeddings** using OpenAI  
- Stores and retrieves embeddings using **FAISS**  
- Lets you ask **questions about the articles** (Q&A with sources)  
- Easy-to-use **Streamlit UI**  

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/stock-trend-analyzer.git
   cd stock-trend-analyzer
pip install -r requirements.txt
Setup OpenAI API Key

Go to OpenAI API Keys

Generate a new key.

Create a .env file in your project folder and add your key:

OPENAI_API_KEY=your_api_key_here


streamlit run app.py
