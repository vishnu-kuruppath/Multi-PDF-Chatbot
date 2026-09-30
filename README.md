## Multi-PDF-Chatbot (RAG + Ollama)

# Overview

This project is a Retrieval-Augmented Generation (RAG) based PDF Chatbot that allows users to upload multiple PDF documents and ask questions from them.

The chatbot retrieves relevant information from the documents and generates accurate answers using a local LLM powered by Ollama.

# Technologies Used
Python
Streamlit
LangChain
ChromaDB (Vector Database)
Ollama (LLM + Embeddings)

# Features
Upload multiple PDF files
Extract and process document content
Convert text into embeddings
Store embeddings in vector database
Retrieve relevant context using semantic search
Generate answers using LLM
Chat history tracking

# How It Works
Upload PDFs
Documents are split into chunks
Each chunk is converted into embeddings
Stored in ChromaDB
User asks a question
Relevant chunks are retrieved
LLM generates final answer

# How to Run
Step 1: Clone Repo
git clone https://github.com/your-username/pdf-chatbot-rag.git
cd pdf-chatbot-rag
Step 2: Install Requirements
pip install -r requirements.txt
Step 3: Run App
streamlit run app.py

# Note
Make sure you have Ollama installed and running:
ollama run gemma3:1b
ollama pull nomic-embed-text


🌟 Future Improvements
Add authentication
Support more file types (DOCX, TXT)
Deploy on cloud
Improve UI/UX
