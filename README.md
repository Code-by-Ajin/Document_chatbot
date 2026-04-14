# Document Intelligence - RAG Chatbot

An intelligent document analysis system that allows you to upload PDF documents and have natural language conversations with them. Utilizing Retrieval-Augmented Generation (RAG), this application provides accurate answers grounded in your specific data.

## 🚀 Introduction

This project is a full-stack **Document Chatbot** designed to analyze complex PDF files and answer user queries with high precision. Unlike standard LLMs, this system uses **RAG (Retrieval-Augmented Generation)** to "read" your documents and provide answers based *only* on the provided context, minimizing hallucinations and ensuring data-specific accuracy.

### Key Features:
- **PDF Ingestion**: Seamlessly upload and process multi-page PDF documents.
- **Smart Retrieval**: Uses high-performance vector search (FAISS) to find relevant document sections.
- **Advanced LLM**: Powered by **Llama 3.1** via Groq API for lightning-fast, high-quality responses.
- **Local Embeddings**: Uses `all-MiniLM-L6-v2` for efficient, private text vectorization.
- **Chat History**: Persistent conversation storage using MongoDB.
- **Modern UI**: Clean, responsive interface built with vanilla HTML/CSS and micro-animations.

---

## 🛠️ Working Procedure

1.  **Document Upload**: The user uploads a PDF through the web interface.
2.  **Processing & Chunking**:
    -   The backend uses `PyPDFLoader` to extract text.
    -   Text is split into optimal chunks (1500 characters with 200 overlap) to preserve context.
3.  **Vectorization**:
    -   Each chunk is converted into a mathematical representation (embedding) using HuggingFace.
    -   These embeddings are stored in a **FAISS vector database**.
4.  **Retrieval & Generation**:
    -   When a question is asked, the system searches the vector database for the top-N most relevant chunks.
    -   A specialized prompt containing these chunks and the user question is sent to the **Groq Llama-3.1-8b** model.
5.  **Response**: The model generates a structured Markdown response, which is displayed in the chat and saved to MongoDB.

---

## 📦 Needed Download Files & Prerequisites

To run this project, you need to have **Python 3.8+** installed.

### Required Dependencies
All necessary libraries are listed in `requirements.txt`. Key packages include:
- `fastapi` & `uvicorn` (Web Framework)
- `langchain` (RAG Orchestration)
- `langchain-groq` (LLM Integration)
- `faiss-cpu` (Vector Database)
- `pymongo` (History Storage)
- `pypdf` (PDF Parsing)

### Configuration
You will need a `.env` file (see `.env.template`) with the following keys:
- `GROQ_API_KEY`: Get one for free at [Groq Console](https://console.groq.com/).
- `MONGODB_URI`: Connectivity string for your MongoDB database (local or Atlas).

---

## 🏃 Procedure to Run

Follow these steps to get the application running locally:

### 1. Clone & Setup
```bash
# Navigate to the project directory
cd document_chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your credentials:
```env
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=your_mongodb_connection_string
```

### 4. Launch the Application
Start the FastAPI server:
```bash
python main.py
```
*Alternatively, use uvicorn directly for hot-reloads:*
```bash
uvicorn main:app --reload
```

### 5. Access the Chatbot
Open your browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

---

## 📁 Project Structure
- `main.py`: FastAPI server and API endpoints (`/upload`, `/ask`, `/history`).
- `rag_engine.py`: Core RAG logic, document processing, and LLM orchestration.
- `static/`: Frontend files (HTML/CSS/JS).
- `uploads/`: Temporary storage for uploaded PDF documents.
- `requirements.txt`: Python package dependencies.
