# Local RAG Pipeline

Ask questions over your own PDF documents using a completely local AI model.
No API costs, no internet required, no data leaves your machine.

## How It Works
```
Your PDFs
    ↓
Text extracted and chunked
    ↓
Stored in local ChromaDB vector database
    ↓
Your question searches for relevant chunks
    ↓
LLaMA answers using your documents as context
    ↓
All running locally on your machine
```

## Requirements
- Python 3.10+
- Ollama (ollama.com)

## Setup

### 1. Install Ollama and download the model
Download from ollama.com then run:
ollama pull llama3.1

### 2. Install Python dependencies
pip install -r requirements.txt

### 3. Add your PDF files
Drop your PDF files into the docs/ folder
(created automatically on first run)

### 4. Run the pipeline
python rag_pipeline.py

## Usage
```
You: What is heteroscedasticity?
LLaMA: Based on your notes, heteroscedasticity refers to...

You: What are the assumptions of linear regression?
LLaMA: According to your documents...

Type 'quit' to exit
```

## Use Cases
- Study assistant over lecture notes and textbooks
- Research assistant over academic papers
- Personal knowledge base over your own documents
- Document Q&A for any PDF collection

## How to Add New Documents
1. Add new PDFs to the docs/ folder
2. Delete the db/ folder
3. Run the script again — it will rebuild the knowledge base

## Tech Stack
- LLaMA 3.1 8B via Ollama — local AI inference
- ChromaDB — local vector database
- PyMuPDF — PDF text extraction
- Python — orchestration