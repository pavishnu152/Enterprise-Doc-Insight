<<<<<<< HEAD
# Enterprise-Doc-Insight
Enterprise Document Insight Engine – FastAPI‑based RAG backend that ingests company PDFs (policies, T&amp;C, technical docs) into ChromaDB and answers natural‑language questions with source citations using SentenceTransformers and a local Ollama phi3:mini LLM.
=======
# 🚀 Enterprise Document Insight Engine

Production-grade RAG system for instant Q&A from enterprise documents with citations.

## 🎯 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

Server: `http://localhost:8000`

## 📡 API Usage

### Ingest Document
```bash
curl -X POST "http://localhost:8000/ingest" -F "file=@document.pdf"
```

### Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "top_k": 3}'
```

## 🛠️ Tech Stack

- FastAPI, LangChain, ChromaDB
- Mistral-7B (4-bit quantized)
- sentence-transformers

## 🧪 Testing
```bash
python test_api.py
```

## 📄 License

MIT
>>>>>>> 5770c86 (Initial RAG app with Ollama backend)
