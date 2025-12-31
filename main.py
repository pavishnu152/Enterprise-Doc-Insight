from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from typing import List
import uvicorn
import requests


app = FastAPI(
    title="Enterprise Document Insight Engine",
    description="RAG-based document Q&A system with source citations",
    version="1.0.0",
)

# Global variables
embedding_model = None
chroma_client = None
collection = None


class Query(BaseModel):
    question: str
    top_k: int = 3


class IngestResponse(BaseModel):
    status: str
    chunks_processed: int
    filename: str


@app.on_event("startup")
async def startup_event():
    """Initialize embedding model and ChromaDB on server startup."""
    global embedding_model, chroma_client, collection

    print("Starting Enterprise Document Insight Engine...")

    # Embedding model
    print("Loading embedding model (sentence-transformers)...")
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("Embedding model loaded.")

    # ChromaDB
    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="enterprise_docs",
        metadata={"hnsw:space": "cosine"},
    )
    print(f"ChromaDB initialized. Documents in DB: {collection.count()}")

    print("System initialized and ready to serve requests.")
    print("API Docs available at: http://localhost:8000/docs")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Enterprise Document Insight Engine",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "ingest": "/ingest",
            "query": "/query",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "documents_count": collection.count(),
        "embedding_model": "all-mpnet-base-v2",
        "vector_db": "ChromaDB",
        "llm": "ollama: phi3:mini",
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest PDF document into vector database."""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    print(f"Processing document: {file.filename}")

    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        print("Loading PDF...")
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages.")

        print("Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        print("Generating embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        ).tolist()
        print("Embeddings generated.")

        metadatas = [
            {
                "source": file.filename,
                "page": chunk.metadata.get("page", 0),
                "chunk_id": i,
            }
            for i, chunk in enumerate(chunks)
        ]

        ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

        print("Storing in ChromaDB...")
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        print("Successfully stored in database.")

        return IngestResponse(
            status="success",
            chunks_processed=len(chunks),
            filename=file.filename,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )

    finally:
        os.unlink(tmp_path)


@app.post("/query")
async def query_documents(query: Query):
    """Query documents with RAG pipeline (retrieval + Ollama LLM)."""

    print(f"Query received: {query.question}")

    if collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents in database. Please ingest documents first using /ingest endpoint.",
        )

    print("Generating query embedding...")
    query_embedding = embedding_model.encode([query.question]).tolist()

    print(f"Searching for top {query.top_k} relevant chunks...")
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=query.top_k,
    )

    if not results["documents"][0]:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found for your query.",
        )

    context_parts = []
    citations = []

    for i, (doc, metadata) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        citation = f"[{i+1}] {metadata['source']} - Page {metadata['page']}"
        citations.append(citation)
        context_parts.append(f"Context {i+1}:\n{doc}")

    context = "\n\n".join(context_parts)
    print(f"Retrieved {len(citations)} relevant chunks.")

    prompt = f"""You are a precise document assistant. Answer the question using ONLY the provided context. Include citation numbers [1], [2], etc. in your answer.

Context:
{context}

Question: {query.question}

Instructions:
- Answer directly and factually
- Use citation numbers [1], [2] when referencing context
- If context doesn't contain the answer, say "Information not found in provided documents"
- Do not make assumptions or add information not in context
"""

    print("Generating answer with LLM via Ollama...")
    ollama_url = "http://localhost:11434/api/chat"

    payload = {
        "model": "phi3:mini",
        "messages": [
            {"role": "system", "content": "You are a precise document assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False  # ensure a single JSON object
    }

    try:
        resp = requests.post(ollama_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        answer_text = data["message"]["content"]
        print("Answer generated via Ollama.")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer with LLM: {str(e)}",
        )

    return {
        "answer": answer_text,
        "citations": citations,
        "num_sources": len(citations),
        "query": query.question,
    }


if __name__ == "__main__":
    print("Starting server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
