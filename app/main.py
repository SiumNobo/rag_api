"""
Smart RAG API - Retrieval-Augmented Generation System
=====================================================
A FastAPI application that processes multiple document types,
creates embeddings, and answers questions using LLM + vector search.

Author: Sium
"""

import os
import uuid
import base64
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.document_processor import DocumentProcessor
from utils.embeddings import EmbeddingManager
from utils.vector_store import VectorStoreManager
from utils.llm_handler import LLMHandler

# Initialize FastAPI app
app = FastAPI(
    title="Smart RAG API",
    description="Retrieval-Augmented Generation API supporting multiple document types",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
doc_processor = DocumentProcessor()
embedding_manager = EmbeddingManager()
vector_store = VectorStoreManager()
llm_handler = LLMHandler()

# In-memory file registry (in production, use a database)
file_registry: Dict[str, Dict[str, Any]] = {}


# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    """Request model for querying documents"""
    question: str = Field(..., description="The question to ask")
    image_base64: Optional[str] = Field(None, description="Optional base64 encoded image for OCR")
    file_ids: Optional[List[str]] = Field(None, description="Specific file IDs to search (searches all if empty)")
    top_k: int = Field(5, description="Number of relevant chunks to retrieve")

class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    context: List[Dict[str, Any]]
    sources: List[Dict[str, str]]
    confidence: float
    processing_time: float

class UploadResponse(BaseModel):
    """Response model for file uploads"""
    file_id: str
    filename: str
    file_type: str
    chunks_created: int
    status: str
    message: str

class FileInfo(BaseModel):
    """Model for file information"""
    file_id: str
    filename: str
    file_type: str
    upload_time: str
    chunks_count: int
    status: str


# ==================== API Endpoints ====================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Smart RAG API",
        "version": "1.0.0",
        "supported_formats": [".pdf", ".docx", ".txt", ".jpg", ".png", ".csv", ".db"]
    }


@app.post("/upload", response_model=UploadResponse, tags=["Document Management"])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a document for processing and indexing.
    
    Supported formats:
    - PDF (.pdf) - including scanned documents with OCR
    - Word (.docx)
    - Text (.txt)
    - Images (.jpg, .png) - processed with OCR
    - CSV (.csv)
    - SQLite (.db)
    
    Returns a file_id for future reference.
    """
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())[:8]
        
        # Get file extension
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Validate file type
        supported_types = ['.pdf', '.docx', '.txt', '.jpg', '.jpeg', '.png', '.csv', '.db']
        if file_ext not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {supported_types}"
            )
        
        # Read file content
        content = await file.read()
        
        # Save file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{file_id}_{filename}")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process document and extract text chunks
        chunks = doc_processor.process_document(file_path, file_ext)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any content from the document"
            )
        
        # Generate embeddings for chunks
        embeddings = embedding_manager.generate_embeddings([c["text"] for c in chunks])
        
        # Store in vector database with metadata
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk["file_id"] = file_id
            chunk["filename"] = filename
            chunk["chunk_index"] = i
        
        vector_store.add_documents(chunks, embeddings, file_id)
        
        # Register file
        file_registry[file_id] = {
            "filename": filename,
            "file_type": file_ext,
            "upload_time": datetime.now().isoformat(),
            "chunks_count": len(chunks),
            "file_path": file_path,
            "status": "indexed"
        }
        
        return UploadResponse(
            file_id=file_id,
            filename=filename,
            file_type=file_ext,
            chunks_created=len(chunks),
            status="success",
            message=f"Document processed successfully. {len(chunks)} chunks indexed."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the indexed documents with a question.
    
    Supports:
    - Text questions against indexed documents
    - Image-based questions (OCR extracts text from image)
    - Multi-document querying
    - Configurable number of context chunks
    
    Returns the answer, context used, and source information.
    """
    import time
    start_time = time.time()
    
    try:
        question = request.question
        
        # If image is provided, extract text via OCR and append to question
        if request.image_base64:
            image_text = doc_processor.process_base64_image(request.image_base64)
            if image_text:
                question = f"{question}\n\nText extracted from image:\n{image_text}"
        
        # Generate embedding for the question
        question_embedding = embedding_manager.generate_embeddings([question])[0]
        
        # Search vector store
        search_results = vector_store.search(
            question_embedding,
            top_k=request.top_k,
            file_ids=request.file_ids
        )
        
        if not search_results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the indexed documents to answer your question.",
                context=[],
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Prepare context from search results
        context_texts = []
        sources = []
        context_details = []
        
        for result in search_results:
            context_texts.append(result["text"])
            sources.append({
                "filename": result.get("filename", "unknown"),
                "page": str(result.get("page", "N/A")),
                "chunk_index": str(result.get("chunk_index", 0)),
                "relevance_score": f"{result.get('score', 0):.3f}"
            })
            context_details.append({
                "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                "metadata": {
                    "filename": result.get("filename"),
                    "page": result.get("page"),
                    "score": result.get("score")
                }
            })
        
        # Construct prompt and get LLM response
        context_str = "\n\n---\n\n".join(context_texts)
        answer = llm_handler.generate_answer(question, context_str)
        
        # Calculate confidence based on similarity scores
        avg_score = sum(r.get("score", 0) for r in search_results) / len(search_results)
        confidence = min(avg_score, 1.0)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            context=context_details,
            sources=sources,
            confidence=confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/files", response_model=List[FileInfo], tags=["Document Management"])
async def list_files():
    """List all indexed files"""
    files = []
    for file_id, info in file_registry.items():
        files.append(FileInfo(
            file_id=file_id,
            filename=info["filename"],
            file_type=info["file_type"],
            upload_time=info["upload_time"],
            chunks_count=info["chunks_count"],
            status=info["status"]
        ))
    return files


@app.delete("/files/{file_id}", tags=["Document Management"])
async def delete_file(file_id: str):
    """Delete an indexed file and its embeddings"""
    if file_id not in file_registry:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Remove from vector store
    vector_store.delete_by_file_id(file_id)
    
    # Remove file
    file_info = file_registry.pop(file_id)
    if os.path.exists(file_info["file_path"]):
        os.remove(file_info["file_path"])
    
    return {"status": "success", "message": f"File {file_id} deleted successfully"}


@app.get("/stats", tags=["Health"])
async def get_stats():
    """Get system statistics"""
    return {
        "total_files": len(file_registry),
        "total_chunks": sum(f["chunks_count"] for f in file_registry.values()),
        "vector_store_size": vector_store.get_size(),
        "supported_formats": [".pdf", ".docx", ".txt", ".jpg", ".png", ".csv", ".db"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
