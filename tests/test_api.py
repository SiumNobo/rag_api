"""
Test Suite for RAG API
======================
Tests the main functionality of the RAG API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_root_endpoint(self):
        """Test the root health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "supported_formats" in data
    
    def test_stats_endpoint(self):
        """Test the stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_files" in data
        assert "total_chunks" in data


class TestDocumentUpload:
    """Test document upload functionality"""
    
    def test_upload_txt_file(self, tmp_path):
        """Test uploading a text file"""
        # Create a sample text file
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("This is a sample document for testing the RAG API. It contains important information about machine learning and artificial intelligence.")
        
        with open(txt_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("sample.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["filename"] == "sample.txt"
        assert data["chunks_created"] >= 1
        assert "file_id" in data
    
    def test_upload_unsupported_format(self, tmp_path):
        """Test uploading an unsupported file format"""
        # Create a file with unsupported extension
        bad_file = tmp_path / "sample.xyz"
        bad_file.write_text("test content")
        
        with open(bad_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("sample.xyz", f, "application/octet-stream")}
            )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


class TestQueryEndpoint:
    """Test the query endpoint"""
    
    def test_query_without_documents(self):
        """Test querying when no documents are indexed"""
        response = client.post(
            "/query",
            json={"question": "What is machine learning?"}
        )
        
        # Should return 200 with empty/no results message
        assert response.status_code == 200
    
    def test_query_with_document(self, tmp_path):
        """Test full workflow: upload then query"""
        # First upload a document
        txt_file = tmp_path / "ml_intro.txt"
        txt_file.write_text("""
        Machine Learning Introduction
        
        Machine learning is a subset of artificial intelligence that enables 
        computers to learn and improve from experience without being explicitly 
        programmed. It focuses on developing algorithms that can access data 
        and use it to learn for themselves.
        
        Key concepts include:
        1. Supervised Learning: Learning from labeled data
        2. Unsupervised Learning: Finding patterns in unlabeled data
        3. Reinforcement Learning: Learning through trial and error
        
        Applications of machine learning include image recognition, 
        natural language processing, and recommendation systems.
        """)
        
        with open(txt_file, "rb") as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("ml_intro.txt", f, "text/plain")}
            )
        
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Now query
        query_response = client.post(
            "/query",
            json={
                "question": "What is machine learning?",
                "top_k": 3
            }
        )
        
        assert query_response.status_code == 200
        data = query_response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        
        # Cleanup
        client.delete(f"/files/{file_id}")


class TestFileManagement:
    """Test file management endpoints"""
    
    def test_list_files(self):
        """Test listing files"""
        response = client.get("/files")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_delete_nonexistent_file(self):
        """Test deleting a file that doesn't exist"""
        response = client.delete("/files/nonexistent123")
        assert response.status_code == 404


class TestQueryOptions:
    """Test query with various options"""
    
    def test_query_with_top_k(self, tmp_path):
        """Test query with custom top_k parameter"""
        # Upload a document first
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test document content for RAG API testing.")
        
        with open(txt_file, "rb") as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        file_id = upload_response.json()["file_id"]
        
        # Query with specific top_k
        response = client.post(
            "/query",
            json={
                "question": "What is this about?",
                "top_k": 1
            }
        )
        
        assert response.status_code == 200
        
        # Cleanup
        client.delete(f"/files/{file_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
