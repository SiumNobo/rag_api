"""
Embedding Manager Module
========================
Handles generation of text embeddings using various providers:
- OpenAI embeddings (ada-002, text-embedding-3-small/large)
- SentenceTransformers (local, no API needed)
- HuggingFace Inference API

Provides a unified interface for embedding generation.
"""

import os
from typing import List, Optional, Union
import numpy as np

# Try to import embedding libraries
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingManager:
    """
    Manages text embedding generation with multiple backend support.
    Falls back to local SentenceTransformers if OpenAI is unavailable.
    """
    
    def __init__(
        self,
        provider: str = "auto",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the embedding manager.
        
        Args:
            provider: 'openai', 'sentence-transformers', or 'auto' (tries OpenAI first)
            model_name: Specific model to use
            api_key: API key for OpenAI (defaults to OPENAI_API_KEY env var)
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = None
        self.model_name = model_name
        self.embedding_dim = None
        
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the embedding provider based on configuration"""
        
        if self.provider == "auto":
            # Try OpenAI first, fall back to SentenceTransformers
            if self.api_key and OpenAI:
                self.provider = "openai"
            elif SentenceTransformer:
                self.provider = "sentence-transformers"
            else:
                raise ImportError(
                    "No embedding provider available. "
                    "Install openai or sentence-transformers."
                )
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "sentence-transformers":
            self._init_sentence_transformers()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI embedding client"""
        if not OpenAI:
            raise ImportError("openai package not installed")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = self.model_name or "text-embedding-3-small"
        
        # Embedding dimensions for OpenAI models
        dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        self.embedding_dim = dims.get(self.model_name, 1536)
        
        print(f"[EmbeddingManager] Using OpenAI: {self.model_name}")
    
    def _init_sentence_transformers(self):
        """Initialize SentenceTransformers model"""
        if not SentenceTransformer:
            raise ImportError("sentence-transformers package not installed")
        
        # Use a good default model
        self.model_name = self.model_name or "all-MiniLM-L6-v2"
        
        print(f"[EmbeddingManager] Loading SentenceTransformer: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingManager] Model loaded. Dimension: {self.embedding_dim}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of numpy arrays containing embeddings
        """
        if not texts:
            return []
        
        if self.provider == "openai":
            return self._generate_openai_embeddings(texts, batch_size)
        else:
            return self._generate_st_embeddings(texts, batch_size)
    
    def _generate_openai_embeddings(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean texts (OpenAI has length limits)
            batch = [self._truncate_text(t, max_tokens=8000) for t in batch]
            
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            for item in response.data:
                all_embeddings.append(np.array(item.embedding, dtype=np.float32))
        
        return all_embeddings
    
    def _generate_st_embeddings(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[np.ndarray]:
        """Generate embeddings using SentenceTransformers"""
        # SentenceTransformers handles batching internally
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        
        return [emb.astype(np.float32) for emb in embeddings]
    
    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        Truncate text to approximate token limit.
        Uses rough estimate of 4 chars per token.
        """
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this manager"""
        return self.embedding_dim
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


# Singleton for shared embedding manager
_embedding_manager: Optional[EmbeddingManager] = None

def get_embedding_manager(**kwargs) -> EmbeddingManager:
    """Get or create a shared embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(**kwargs)
    return _embedding_manager
