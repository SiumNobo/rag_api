"""
LLM Handler Module
==================
Manages interactions with Large Language Models for answer generation.
Supports:
- Groq (Llama, Mixtral - FAST & FREE)
- Google Gemini
- OpenAI (GPT-3.5, GPT-4)
- Anthropic Claude

Implements prompt engineering for RAG-based question answering.
"""

import os
from typing import Optional, List, Dict, Any

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None


class LLMHandler:
    """
    Handles LLM interactions for RAG answer generation.
    Provides a unified interface for multiple LLM providers.
    """
    
    def __init__(
        self,
        provider: str = "auto",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize the LLM handler.
        
        Args:
            provider: 'groq', 'gemini', 'openai', 'anthropic', or 'auto'
            model_name: Specific model to use
            api_key: API key (defaults to environment variables)
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API keys from environment
        self.groq_key = api_key or os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        self.client = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the LLM provider"""
        
        if self.provider == "auto":
            # Priority: Groq (fast & free) > Gemini > OpenAI > Anthropic
            if self.groq_key and Groq:
                self.provider = "groq"
            elif self.gemini_key and genai:
                self.provider = "gemini"
            elif self.openai_key and OpenAI:
                self.provider = "openai"
            elif self.anthropic_key and anthropic:
                self.provider = "anthropic"
            else:
                # Fall back to mock for demo
                self.provider = "mock"
                print("[LLMHandler] No API keys found, using mock responses")
                return
        
        if self.provider == "groq":
            self._init_groq()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "mock":
            pass
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _init_groq(self):
        """Initialize Groq client (Fast & Free!)"""
        if not Groq:
            raise ImportError("groq package not installed. Run: pip install groq")
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY required")
        
        self.client = Groq(api_key=self.groq_key)
        # Groq models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
        self.model_name = self.model_name or "llama-3.3-70b-versatile"
        print(f"[LLMHandler] Using Groq: {self.model_name}")
    
    def _init_gemini(self):
        """Initialize Google Gemini client"""
        if not genai:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        
        genai.configure(api_key=self.gemini_key)
        # Gemini models: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
        self.model_name = self.model_name or "gemini-1.5-flash"
        self.client = genai.GenerativeModel(self.model_name)
        print(f"[LLMHandler] Using Gemini: {self.model_name}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if not OpenAI:
            raise ImportError("openai package not installed")
        if not self.openai_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.openai_key)
        self.model_name = self.model_name or "gpt-3.5-turbo"
        print(f"[LLMHandler] Using OpenAI: {self.model_name}")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        if not anthropic:
            raise ImportError("anthropic package not installed")
        if not self.anthropic_key:
            raise ValueError("Anthropic API key required")
        
        self.client = anthropic.Anthropic(api_key=self.anthropic_key)
        self.model_name = self.model_name or "claude-3-sonnet-20240229"
        print(f"[LLMHandler] Using Anthropic: {self.model_name}")
    
    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer based on the question and context.
        
        Args:
            question: The user's question
            context: Retrieved context from vector search
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated answer string
        """
        # Build the prompt
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()
        
        user_prompt = self._build_user_prompt(question, context)
        
        # Generate response based on provider
        if self.provider == "groq":
            return self._generate_groq(system_prompt, user_prompt)
        elif self.provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        else:
            return self._generate_mock(question, context)
    
    def _get_default_system_prompt(self) -> str:
        """Get the default RAG system prompt"""
        return """You are a helpful AI assistant that answers questions based on the provided context.

Your responsibilities:
1. Answer questions accurately using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite relevant parts of the context when appropriate
4. Be concise but thorough in your answers
5. If you're unsure, express uncertainty rather than making up information

Important: Do not use any external knowledge. Base your answer solely on the provided context."""
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        """Build the user prompt with context and question"""
        return f"""Context from documents:
---
{context}
---

Question: {question}

Please provide a helpful answer based on the context above. If the context doesn't contain relevant information, let me know."""
    
    def _generate_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Groq (Llama/Mixtral) - FAST!"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response with Groq: {str(e)}"
    
    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Google Gemini"""
        try:
            # Combine system and user prompt for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating response with Gemini: {str(e)}"
    
    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Anthropic Claude"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_mock(self, question: str, context: str) -> str:
        """Generate a mock response for testing without API keys"""
        # Extract key information from context for demo
        context_preview = context[:500] + "..." if len(context) > 500 else context
        
        return f"""Based on the provided documents, here's what I found relevant to your question:

**Question:** {question}

**Summary from context:**
The documents contain information that may be relevant to your query. Here are the key points from the retrieved context:

{context_preview}

**Note:** This is a demo response. In production with a configured LLM API (OpenAI or Anthropic), you would receive a more detailed and contextual answer synthesized from the document content.

To enable full functionality, please configure your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."""
    
    def generate_multimodal_answer(
        self,
        question: str,
        context: str,
        image_base64: Optional[str] = None
    ) -> str:
        """
        Generate an answer using multimodal capabilities (image + text).
        Only available with GPT-4 Vision or Claude 3.
        
        Args:
            question: The user's question
            context: Retrieved text context
            image_base64: Optional base64 encoded image
            
        Returns:
            Generated answer string
        """
        if not image_base64:
            return self.generate_answer(question, context)
        
        if self.provider == "openai" and "gpt-4" in self.model_name.lower():
            return self._generate_openai_vision(question, context, image_base64)
        elif self.provider == "anthropic":
            return self._generate_anthropic_vision(question, context, image_base64)
        else:
            # Fall back to text-only
            return self.generate_answer(question, context)
    
    def _generate_openai_vision(
        self,
        question: str,
        context: str,
        image_base64: str
    ) -> str:
        """Generate response using GPT-4 Vision"""
        try:
            # Ensure proper data URL format
            if not image_base64.startswith("data:"):
                image_base64 = f"data:image/jpeg;base64,{image_base64}"
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_default_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._build_user_prompt(question, context)
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_base64}
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with vision model: {str(e)}"
    
    def _generate_anthropic_vision(
        self,
        question: str,
        context: str,
        image_base64: str
    ) -> str:
        """Generate response using Claude 3 Vision"""
        try:
            # Remove data URL prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self._get_default_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._build_user_prompt(question, context)
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error with vision model: {str(e)}"
