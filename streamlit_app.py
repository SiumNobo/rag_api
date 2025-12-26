"""
RAG API - Streamlit Frontend
============================
A minimal web interface for the RAG API.
Allows users to upload documents and ask questions.
"""

import streamlit as st
import requests
import base64
import os

# Configuration
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Smart RAG Assistant",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Smart RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload documents and ask questions powered by AI</p>', unsafe_allow_html=True)

# Sidebar for file management
with st.sidebar:
    st.header("📁 Document Management")
    
    # File upload
    st.subheader("Upload New Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'csv'],
        help="Supported formats: PDF, Word, Text, Images, CSV"
    )
    
    if uploaded_file:
        if st.button("📤 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Uploaded successfully!")
                        st.info(f"File ID: `{result['file_id']}`")
                        st.info(f"Chunks created: {result['chunks_created']}")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
    
    st.divider()
    
    # List files
    st.subheader("Indexed Documents")
    if st.button("🔄 Refresh List"):
        st.rerun()
    
    try:
        response = requests.get(f"{API_URL}/files")
        if response.status_code == 200:
            files = response.json()
            if files:
                for file in files:
                    with st.expander(f"📄 {file['filename'][:20]}..."):
                        st.write(f"**ID:** `{file['file_id']}`")
                        st.write(f"**Type:** {file['file_type']}")
                        st.write(f"**Chunks:** {file['chunks_count']}")
                        if st.button("🗑️ Delete", key=f"del_{file['file_id']}"):
                            requests.delete(f"{API_URL}/files/{file['file_id']}")
                            st.rerun()
            else:
                st.info("No documents indexed yet.")
    except Exception as e:
        st.warning(f"Could not load files: {str(e)}")
    
    st.divider()
    
    # Stats
    st.subheader("📊 System Stats")
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            col1, col2 = st.columns(2)
            col1.metric("Files", stats['total_files'])
            col2.metric("Chunks", stats['total_chunks'])
    except:
        pass

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask a Question")
    
    # Question input
    question = st.text_area(
        "Your question:",
        placeholder="e.g., What are the key points mentioned in the document?",
        height=100
    )
    
    # Optional image upload for OCR
    with st.expander("📷 Add an image (optional)"):
        image_file = st.file_uploader(
            "Upload an image to include in your question",
            type=['jpg', 'jpeg', 'png'],
            key="question_image"
        )
        image_base64 = None
        if image_file:
            image_base64 = base64.b64encode(image_file.getvalue()).decode()
            st.image(image_file, caption="Uploaded image", width=300)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        top_k = st.slider("Number of context chunks", 1, 10, 5)
    
    # Submit button
    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    payload = {
                        "question": question,
                        "top_k": top_k
                    }
                    if image_base64:
                        payload["image_base64"] = image_base64
                    
                    response = requests.post(
                        f"{API_URL}/query",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display answer
                        st.subheader("📝 Answer")
                        st.markdown(result['answer'])
                        
                        # Display metrics
                        st.divider()
                        mcol1, mcol2 = st.columns(2)
                        mcol1.metric("Confidence", f"{result['confidence']:.1%}")
                        mcol2.metric("Processing Time", f"{result['processing_time']:.2f}s")
                        
                        # Display sources
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources']):
                            with st.container():
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i+1}:</strong> {source['filename']}<br>
                                    <small>Page: {source['page']} | Relevance: {source['relevance_score']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Display context (expandable)
                        with st.expander("🔍 View Retrieved Context"):
                            for i, ctx in enumerate(result['context']):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.text(ctx['text'])
                                st.divider()
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Make sure the server is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.header("ℹ️ Quick Guide")
    
    st.markdown("""
    ### How to use:
    
    1. **Upload Documents**
       - Use the sidebar to upload PDF, Word, Text, Images, or CSV files
       - Documents are automatically processed and indexed
    
    2. **Ask Questions**
       - Type your question in the text box
       - Optionally add an image for OCR
       - Click "Get Answer" to query
    
    3. **Review Results**
       - See the AI-generated answer
       - Check confidence score
       - View source documents
    
    ### Supported Formats:
    - 📄 PDF (text & scanned)
    - 📝 Word (.docx)
    - 📃 Text (.txt)
    - 🖼️ Images (.jpg, .png)
    - 📊 CSV files
    """)
    
    st.divider()
    
    st.markdown("""
    ### Example Questions:
    - "What are the main topics covered?"
    - "Summarize the key findings"
    - "What does it say about [topic]?"
    - "List all mentioned dates/names"
    """)

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Smart RAG API - Powered by FAISS & OpenAI</p>",
    unsafe_allow_html=True
)
