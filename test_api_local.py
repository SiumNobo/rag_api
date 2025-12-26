"""
Local API Test Script
Run this after starting the server to verify everything works.
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"

def test_api():
    print("=" * 50)
    print("  RAG API Test Script")
    print("=" * 50)
    print()
    
    # Test 1: Health check
    print("[1/4] Testing health endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/")
        if r.status_code == 200:
            print(f"  ✓ Server is running!")
            print(f"  Supported formats: {r.json()['supported_formats']}")
        else:
            print(f"  ✗ Server returned status {r.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("  ✗ Could not connect to server!")
        print("  Make sure the server is running (run_server.bat)")
        return
    print()
    
    # Test 2: Upload sample document
    print("[2/4] Uploading sample document...")
    sample_file = "sample_files/sample_invoice.txt"
    
    if not os.path.exists(sample_file):
        # Create a sample if it doesn't exist
        os.makedirs("sample_files", exist_ok=True)
        with open(sample_file, "w") as f:
            f.write("""Invoice #INV-2024-001
Company: TechCorp Solutions
Date: January 15, 2024
Total: $24,192.00

Payment Terms: Net 30 days
Late Fee: 1.5% monthly

Items:
1. Cloud Infrastructure Setup - $5,000
2. Software Development - $15,000
3. Annual Support License - $2,400

Contact: billing@techcorp.com
""")
    
    with open(sample_file, "rb") as f:
        r = requests.post(f"{BASE_URL}/upload", files={"file": f})
    
    if r.status_code == 200:
        result = r.json()
        file_id = result["file_id"]
        print(f"  ✓ Document uploaded!")
        print(f"  File ID: {file_id}")
        print(f"  Chunks created: {result['chunks_created']}")
    else:
        print(f"  ✗ Upload failed: {r.json()}")
        return
    print()
    
    # Test 3: Query the document
    print("[3/4] Querying document...")
    questions = [
        "What is the total amount?",
        "What are the payment terms?",
        "What services were provided?"
    ]
    
    for q in questions:
        print(f"\n  Question: {q}")
        r = requests.post(
            f"{BASE_URL}/query",
            json={"question": q, "top_k": 3}
        )
        
        if r.status_code == 200:
            result = r.json()
            answer = result["answer"]
            # Truncate long answers for display
            if len(answer) > 200:
                answer = answer[:200] + "..."
            print(f"  Answer: {answer}")
            print(f"  Confidence: {result['confidence']:.1%}")
        else:
            print(f"  ✗ Query failed: {r.json()}")
    print()
    
    # Test 4: List files and cleanup
    print("[4/4] Checking indexed files...")
    r = requests.get(f"{BASE_URL}/files")
    if r.status_code == 200:
        files = r.json()
        print(f"  ✓ Found {len(files)} indexed file(s)")
        for f in files:
            print(f"    - {f['filename']} (ID: {f['file_id']}, Chunks: {f['chunks_count']})")
    print()
    
    print("=" * 50)
    print("  All tests completed!")
    print("=" * 50)
    print()
    print("Try the interactive API docs at: http://localhost:8000/docs")

if __name__ == "__main__":
    test_api()
