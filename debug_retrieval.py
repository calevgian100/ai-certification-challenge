#!/usr/bin/env python3
"""
Debug script to test document retrieval for Peter Parker query
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from api.features.store.vector_store import QdrantVectorStore

def debug_retrieval():
    """Debug document retrieval for Peter Parker query"""
    
    # Initialize vector store
    vector_store = QdrantVectorStore()
    
    # Test query
    query = "I'm Peter Parker, what injury do I have?"
    
    print(f"🔍 Testing query: '{query}'")
    print("=" * 60)
    
    try:
        # Get search results
        results = vector_store.similarity_search(query, k=10)
        
        print(f"📊 Retrieved {len(results)} documents:")
        print("-" * 40)
        
        for i, result in enumerate(results):
            print(f"\n📄 Document {i+1}:")
            print(f"   Source: {result.get('source', 'Unknown')}")
            print(f"   Score: {result.get('score', 'N/A')}")
            print(f"   Text preview: {result.get('text', '')[:200]}...")
            
            # Check if this document mentions Peter Parker
            text = result.get('text', '').lower()
            if 'peter parker' in text:
                print(f"   ✅ Contains 'Peter Parker'")
            else:
                print(f"   ❌ No 'Peter Parker' mention")
        
        # Also check all PDFs in the system
        print("\n" + "=" * 60)
        print("📚 All PDFs in system:")
        print("-" * 40)
        
        all_pdfs = vector_store.get_all_pdf_metadata()
        for pdf in all_pdfs:
            filename = pdf.get('filename', 'Unknown')
            file_id = pdf.get('file_id', 'Unknown')
            num_chunks = pdf.get('num_chunks', 0)
            print(f"   📄 {filename} (ID: {file_id}, Chunks: {num_chunks})")
            
            # Check if filename suggests Peter Parker content
            if 'peter' in filename.lower() or 'parker' in filename.lower():
                print(f"      ✅ Filename suggests Peter Parker content")
        
    except Exception as e:
        print(f"❌ Error during retrieval test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_retrieval()
