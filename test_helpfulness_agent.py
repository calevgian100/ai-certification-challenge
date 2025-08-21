"""
Test script for the PubMed CrossFit Agent with integrated helpfulness evaluation
"""
import os
import sys
import time

# Set up environment variables if needed
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

# Import the agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api.features.agents.pub_med_agent import create_pubmed_agent

def test_helpfulness_evaluation():
    """Test the PubMed agent with integrated helpfulness evaluation"""
    print("\n🧪 Testing PubMed CrossFit Agent with helpfulness evaluation...\n")
    
    # Create the agent
    agent = create_pubmed_agent(
        initialize_enhanced_rag=True,
        max_attempts=3  # Set maximum number of refinement attempts
    )
    
    # Test queries
    test_queries = [
        "What are the benefits of CrossFit for older adults?",
        "How should I modify burpees if I have knee pain?",
        "Tell me about the history of CrossFit games",  # Intentionally vague to trigger refinement
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n\n📝 Test Query {i+1}: {query}")
        print("=" * 80)
        
        # Time the response
        start_time = time.time()
        result = agent.query(query)
        end_time = time.time()
        
        # Print results
        print(f"\n✅ Response (took {end_time - start_time:.2f}s):")
        print("-" * 80)
        print(result["response"])
        print("-" * 80)
        
        # Print helpfulness metrics
        print(f"\n📊 Helpfulness Score: {result.get('helpfulness_score', 'N/A')}")
        print(f"🔄 Refinement Attempts: {result.get('attempts', 1)}")
        
        # Print sources if available
        if result.get("pubmed_sources") or result.get("local_sources"):
            print("\n📚 Sources Used:")
            
            if result.get("pubmed_sources"):
                print(f"  - PubMed: {len(result['pubmed_sources'])} articles")
                
            if result.get("local_sources"):
                print(f"  - Local: {len(result['local_sources'])} documents")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    test_helpfulness_evaluation()
