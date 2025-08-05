#!/usr/bin/env python3
"""
Test script for Enhanced PubMed CrossFit Agent
Demonstrates the improved chunking and retrieval capabilities
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from api.features.agents.pub_med_agent import create_pubmed_agent

def load_env_vars():
    """Load environment variables from env.yaml"""
    env_vars = {}
    try:
        env_file_path = project_root / 'env.yaml'
        if env_file_path.exists():
            with open(env_file_path, 'r') as file:
                content = file.read()
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('\'"')
                        if value:
                            env_vars[key] = value
                            os.environ[key.upper()] = value
        return env_vars
    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return {}

async def test_enhanced_agent():
    """Test the enhanced agent with various queries"""
    
    print("🚀 Testing Enhanced PubMed CrossFit Agent")
    print("=" * 50)
    
    # Load environment variables
    env_vars = load_env_vars()
    
    # Create agent with enhanced RAG
    data_path = str(project_root / "api" / "data")
    print(f"📁 Using data path: {data_path}")
    
    try:
        agent = create_pubmed_agent(
            data_path=data_path,
            initialize_enhanced_rag=True,
            openai_api_key=env_vars.get('openai_api_key'),
            langsmith_api_key=env_vars.get('langsmith_api_key')
        )
        
        print("\n✅ Agent created successfully!")
        
        # Test queries that should benefit from enhanced chunking
        test_queries = [
            "What are the common faults in deadlifts and how to correct them?",
            "How can I use medicine ball training for better squats?",
            "What does triaging mean in movement coaching?",
            "What are the safety considerations for CrossFit training?",
            "How should I modify workouts for athletes with injuries?"
        ]
        
        print(f"\n🧪 Testing {len(test_queries)} queries with Enhanced RAG:")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 30)
            
            try:
                # Test local document search directly
                local_results = agent.search_local_documents(query)
                
                print(f"📊 Found {len(local_results)} local document results")
                
                if local_results:
                    # Show first result preview
                    first_result = local_results[0]
                    content_preview = first_result['content'][:200] + "..." if len(first_result['content']) > 200 else first_result['content']
                    print(f"🔍 Preview: {content_preview}")
                    print(f"📄 Source: {first_result['source']}")
                    print(f"⭐ Relevance: {first_result['relevance_score']:.3f}")
                else:
                    print("❌ No results found")
                
                # Test full agent query
                print("\n🤖 Full Agent Response:")
                response = await agent.aquery(query)
                
                if response and 'messages' in response:
                    final_message = response['messages'][-1]
                    if hasattr(final_message, 'content'):
                        response_preview = final_message.content[:300] + "..." if len(final_message.content) > 300 else final_message.content
                        print(f"💬 Response: {response_preview}")
                    else:
                        print(f"💬 Response: {str(final_message)[:300]}...")
                
            except Exception as e:
                print(f"❌ Error testing query: {e}")
            
            print("\n" + "="*50)
        
        # Show retriever information
        if agent.enhanced_rag_initialized:
            print("\n📊 Enhanced RAG Configuration:")
            info = agent.enhanced_rag_engine.get_retriever_info()
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        print("\n🎉 Enhanced Agent testing completed!")
        
    except Exception as e:
        print(f"❌ Error creating or testing agent: {e}")
        import traceback
        traceback.print_exc()

def test_retriever_comparison():
    """Compare different retriever types"""
    
    print("\n🔬 Comparing Retriever Types")
    print("=" * 50)
    
    env_vars = load_env_vars()
    data_path = str(project_root / "api" / "data")
    
    retriever_types = ["ensemble", "bm25", "semantic"]
    test_query = "What are the common faults in deadlifts?"
    
    for retriever_type in retriever_types:
        print(f"\n🧪 Testing {retriever_type} retriever:")
        print("-" * 30)
        
        try:
            # Create agent with specific retriever type
            from api.features.rag.enhanced_rag import EnhancedRAGQueryEngine
            
            rag_engine = EnhancedRAGQueryEngine(
                retriever_type=retriever_type,
                chunk_size=1000,
                chunk_overlap=200,
                k=3
            )
            
            success = rag_engine.load_documents_from_directory(data_path)
            
            if success:
                results = rag_engine.query(test_query)
                
                print(f"📊 Results found: {len(results.get('sources', []))}")
                
                if results.get('sources'):
                    first_source = results['sources'][0]
                    content_preview = first_source['text'][:150] + "..." if len(first_source['text']) > 150 else first_source['text']
                    print(f"🔍 Top result: {content_preview}")
                    print(f"📄 Source: {first_source.get('source', 'Unknown')}")
            else:
                print("❌ Failed to load documents")
                
        except Exception as e:
            print(f"❌ Error testing {retriever_type}: {e}")

if __name__ == "__main__":
    print("🎯 Enhanced PubMed CrossFit Agent Test Suite")
    print("=" * 60)
    
    # Run main test
    asyncio.run(test_enhanced_agent())
    
    # Run retriever comparison
    test_retriever_comparison()
    
    print("\n✨ All tests completed!")
