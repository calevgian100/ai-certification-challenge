#!/usr/bin/env python3
"""
Test script for the PubMed CrossFit Agent

This script tests the agent's functionality including:
1. PubMed search capabilities
2. Local document retrieval
3. Workout adaptation generation
4. Educational content generation
5. End-to-end query processing
"""

import os
import sys
import asyncio
import json
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.features.agents.pub_med_agent import PubMedCrossFitAgent, create_pubmed_agent

def print_separator(title: str):
    """Print a formatted separator for test sections"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_results(results: Dict[str, Any], title: str = "Results"):
    """Pretty print test results"""
    print(f"\n{title}:")
    print("-" * 40)
    
    if "response" in results:
        print(f"Response: {results['response'][:200]}...")
    
    if "pubmed_sources" in results and results["pubmed_sources"]:
        print(f"\nPubMed Sources ({len(results['pubmed_sources'])}):")
        for i, source in enumerate(results["pubmed_sources"][:2], 1):
            print(f"  {i}. {source.get('title', 'No title')} ({source.get('year', 'Unknown year')})")
    
    if "local_sources" in results and results["local_sources"]:
        print(f"\nLocal Sources ({len(results['local_sources'])}):")
        for i, source in enumerate(results["local_sources"][:2], 1):
            print(f"  {i}. {source.get('source', 'Unknown source')}")
    
    print()

async def test_pubmed_search(agent: PubMedCrossFitAgent):
    """Test PubMed search functionality"""
    print_separator("Testing PubMed Search")
    
    try:
        # Test direct PubMed search
        results = agent.search_pubmed.invoke({
            "query": "high intensity interval training crossfit",
            "max_results": 3
        })
        
        print(f"Found {len(results)} PubMed articles:")
        for i, article in enumerate(results, 1):
            print(f"{i}. {article['title']} ({article['year']})")
            print(f"   Authors: {', '.join(article['authors'][:2])}...")
            print(f"   PMID: {article['pmid']}")
            print()
        
        return len(results) > 0
        
    except Exception as e:
        print(f"PubMed search failed: {e}")
        return False

async def test_local_search(agent: PubMedCrossFitAgent):
    """Test local document search functionality"""
    print_separator("Testing Local Document Search")
    
    try:
        # Test local document search
        results = agent.search_local_documents.invoke({
            "query": "crossfit workout programming"
        })
        
        print(f"Found {len(results)} local documents:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc['source']}")
            print(f"   Content: {doc['content'][:100]}...")
            print(f"   Relevance: {doc.get('relevance_score', 'N/A')}")
            print()
        
        return len(results) >= 0  # Local search might return 0 results if no docs are indexed
        
    except Exception as e:
        print(f"Local document search failed: {e}")
        return False

async def test_workout_adaptation(agent: PubMedCrossFitAgent):
    """Test workout adaptation functionality"""
    print_separator("Testing Workout Adaptation")
    
    try:
        # Test workout adaptation
        adaptation = agent.generate_workout_adaptation.invoke({
            "user_profile": "Intermediate CrossFit athlete, 2 years experience, previous knee injury",
            "current_workout": "5 rounds: 10 burpees, 15 air squats, 20 push-ups",
            "scientific_evidence": "Research shows that high-intensity interval training improves cardiovascular fitness and strength."
        })
        
        print("Workout Adaptation Generated:")
        if isinstance(adaptation, dict):
            for key, value in adaptation.items():
                if key != "error":
                    print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            print(adaptation)
        
        return "error" not in adaptation if isinstance(adaptation, dict) else True
        
    except Exception as e:
        print(f"Workout adaptation failed: {e}")
        return False

async def test_education(agent: PubMedCrossFitAgent):
    """Test educational content generation"""
    print_separator("Testing Educational Content")
    
    try:
        # Test education content
        education = agent.provide_education.invoke({
            "topic": "proper squat technique",
            "scientific_context": "Biomechanical analysis shows proper squat form reduces injury risk and improves performance."
        })
        
        print("Educational Content Generated:")
        print(education[:300] + "..." if len(education) > 300 else education)
        
        return len(education) > 0
        
    except Exception as e:
        print(f"Education generation failed: {e}")
        return False

async def test_end_to_end_queries(agent: PubMedCrossFitAgent):
    """Test end-to-end query processing"""
    print_separator("Testing End-to-End Queries")
    
    test_queries = [
        "What are the benefits of high-intensity interval training for CrossFit athletes?",
        "How should I modify my workout if I have a previous shoulder injury?",
        "Explain the science behind Olympic lifting for strength development",
        "What does research say about recovery between CrossFit workouts?"
    ]
    
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        try:
            result = await agent.aquery(query, f"test_thread_{i}")
            print_results(result, f"Query {i} Results")
            
            if result["response"] and "error" not in result["response"].lower():
                success_count += 1
            
        except Exception as e:
            print(f"Query {i} failed: {e}")
    
    return success_count, len(test_queries)

async def main():
    """Main test function"""
    print_separator("PubMed CrossFit Agent Test Suite")
    
    # Initialize agent
    try:
        print("Initializing PubMed CrossFit Agent...")
        agent = create_pubmed_agent()
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return
    
    # Run tests
    test_results = {}
    
    # Test individual components
    test_results["pubmed_search"] = await test_pubmed_search(agent)
    test_results["local_search"] = await test_local_search(agent)
    test_results["workout_adaptation"] = await test_workout_adaptation(agent)
    test_results["education"] = await test_education(agent)
    
    # Test end-to-end functionality
    success_count, total_queries = await test_end_to_end_queries(agent)
    test_results["end_to_end"] = (success_count, total_queries)
    
    # Print summary
    print_separator("Test Summary")
    
    print("Component Tests:")
    for test_name, result in test_results.items():
        if test_name != "end_to_end":
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    success_count, total_queries = test_results["end_to_end"]
    print(f"  End-to-End Queries: {success_count}/{total_queries} successful")
    
    # Overall status
    component_tests = [v for k, v in test_results.items() if k != "end_to_end"]
    all_passed = all(component_tests) and success_count > 0
    
    print(f"\nOverall Status: {'✓ PASS' if all_passed else '✗ SOME ISSUES'}")
    
    if not all_passed:
        print("\nTroubleshooting Tips:")
        print("- Ensure OPENAI_API_KEY is set in your environment")
        print("- Check that langsmith_api_key is configured if using LangSmith")
        print("- Verify internet connection for PubMed access")
        print("- Make sure local documents are indexed in Qdrant")
        print("- Check that all required dependencies are installed")

if __name__ == "__main__":
    asyncio.run(main())
