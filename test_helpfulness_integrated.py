"""
Direct test for the PubMed agent's helpfulness evaluation functionality
"""
import os
import sys

# Import the specific methods we need to test
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api.features.agents.pub_med_agent import PubMedCrossFitAgent

def test_evaluate_helpfulness():
    """Test the _evaluate_helpfulness method directly"""
    print("\n🧪 Testing PubMed Agent's Helpfulness Evaluation...\n")
    
    # Create the agent with minimal setup
    agent = PubMedCrossFitAgent(entrez_email="test@example.com")
    
    # Test cases
    test_cases = [
        {
            "query": "What are the benefits of CrossFit for older adults?",
            "response": "CrossFit can help older adults improve strength, balance, and bone density. It also promotes functional fitness which helps with everyday activities."
        },
        {
            "query": "How should I modify burpees if I have knee pain?",
            "response": "Burpees are a challenging exercise."  # Intentionally unhelpful
        },
        {
            "query": "Tell me about the history of CrossFit games",
            "response": "The CrossFit Games started in 2007 at a ranch in Aromas, California. They've grown from a small competition to a global event with hundreds of thousands of participants in the Open."
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        response = test_case["response"]
        
        print(f"\n\n📝 Test Case {i+1}:")
        print(f"Query: {query}")
        print(f"Response: {response}")
        print("-" * 80)
        
        # Create a minimal state for testing
        state = {
            "query": query,
            "response": response,
            "attempt_count": 1,
            "max_attempts": 3
        }
        
        # Call the evaluation method directly
        try:
            result_state = agent._evaluate_helpfulness(state)
            helpfulness_score = result_state.get("helpfulness_score", "N/A")
            
            # Print results
            print(f"✅ Helpfulness Score: {helpfulness_score}")
            print(f"🔄 Attempt Count: {result_state.get('attempt_count', 1)}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("=" * 80)

def test_route_based_on_helpfulness():
    """Test the _route_based_on_helpfulness method directly"""
    print("\n🧪 Testing Helpfulness Routing Logic...\n")
    
    # Create the agent with minimal setup
    agent = PubMedCrossFitAgent(entrez_email="test@example.com")
    
    # Test cases for routing
    routing_cases = [
        {"helpfulness_score": "Y", "attempt_count": 1, "max_attempts": 3, "expected": "end"},
        {"helpfulness_score": "N", "attempt_count": 1, "max_attempts": 3, "expected": "refine"},
        {"helpfulness_score": "N", "attempt_count": 3, "max_attempts": 3, "expected": "max_attempts_reached"}
    ]
    
    for i, case in enumerate(routing_cases):
        print(f"\n📝 Routing Test {i+1}:")
        print(f"Helpfulness Score: {case['helpfulness_score']}")
        print(f"Attempt Count: {case['attempt_count']}/{case['max_attempts']}")
        print(f"Expected Route: {case['expected']}")
        print("-" * 80)
        
        # Create a minimal state for testing
        state = {
            "query": "Test query",
            "response": "Test response",
            "helpfulness_score": case["helpfulness_score"],
            "attempt_count": case["attempt_count"],
            "max_attempts": case["max_attempts"]
        }
        
        # Call the routing method directly
        try:
            route = agent._route_based_on_helpfulness(state)
            print(f"✅ Actual Route: {route}")
            print(f"✅ Match Expected: {'Yes' if route == case['expected'] else 'No'}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("=" * 80)

if __name__ == "__main__":
    test_evaluate_helpfulness()
    test_route_based_on_helpfulness()
