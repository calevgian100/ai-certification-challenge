"""
Simple test for the helpfulness evaluation functionality
"""
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the agent_with_helpfulness module
from agent_with_helpfulness import HelpfulnessEvaluator

def test_helpfulness_evaluation():
    """Test the helpfulness evaluation functionality"""
    print("\n🧪 Testing Helpfulness Evaluation...\n")
    
    # Create the evaluator
    evaluator = HelpfulnessEvaluator()
    
    # Test queries and responses
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
        
        # Evaluate helpfulness
        result = evaluator.evaluate_helpfulness(query, response)
        is_helpful = result.get("is_helpful", False)
        score = result.get("score", "N/A")
        feedback = result.get("feedback", "No feedback provided")
        
        # Print results
        print(f"✅ Helpful: {'Yes' if is_helpful else 'No'}")
        print(f"📊 Score: {score}")
        print(f"💬 Feedback: {feedback}")
        
        print("=" * 80)

if __name__ == "__main__":
    test_helpfulness_evaluation()
