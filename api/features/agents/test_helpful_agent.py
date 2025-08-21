"""Test script for the Helpful Evaluator Agent.

This script tests the functionality of the Helpful Evaluator Agent
by running a series of test queries and validating the responses.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the agent
from api.features.agents.helpful_evaluator_agent import HelpfulEvaluatorAgent, create_helpful_evaluator_agent


class TestHelpfulEvaluatorAgent(unittest.TestCase):
    """Test cases for the Helpful Evaluator Agent"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock PubMed agent
        self.mock_pubmed_agent = MagicMock()
        self.mock_pubmed_agent.query.return_value = {
            "response": "This is a test response from PubMed agent.",
            "pubmed_sources": [],
            "local_sources": [],
            "context_used": ""
        }
        
        # Create a mock LLM
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = MagicMock(content="Y")
        
        # Create agent with mocks
        self.agent = HelpfulEvaluatorAgent(max_attempts=2)
        self.agent.pubmed_agent = self.mock_pubmed_agent
        self.agent.llm = self.mock_llm
    
    def test_helpful_response(self):
        """Test when the response is immediately helpful"""
        # Set up mock to return "Y" for helpfulness
        self.mock_llm.invoke.return_value = MagicMock(content="Y")
        
        # Query the agent
        result = self.agent.query("Test query")
        
        # Verify results
        self.assertEqual(result["helpfulness_score"], "Y")
        self.assertEqual(result["attempts"], 1)
        self.assertIn("This is a test response", result["response"])
        
        # Verify PubMed agent was called once
        self.mock_pubmed_agent.query.assert_called_once()
    
    def test_unhelpful_response_refinement(self):
        """Test when the response is initially unhelpful but gets refined"""
        # Set up mocks to return "N" then "Y" for helpfulness
        self.mock_llm.invoke.side_effect = [
            MagicMock(content="N"),  # First evaluation - not helpful
            MagicMock(content="Refined query"),  # Refinement
            MagicMock(content="Y")   # Second evaluation - helpful
        ]
        
        # Set up PubMed agent to return different responses
        self.mock_pubmed_agent.query.side_effect = [
            {"response": "Initial unhelpful response"},
            {"response": "Refined helpful response"}
        ]
        
        # Query the agent
        result = self.agent.query("Test query")
        
        # Verify results
        self.assertEqual(result["helpfulness_score"], "Y")
        self.assertEqual(result["attempts"], 2)
        self.assertIn("Refined helpful response", result["response"])
        
        # Verify PubMed agent was called twice
        self.assertEqual(self.mock_pubmed_agent.query.call_count, 2)
    
    def test_max_attempts_reached(self):
        """Test when max attempts are reached without a helpful response"""
        # Set up mocks to always return "N" for helpfulness
        self.mock_llm.invoke.side_effect = [
            MagicMock(content="N"),  # First evaluation - not helpful
            MagicMock(content="Refined query 1"),  # First refinement
            MagicMock(content="N"),  # Second evaluation - not helpful
            MagicMock(content="Refined query 2"),  # Second refinement (not used due to max_attempts=2)
        ]
        
        # Set up PubMed agent to return different responses
        self.mock_pubmed_agent.query.side_effect = [
            {"response": "Initial unhelpful response"},
            {"response": "Still unhelpful response"}
        ]
        
        # Query the agent
        result = self.agent.query("Test query")
        
        # Verify results
        self.assertEqual(result["helpfulness_score"], "N")
        self.assertEqual(result["attempts"], 2)
        self.assertIn("Still unhelpful response", result["response"])
        self.assertIn("attempts to provide", result["response"])
        
        # Verify PubMed agent was called twice
        self.assertEqual(self.mock_pubmed_agent.query.call_count, 2)


if __name__ == "__main__":
    unittest.main()
