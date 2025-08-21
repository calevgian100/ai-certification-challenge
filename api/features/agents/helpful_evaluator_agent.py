"""A helpful evaluator agent that works with the PubMed agent.

This agent evaluates responses from the PubMed agent for helpfulness.
If the response is not helpful, it routes back to the main agent for refinement.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import PubMed agent
from api.features.agents.pub_med_agent import PubMedCrossFitAgent, create_pubmed_agent


class EvaluatorAgentState(TypedDict):
    """State for the Helpful Evaluator Agent workflow"""
    messages: Annotated[List, "Messages in the conversation"]
    query: str
    pubmed_response: str
    helpfulness_score: Annotated[str, "Y if helpful, N if not"]
    attempt_count: int
    max_attempts: int
    final_response: str


class HelpfulEvaluatorAgent:
    """An agent that evaluates responses for helpfulness and routes accordingly"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 max_attempts: int = 3):
        
        # Set up API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.openai_api_key
        )
        
        # Initialize PubMed agent
        self.pubmed_agent = create_pubmed_agent()
        
        # Set maximum attempts for refinement
        self.max_attempts = max_attempts
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(EvaluatorAgentState)
        
        # Add nodes
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("evaluate_helpfulness", self._evaluate_helpfulness)
        workflow.add_node("refine_response", self._refine_response)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # Define the flow
        workflow.set_entry_point("process_query")
        
        # From process_query, always go to evaluate_helpfulness
        workflow.add_edge("process_query", "evaluate_helpfulness")
        
        # From evaluate_helpfulness, conditionally route based on helpfulness
        workflow.add_conditional_edges(
            "evaluate_helpfulness",
            self._route_based_on_helpfulness,
            {
                "helpful": "finalize_response",
                "not_helpful": "refine_response",
                "max_attempts": "finalize_response"
            }
        )
        
        # From refine_response, go back to evaluate_helpfulness
        workflow.add_edge("refine_response", "evaluate_helpfulness")
        
        # From finalize_response, end the workflow
        workflow.add_edge("finalize_response", END)
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _process_query(self, state: EvaluatorAgentState) -> EvaluatorAgentState:
        """Process the initial query using the PubMed agent"""
        query = state["query"]
        
        # Call PubMed agent
        result = self.pubmed_agent.query(query)
        pubmed_response = result["response"]
        
        # Update state
        state["pubmed_response"] = pubmed_response
        state["attempt_count"] = 1
        
        # Add to messages
        if "messages" not in state:
            state["messages"] = []
        
        state["messages"].append(HumanMessage(content=query))
        state["messages"].append(AIMessage(content=pubmed_response))
        
        return state
    
    def _evaluate_helpfulness(self, state: EvaluatorAgentState) -> EvaluatorAgentState:
        """Evaluate if the response is helpful"""
        query = state["query"]
        response = state["pubmed_response"]
        
        # Create prompt for helpfulness evaluation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpfulness evaluator for CrossFit and fitness information.
            Your job is to determine if the provided response adequately answers the user's query.
            
            Evaluate based on:
            1. Relevance to the query
            2. Accuracy of information
            3. Completeness of the answer
            4. Actionability of advice (if applicable)
            5. Clarity and understandability
            
            Respond with ONLY "Y" if the response is helpful, or "N" if it is not helpful.
            """),
            ("human", f"""
            User Query: {query}
            
            Response to Evaluate:
            {response}
            
            Is this response helpful? (Y/N)
            """)
        ])
        
        # Get evaluation
        evaluation = self.llm.invoke(prompt.format_messages())
        helpfulness_score = evaluation.content.strip().upper()
        
        # Normalize to Y or N
        if "Y" in helpfulness_score:
            helpfulness_score = "Y"
        else:
            helpfulness_score = "N"
        
        # Update state
        state["helpfulness_score"] = helpfulness_score
        
        return state
    
    def _route_based_on_helpfulness(self, state: EvaluatorAgentState) -> str:
        """Determine next step based on helpfulness score and attempt count"""
        helpfulness_score = state["helpfulness_score"]
        attempt_count = state["attempt_count"]
        max_attempts = state["max_attempts"]
        
        # If helpful, route to finalize
        if helpfulness_score == "Y":
            return "helpful"
        
        # If max attempts reached, route to finalize
        if attempt_count >= max_attempts:
            return "max_attempts"
        
        # Otherwise, route to refine
        return "not_helpful"
    
    def _refine_response(self, state: EvaluatorAgentState) -> EvaluatorAgentState:
        """Refine the response based on the original query"""
        query = state["query"]
        previous_response = state["pubmed_response"]
        attempt_count = state["attempt_count"]
        
        # Create a refined query
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at refining queries to get better responses.
            Your job is to analyze the original query and previous response,
            then create an improved query that will yield a more helpful response.
            
            Focus on:
            1. Adding specificity to ambiguous parts
            2. Requesting more actionable information
            3. Asking for missing details
            4. Clarifying any misunderstandings
            
            Provide ONLY the refined query, nothing else.
            """),
            ("human", f"""
            Original Query: {query}
            
            Previous Response (deemed not helpful):
            {previous_response}
            
            Please provide a refined query:
            """)
        ])
        
        # Get refined query
        refined_query_response = self.llm.invoke(refine_prompt.format_messages())
        refined_query = refined_query_response.content.strip()
        
        # Call PubMed agent with refined query
        result = self.pubmed_agent.query(refined_query)
        new_response = result["response"]
        
        # Update state
        state["pubmed_response"] = new_response
        state["attempt_count"] = attempt_count + 1
        
        # Add to messages
        state["messages"].append(SystemMessage(content=f"Refining query (attempt {attempt_count + 1}/{state['max_attempts']}): {refined_query}"))
        state["messages"].append(AIMessage(content=new_response))
        
        return state
    
    def _finalize_response(self, state: EvaluatorAgentState) -> EvaluatorAgentState:
        """Finalize the response to be returned to the user"""
        helpfulness_score = state["helpfulness_score"]
        pubmed_response = state["pubmed_response"]
        attempt_count = state["attempt_count"]
        max_attempts = state["max_attempts"]
        
        # If response is helpful, use it directly
        if helpfulness_score == "Y":
            final_response = pubmed_response
        else:
            # If max attempts reached, add a note about limitations
            final_response = f"""
            {pubmed_response}
            
            Note: I've made {attempt_count} attempts to provide the most helpful response possible.
            If you need more specific information, please consider refining your question.
            """
        
        # Update state
        state["final_response"] = final_response
        
        return state
    
    def query(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Main entry point for querying the agent"""
        try:
            # Create initial state
            initial_state = {
                "query": user_query,
                "messages": [],
                "pubmed_response": "",
                "helpfulness_score": "",
                "attempt_count": 0,
                "max_attempts": self.max_attempts,
                "final_response": ""
            }
            
            # Run the graph
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract the response
            response_content = final_state["final_response"]
            
            return {
                "response": response_content,
                "messages": final_state["messages"],
                "helpfulness_score": final_state["helpfulness_score"],
                "attempts": final_state["attempt_count"]
            }
            
        except Exception as e:
            print(f"Agent query error: {e}")
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "messages": [],
                "helpfulness_score": "N",
                "attempts": 0
            }


# Convenience function for easy import
def create_helpful_evaluator_agent(**kwargs):
    """Create a Helpful Evaluator Agent instance"""
    return HelpfulEvaluatorAgent(**kwargs)
