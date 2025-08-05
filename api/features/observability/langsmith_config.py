"""
LangSmith Configuration and Utilities for Enhanced RAG Agent
Provides comprehensive tracing and observability for the CrossFit agent system
"""

import os
from typing import Dict, Any, Optional
from langsmith import Client, traceable
from datetime import datetime

class LangSmithConfig:
    """Configuration and utilities for LangSmith integration"""
    
    def __init__(self, api_key: Optional[str] = None, project_name: str = "crossfit-pubmed-agent"):
        """
        Initialize LangSmith configuration
        
        Args:
            api_key: LangSmith API key (if not provided, will use environment variable)
            project_name: Name of the LangSmith project
        """
        self.api_key = api_key or os.getenv("langsmith_api_key")
        self.project_name = project_name
        self.client = None
        
        if self.api_key:
            self.setup_langsmith()
    
    def setup_langsmith(self) -> bool:
        """
        Set up LangSmith environment variables and client
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Set environment variables for LangChain tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            
            # Initialize LangSmith client
            self.client = Client(api_key=self.api_key)
            
            print(f"✅ LangSmith tracing enabled for project: {self.project_name}")
            print(f"🔗 View traces at: https://smith.langchain.com/projects/{self.project_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup LangSmith: {e}")
            return False
    
    @traceable(name="log_enhanced_rag_metrics")
    def log_rag_metrics(self, 
                       query: str, 
                       num_chunks_retrieved: int,
                       retriever_type: str,
                       response_time_ms: float,
                       sources: list,
                       metadata: Dict[str, Any] = None) -> None:
        """
        Log enhanced RAG metrics to LangSmith
        
        Args:
            query: User query
            num_chunks_retrieved: Number of chunks retrieved
            retriever_type: Type of retriever used
            response_time_ms: Response time in milliseconds
            sources: List of source documents
            metadata: Additional metadata
        """
        if not self.client:
            return
        
        try:
            metrics = {
                "query": query,
                "num_chunks_retrieved": num_chunks_retrieved,
                "retriever_type": retriever_type,
                "response_time_ms": response_time_ms,
                "num_sources": len(sources),
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Add source information
            if sources:
                metrics["source_preview"] = sources[0].get("text", "")[:200] + "..."
                metrics["source_files"] = list(set([s.get("source", "Unknown") for s in sources]))
            
            print(f"📊 Logged RAG metrics to LangSmith: {retriever_type} retriever, {num_chunks_retrieved} chunks")
            
        except Exception as e:
            print(f"⚠️  Failed to log RAG metrics: {e}")
    
    @traceable(name="log_agent_interaction")
    def log_agent_interaction(self,
                            user_query: str,
                            pubmed_results: int,
                            local_results: int,
                            final_response: str,
                            execution_time_ms: float,
                            metadata: Dict[str, Any] = None) -> None:
        """
        Log complete agent interaction to LangSmith
        
        Args:
            user_query: Original user query
            pubmed_results: Number of PubMed results found
            local_results: Number of local document results found
            final_response: Final agent response
            execution_time_ms: Total execution time
            metadata: Additional metadata
        """
        if not self.client:
            return
        
        try:
            interaction = {
                "user_query": user_query,
                "pubmed_results_count": pubmed_results,
                "local_results_count": local_results,
                "response_length": len(final_response),
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            print(f"🤖 Logged agent interaction to LangSmith: {pubmed_results} PubMed + {local_results} local results")
            
        except Exception as e:
            print(f"⚠️  Failed to log agent interaction: {e}")
    
    def create_session(self, session_name: str = None) -> str:
        """
        Create a new LangSmith session for grouping related traces
        
        Args:
            session_name: Optional session name
            
        Returns:
            str: Session ID
        """
        if not self.client:
            return "no-session"
        
        try:
            session_name = session_name or f"crossfit-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            session = self.client.create_session(session_name)
            print(f"🆔 Created LangSmith session: {session_name}")
            return session.id
        except Exception as e:
            print(f"⚠️  Failed to create session: {e}")
            return "no-session"

# Global LangSmith configuration instance
langsmith_config = LangSmithConfig()

# Convenience decorators for common tracing patterns
def trace_rag_query(name: str = "rag_query"):
    """Decorator for tracing RAG queries"""
    return traceable(name=name)

def trace_agent_step(name: str = "agent_step"):
    """Decorator for tracing agent steps"""
    return traceable(name=name)

def trace_retrieval(name: str = "document_retrieval"):
    """Decorator for tracing document retrieval"""
    return traceable(name=name)
