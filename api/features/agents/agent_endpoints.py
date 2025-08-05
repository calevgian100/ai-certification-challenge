from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import logging
from langsmith import traceable

from api.features.agents.pub_med_agent import PubMedCrossFitAgent, create_pubmed_agent
from api.features.observability.langsmith_config import LangSmithConfig
from api.features.agents.config import AgentConfig
from api.utils.env_loader import load_env_vars

# Load environment variables at module level to ensure LangSmith tracing works
load_env_vars()
import os
langsmith_api_key = os.getenv("langsmith_api_key")
if langsmith_api_key:
    # Set LangSmith environment variables globally for @traceable decorators
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "crossfit-pubmed-agent"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/agents", tags=["agents"])

# Global agent instance (singleton pattern for efficiency)
_agent_instance: Optional[PubMedCrossFitAgent] = None

def get_agent() -> PubMedCrossFitAgent:
    """Get or create the PubMed agent instance"""
    global _agent_instance
    if _agent_instance is None:
        # Load environment variables from env.yaml first
        load_env_vars()
        
        # Create config after environment variables are loaded
        config = AgentConfig.from_env()
        langsmith_api_key = config.langsmith_api_key
        if langsmith_api_key:
            langsmith_config = LangSmithConfig(
                api_key=langsmith_api_key,
                project_name="crossfit-pubmed-agent"
            )
            logger.info(f"✅ LangSmith initialized for FastAPI endpoints: {langsmith_config.project_name}")
            

        else:
            logger.warning("⚠️ LangSmith API key not found - tracing will be disabled")
        
        _agent_instance = create_pubmed_agent(langsmith_api_key=langsmith_api_key)
    return _agent_instance

# Request/Response models
class AgentQueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default"
    user_profile: Optional[str] = None
    current_workout: Optional[str] = None

class AgentQueryResponse(BaseModel):
    response: str
    pubmed_sources: List[Dict[str, Any]]
    local_sources: List[Dict[str, Any]]
    context_used: str
    thread_id: str

class WorkoutAdaptationRequest(BaseModel):
    user_profile: str
    current_workout: str
    query: Optional[str] = None

class EducationRequest(BaseModel):
    topic: str
    query: Optional[str] = None

@router.post("/query", response_model=AgentQueryResponse)
@traceable(name="frontend_agent_query")
async def query_agent(request: AgentQueryRequest):
    """
    Query the PubMed CrossFit Agent with a general question.
    
    The agent will:
    1. Search PubMed for relevant scientific articles
    2. Search local documents and Qdrant vector store
    3. Synthesize information from all sources
    4. Provide an evidence-based response
    """
    try:
        agent = get_agent()
        
        # Build enhanced query if user profile or workout is provided
        enhanced_query = request.query
        if request.user_profile:
            enhanced_query += f" (User profile: {request.user_profile})"
        if request.current_workout:
            enhanced_query += f" (Current workout: {request.current_workout})"
        
        # Query the agent (using sync method in async context)
        result = await asyncio.to_thread(agent.query, enhanced_query, request.thread_id)
        
        return AgentQueryResponse(
            response=result["response"],
            pubmed_sources=result["pubmed_sources"],
            local_sources=result["local_sources"],
            context_used=result["context_used"],
            thread_id=request.thread_id
        )
        
    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process agent query: {str(e)}"
        )

@router.post("/workout-adaptation")
@traceable(name="frontend_workout_adaptation")
async def adapt_workout(request: WorkoutAdaptationRequest):
    """
    Get personalized workout adaptations based on user profile and scientific evidence.
    """
    try:
        agent = get_agent()
        
        # First, gather scientific evidence related to the workout
        search_query = request.query or f"workout adaptation {request.current_workout}"
        evidence_result = await agent.aquery(search_query, "workout_adaptation")
        
        # Use the agent's workout adaptation tool
        adaptation = agent.generate_workout_adaptation(
            user_profile=request.user_profile,
            current_workout=request.current_workout,
            scientific_evidence=evidence_result["context_used"]
        )
        
        return {
            "adaptation": adaptation,
            "scientific_evidence": evidence_result["pubmed_sources"],
            "local_sources": evidence_result["local_sources"]
        }
        
    except Exception as e:
        logger.error(f"Workout adaptation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate workout adaptation: {str(e)}"
        )

@router.post("/education")
@traceable(name="frontend_education")
async def get_education(request: EducationRequest):
    """
    Get educational content about CrossFit, exercise science, or fitness topics.
    """
    try:
        agent = get_agent()
        
        # Gather scientific context for the topic
        search_query = request.query or f"education {request.topic}"
        context_result = await agent.aquery(search_query, "education")
        
        # Generate educational content
        education_content = agent.provide_education(
            topic=request.topic,
            scientific_context=context_result["context_used"]
        )
        
        return {
            "educational_content": education_content,
            "scientific_sources": context_result["pubmed_sources"],
            "local_sources": context_result["local_sources"],
            "topic": request.topic
        }
        
    except Exception as e:
        logger.error(f"Education generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate educational content: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for the agent service"""
    try:
        agent = get_agent()
        return {
            "status": "healthy",
            "agent_initialized": agent is not None,
            "langsmith_enabled": "LANGCHAIN_TRACING_V2" in agent.llm.model_kwargs.get("extra_headers", {}) if hasattr(agent.llm, 'model_kwargs') else False
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/reset-conversation")
async def reset_conversation(thread_id: str = "default"):
    """Reset the conversation memory for a specific thread"""
    try:
        # Note: With MemorySaver, we'd need to implement a way to clear specific thread memory
        # For now, we'll recreate the agent instance to clear all memory
        global _agent_instance
        _agent_instance = None
        
        return {
            "message": f"Conversation memory reset for thread: {thread_id}",
            "thread_id": thread_id
        }
        
    except Exception as e:
        logger.error(f"Conversation reset error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset conversation: {str(e)}"
        )
