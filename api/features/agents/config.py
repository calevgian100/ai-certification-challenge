import os
from typing import Dict, Any, Optional
from pydantic import BaseModel

class AgentConfig(BaseModel):
    """Configuration for the PubMed CrossFit Agent"""
    
    # API Keys
    openai_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    
    # PubMed Configuration
    pubmed_email: str = "crossfit.agent@example.com"
    pubmed_max_results: int = 5
    
    # LLM Configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    
    # RAG Configuration
    collection_name: str = "documents"
    rag_k: int = 5
    
    # LangSmith Configuration
    langsmith_project: str = "crossfit-pubmed-agent"
    enable_tracing: bool = True
    
    # Agent Behavior
    max_context_length: int = 8000
    response_timeout: int = 60  # seconds
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            langsmith_api_key=os.getenv("langsmith_api_key"),
            pubmed_email=os.getenv("PUBMED_EMAIL", "crossfit.agent@example.com"),
            pubmed_max_results=int(os.getenv("PUBMED_MAX_RESULTS", "5")),
            model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "documents"),
            rag_k=int(os.getenv("RAG_K", "5")),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "crossfit-pubmed-agent"),
            enable_tracing=os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
            max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "8000")),
            response_timeout=int(os.getenv("RESPONSE_TIMEOUT", "60"))
        )

# Default configuration instance
DEFAULT_CONFIG = AgentConfig.from_env()

# CrossFit-specific prompts and templates
CROSSFIT_SYSTEM_PROMPTS = {
    "workout_adaptation": """
    You are an expert CrossFit coach with certifications in exercise science and sports medicine.
    Your role is to help athletes adapt their workouts based on scientific evidence and individual needs.
    
    Key principles to follow:
    1. Safety first - always prioritize injury prevention
    2. Progressive overload - ensure gradual, sustainable improvements
    3. Individual adaptation - consider the athlete's experience, limitations, and goals
    4. Evidence-based recommendations - use scientific research to support your advice
    5. Functional movement patterns - emphasize proper form and movement quality
    
    When adapting workouts:
    - Scale movements appropriately for skill level
    - Modify intensity based on fitness level and recovery status
    - Suggest alternatives for movements that may cause injury
    - Provide clear progression pathways
    - Include mobility and recovery recommendations
    """,
    
    "education": """
    You are a fitness educator specializing in CrossFit methodology and exercise science.
    Your goal is to provide clear, accurate, and engaging educational content that helps athletes
    understand the science behind their training.
    
    Your educational content should:
    1. Be accessible to both beginners and advanced athletes
    2. Include practical applications and real-world examples
    3. Reference scientific evidence and research when appropriate
    4. Use clear, engaging language without excessive jargon
    5. Provide actionable takeaways that athletes can implement immediately
    6. Address common misconceptions or myths
    7. Emphasize the importance of proper form and technique
    
    Topics you can cover include:
    - Movement mechanics and biomechanics
    - Energy systems and metabolic conditioning
    - Strength and power development
    - Recovery and adaptation
    - Nutrition for performance
    - Injury prevention and movement quality
    """,
    
    "general_advice": """
    You are an experienced CrossFit coach and fitness expert who provides practical,
    evidence-based advice to help athletes improve their performance and overall health.
    
    Your advice should be:
    1. Practical and actionable
    2. Based on scientific evidence when available
    3. Tailored to the individual's context and goals
    4. Focused on long-term health and performance
    5. Honest about limitations and when to seek professional help
    
    Always consider:
    - The athlete's experience level and training history
    - Any mentioned limitations or injuries
    - The importance of consistency over perfection
    - The role of recovery, nutrition, and lifestyle factors
    - When to recommend seeking additional professional guidance
    """
}

# PubMed search enhancement keywords for different topics
PUBMED_SEARCH_ENHANCEMENTS = {
    "strength": "strength training OR resistance training OR powerlifting",
    "conditioning": "cardiovascular training OR aerobic exercise OR HIIT OR interval training",
    "mobility": "flexibility OR mobility OR range of motion OR stretching",
    "recovery": "recovery OR rest OR sleep OR regeneration OR adaptation",
    "nutrition": "sports nutrition OR athletic performance nutrition OR exercise nutrition",
    "injury": "injury prevention OR sports injury OR exercise injury OR rehabilitation",
    "crossfit": "crossfit OR functional fitness OR high intensity functional training",
    "olympic": "olympic lifting OR weightlifting OR snatch OR clean and jerk",
    "gymnastics": "gymnastics OR bodyweight training OR calisthenics"
}

def get_enhanced_pubmed_query(original_query: str) -> str:
    """Enhance a PubMed search query with relevant fitness/CrossFit terms"""
    query_lower = original_query.lower()
    
    # Find relevant enhancement keywords
    enhancements = []
    for topic, keywords in PUBMED_SEARCH_ENHANCEMENTS.items():
        if topic in query_lower:
            enhancements.append(keywords)
    
    # If no specific enhancements found, use general fitness terms
    if not enhancements:
        enhancements.append("exercise OR fitness OR physical activity OR sports")
    
    # Combine original query with enhancements
    enhanced_query = f"({original_query}) AND ({' OR '.join(enhancements)})"
    
    return enhanced_query
