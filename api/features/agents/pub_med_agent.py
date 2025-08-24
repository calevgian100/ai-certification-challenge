import os
import json
import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import hashlib
import time

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

# Import LangSmith configuration
from api.features.observability.langsmith_config import LangSmithConfig, trace_agent_step

# Local imports
from api.features.rag.rag import RAGQueryEngine
from api.features.rag.enhanced_rag import EnhancedRAGQueryEngine
from api.features.store.vector_store import QdrantVectorStore
from api.features.processors.local_document_manager import LocalDocumentManager

# PubMed and Bio imports
import requests
from Bio import Entrez
from xml.etree import ElementTree as ET


class AgentState(TypedDict):
    """State for the PubMed CrossFit Agent"""
    messages: Annotated[List, add_messages]
    query: str
    context: str
    pubmed_results: List[Dict[str, Any]]
    local_rag_results: List[Dict[str, Any]]
    workout_plan: Optional[Dict[str, Any]]
    educational_content: Optional[str]
    next_action: str
    response: str
    helpfulness_score: str
    attempt_count: int
    max_attempts: int
    context_quality_score: float
    pubmed_quality_score: float
    local_quality_score: float
    local_confidence: float
    cache_key: str
    cache_hit: bool


class PubMedCrossFitAgent:
    """An agentic RAG system that leverages PubMed and local documents for CrossFit guidance"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 langsmith_api_key: str = None,
                 email: str = "user@example.com",
                 max_attempts: int = 3):
        
        # Set up API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.langsmith_api_key = langsmith_api_key or os.getenv("langsmith_api_key")
        
        # Set maximum attempts for helpfulness refinement
        self.max_attempts = max_attempts
        
        # Initialize LangSmith configuration
        self.langsmith = LangSmithConfig(
            api_key=self.langsmith_api_key,
            project_name="crossfit-pubmed-agent"
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=self.openai_api_key
        )
        
        # Initialize RAG components with enhanced retrieval
        self.rag_engine = RAGQueryEngine()  # Keep original for backward compatibility
        self.enhanced_rag_engine = EnhancedRAGQueryEngine(
            retriever_type="ensemble",  # Best results from RAGAS evaluation
            chunk_size=1000,
            chunk_overlap=200,
            k=5
        )
        self.local_doc_manager = LocalDocumentManager()
        
        # Flag to track if enhanced RAG is initialized
        self.enhanced_rag_initialized = False
        
        # Set up PubMed
        Entrez.email = email
        
        # Initialize cache
        self._cache = {}
        
        # Build the graph
        self.graph = self._build_graph()
    
    @traceable(name="search_pubmed")
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search PubMed for scientific articles related to CrossFit, exercise, and fitness"""
        try:
            # Enhance query with CrossFit/fitness context
            enhanced_query = f"({query}) AND (crossfit OR exercise OR fitness OR strength training OR conditioning)"
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=enhanced_query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            if not search_results["IdList"]:
                return []
            
            # Fetch article details
            ids = ",".join(search_results["IdList"])
            handle = Entrez.efetch(
                db="pubmed",
                id=ids,
                rettype="abstract",
                retmode="xml"
            )
            
            records = Entrez.read(handle)
            handle.close()
            
            articles = []
            for record in records["PubmedArticle"]:
                try:
                    article = record["MedlineCitation"]["Article"]
                    
                    # Extract title
                    title = str(article.get("ArticleTitle", "No title"))
                    
                    # Extract abstract
                    abstract = ""
                    if "Abstract" in article and "AbstractText" in article["Abstract"]:
                        abstract_parts = article["Abstract"]["AbstractText"]
                        if isinstance(abstract_parts, list):
                            abstract = " ".join([str(part) for part in abstract_parts])
                        else:
                            abstract = str(abstract_parts)
                    
                    # Extract authors
                    authors = []
                    if "AuthorList" in article:
                        for author in article["AuthorList"][:3]:  # First 3 authors
                            if "LastName" in author and "ForeName" in author:
                                authors.append(f"{author['ForeName']} {author['LastName']}")
                    
                    # Extract publication year
                    pub_year = "Unknown"
                    if "Journal" in article and "JournalIssue" in article["Journal"]:
                        if "PubDate" in article["Journal"]["JournalIssue"]:
                            pub_date = article["Journal"]["JournalIssue"]["PubDate"]
                            if "Year" in pub_date:
                                pub_year = str(pub_date["Year"])
                    
                    articles.append({
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": pub_year,
                        "pmid": record["MedlineCitation"]["PMID"],
                        "source": "PubMed"
                    })
                    
                except Exception as e:
                    continue
            
            return articles
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def initialize_enhanced_rag(self, data_path: str = "api/data") -> bool:
        """Initialize enhanced RAG system with documents from specified path"""
        try:
            print(f"🚀 Initializing Enhanced RAG system with path: {data_path}")
            
            # Check if path exists
            if not os.path.exists(data_path):
                print(f"❌ Data path does not exist: {data_path}")
                return False
            
            # Load documents
            success = self.enhanced_rag_engine.load_documents_from_directory(data_path)
            
            if success:
                self.enhanced_rag_initialized = True
                print("✅ Enhanced RAG system initialized successfully!")
                info = self.enhanced_rag_engine.get_retriever_info()
                print(f"📊 Retriever info: {info}")
                
                # Test that retriever is working
                if self.enhanced_rag_engine.retriever is None:
                    print("⚠️  Warning: Retriever is None after initialization")
                    return False
                else:
                    print(f"✅ Retriever type: {type(self.enhanced_rag_engine.retriever).__name__}")
                    
            else:
                print("❌ Failed to initialize Enhanced RAG system")
                
            return success
            
        except Exception as e:
            print(f"❌ Error initializing Enhanced RAG: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @traceable(name="search_local_documents")
    def search_local_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search local documents using enhanced RAG if available, fallback to original RAG"""
        try:
            # Use enhanced RAG if initialized, otherwise fallback to original
            if self.enhanced_rag_initialized:
                print("🔍 Using Enhanced RAG for local document search...")
                print(f"🔍 Query: {query}")
                print(f"🔍 Retriever status: {self.enhanced_rag_engine.retriever is not None}")
                print(f"🔍 Number of docs: {len(self.enhanced_rag_engine.docs)}")
                
                results = self.enhanced_rag_engine.query(query)
                print(f"🔍 Enhanced RAG returned: {len(results.get('sources', []))} sources")
            else:
                print("🔍 Using original RAG for local document search...")
                results = self.rag_engine.query(query)
            
            if results.get("sources"):
                formatted_sources = []
                for source in results["sources"]:
                    try:
                        formatted_source = {
                            "content": source.get("text", ""),
                            "source": source.get("source", "Local Document"),
                            "relevance_score": source.get("score", 0.0)
                        }
                        formatted_sources.append(formatted_source)
                    except Exception as source_error:
                        print(f"Error processing source: {source_error}")
                        # Add a fallback source entry
                        formatted_sources.append({
                            "content": str(source) if source else "",
                            "source": "Local Document",
                            "relevance_score": 0.0
                        })
                
                return formatted_sources
            
            return []
            
        except Exception as e:
            print(f"Local document search error: {e}")
            # Return empty list instead of failing
            return []
    
    def generate_workout_adaptation(self, 
                                  user_profile: str, 
                                  current_workout: str, 
                                  scientific_evidence: str) -> Dict[str, Any]:
        """Generate personalized workout adaptations based on scientific evidence"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are an expert CrossFit coach with deep knowledge of exercise science.
                Based on the user profile, current workout, and scientific evidence provided,
                create a personalized workout adaptation.
                
                Focus on:
                1. Safety considerations
                2. Progressive overload principles
                3. Individual limitations and goals
                4. Evidence-based modifications
                
                Provide your response in JSON format with the following structure:
                {
                    "adapted_workout": "detailed workout plan",
                    "modifications": ["list of specific modifications"],
                    "safety_notes": ["important safety considerations"],
                    "progression_plan": "how to progress over time",
                    "scientific_rationale": "evidence-based reasoning"
                }
                """),
                ("human", """
                User Profile: {user_profile}
                Current Workout: {current_workout}
                Scientific Evidence: {scientific_evidence}
                
                Please provide a personalized workout adaptation.
                """)
            ])
            
            response = self.llm.invoke(
                prompt.format_messages(
                    user_profile=user_profile,
                    current_workout=current_workout,
                    scientific_evidence=scientific_evidence
                )
            )
            
            # Try to parse JSON response
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {
                    "adapted_workout": response.content,
                    "modifications": [],
                    "safety_notes": [],
                    "progression_plan": "",
                    "scientific_rationale": ""
                }
                
        except Exception as e:
            print(f"Workout adaptation error: {e}")
            return {
                "adapted_workout": "Unable to generate adaptation at this time.",
                "error": str(e)
            }
    
    def provide_education(self, topic: str, scientific_context: str) -> str:
        """Provide educational content about CrossFit, exercise science, or fitness topics"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are an expert fitness educator specializing in CrossFit and exercise science.
                Provide clear, accurate, and engaging educational content based on scientific evidence.
                
                Your response should:
                1. Be accessible to both beginners and advanced athletes
                2. Include practical applications
                3. Reference scientific evidence when appropriate
                4. Use clear, engaging language
                5. Include actionable takeaways
                """),
                ("human", """
                Topic: {topic}
                Scientific Context: {scientific_context}
                
                Please provide comprehensive educational content on this topic.
                """)
            ])
            
            response = self.llm.invoke(
                prompt.format_messages(
                    topic=topic,
                    scientific_context=scientific_context
                )
            )
            
            return response.content
            
        except Exception as e:
            print(f"Education generation error: {e}")
            return f"Unable to generate educational content for {topic} at this time."
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with helpfulness evaluation"""
        workflow = StateGraph(AgentState)
        
        # Add core nodes
        workflow.add_node("check_cache", self._check_cache)
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("search_local", self._search_local_node)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence)
        workflow.add_node("search_pubmed", self._search_pubmed_node)
        workflow.add_node("synthesize_information", self._synthesize_information)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add helpfulness evaluation nodes
        workflow.add_node("evaluate_helpfulness", self._evaluate_helpfulness)
        workflow.add_node("refine_response", self._refine_response)
        workflow.add_node("cache_response", self._cache_response)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # Define the flow
        workflow.set_entry_point("check_cache")
        
        # Cache decision routing
        workflow.add_conditional_edges(
            "check_cache",
            self._route_cache_decision,
            {
                "cache_hit": "finalize_response",
                "cache_miss": "analyze_query"
            }
        )
        
        # Core workflow - EARLY EXIT OPTIMIZATION
        workflow.add_edge("analyze_query", "search_local")
        workflow.add_edge("search_local", "evaluate_confidence")
        
        # Confidence-based routing for early exit
        workflow.add_conditional_edges(
            "evaluate_confidence",
            self._route_based_on_confidence,
            {
                "skip_pubmed": "synthesize_information",  # High confidence - skip PubMed
                "search_pubmed": "search_pubmed"  # Low confidence - search PubMed
            }
        )
        
        workflow.add_edge("search_pubmed", "synthesize_information")
        workflow.add_edge("synthesize_information", "generate_response")
        workflow.add_edge("generate_response", "evaluate_helpfulness")
        
        # Helpfulness evaluation routing
        workflow.add_conditional_edges(
            "evaluate_helpfulness",
            self._route_based_on_helpfulness,
            {
                "end": "cache_response",
                "refine": "refine_response",
                "max_attempts_reached": "cache_response"
            }
        )
        
        # Complete the loop or end
        workflow.add_edge("refine_response", "analyze_query")  # Loop back to start with refined query
        workflow.add_edge("cache_response", "finalize_response")  # Cache then finalize
        workflow.add_edge("finalize_response", END)  # End the workflow
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _check_cache(self, state: AgentState) -> AgentState:
        """Check if the query is in the cache"""
        query = state["query"]
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        state["cache_key"] = cache_key
        
        # Check if we have cached results for this query
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            state["cache_hit"] = True
            state["pubmed_results"] = cached_data.get("pubmed_results", [])
            state["context"] = cached_data.get("context", "")
            state["response"] = cached_data.get("response", "")
            print(f"🎯 Cache hit for query: {query[:50]}...")
        else:
            state["cache_hit"] = False
            print(f"🔍 Cache miss for query: {query[:50]}...")
        
        return state
    
    def _route_cache_decision(self, state: AgentState) -> str:
        """Route based on cache hit/miss"""
        if state["cache_hit"]:
            return "cache_hit"
        else:
            return "cache_miss"
    
    def _cache_response(self, state: AgentState) -> AgentState:
        """Cache the response and related data for future queries"""
        cache_key = state["cache_key"]
        
        # Cache the complete response data
        cache_data = {
            "pubmed_results": state.get("pubmed_results", []),
            "context": state.get("context", ""),
            "response": state.get("response", ""),
            "timestamp": time.time()
        }
        
        self._cache[cache_key] = cache_data
        print(f"💾 Cached response for future queries")
        
        return state
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query to determine the best approach"""
        query = state["query"]
        
        # Use LLM to analyze query intent
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Analyze the user query and determine what type of response is needed.
            Classify the query into one of these categories:
            - workout_adaptation: User wants help adapting a workout
            - education: User wants to learn about a fitness/CrossFit topic
            - general_advice: User wants general fitness advice
            - research_question: User has a specific question that needs scientific backing
            
            Respond with just the category name.
            """),
            ("human", f"Query: {query}")
        ])
        
        response = self.llm.invoke(analysis_prompt.format_messages())
        
        state["next_action"] = response.content.strip().lower()
        return state
    
    def _search_pubmed_node(self, state: AgentState) -> AgentState:
        """Search PubMed for relevant articles"""
        query = state["query"]
        results = self.search_pubmed(query, max_results=5)
        state["pubmed_results"] = results
        return state
    
    def _search_local_node(self, state: AgentState) -> AgentState:
        """Search local documents"""
        query = state["query"]
        results = self.search_local_documents(query)
        state["local_rag_results"] = results
        return state
    
    def _evaluate_confidence(self, state: AgentState) -> AgentState:
        """Evaluate confidence in local results"""
        local_results = state["local_rag_results"]
        query = state["query"]
        
        # Convert local_results to string and escape curly braces for safe formatting
        local_results_str = str(local_results)
        # Escape curly braces to prevent formatting errors
        local_results_escaped = local_results_str.replace('{', '{{').replace('}', '}}')
        
        # Use LLM to evaluate confidence based on query and local results
        confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Evaluate the confidence in the local results for answering the user's query.
            Consider:
            1. Relevance of results to the query
            2. Completeness of information
            3. Quality and accuracy of content
            4. Whether results fully address the query
            
            Respond with ONLY a confidence score between 0.0 and 1.0 (e.g., 0.85)
            """),
            ("human", f"""
            User Query: {query}
            
            Local Results: {local_results_escaped}
            
            Confidence score (0.0-1.0):
            """)
        ])
        
        try:
            response = self.llm.invoke(confidence_prompt.format_messages())
            confidence_score = float(response.content.strip())
            # Ensure score is within valid range
            confidence_score = max(0.0, min(1.0, confidence_score))
        except (ValueError, AttributeError):
            # Default to low confidence if parsing fails
            confidence_score = 0.3
        
        state["local_confidence"] = confidence_score
        print(f"🎯 Local confidence score: {confidence_score:.2f}")
        
        return state
    
    def _route_based_on_confidence(self, state: AgentState) -> str:
        """Determine next step based on confidence score"""
        confidence_score = state["local_confidence"]
        
        # If high confidence, skip PubMed search
        if confidence_score >= 0.7:
            print(f"✅ High confidence ({confidence_score:.2f}) - skipping PubMed search")
            return "skip_pubmed"
        
        # Otherwise, search PubMed
        print(f"🔍 Low confidence ({confidence_score:.2f}) - searching PubMed")
        return "search_pubmed"
    
    def _synthesize_information(self, state: AgentState) -> AgentState:
        """Synthesize information from all sources"""
        pubmed_info = ""
        if state["pubmed_results"]:
            pubmed_info = "\n\n".join([
                f"**{article['title']}** ({article['year']})\n{article['abstract'][:500]}..."
                for article in state["pubmed_results"][:3]
            ])
        
        local_info = ""
        if state["local_rag_results"]:
            local_info = "\n\n".join([
                f"**{result['source']}**\n{result['content'][:500]}..."
                for result in state["local_rag_results"][:3]
            ])
        
        state["context"] = f"PubMed Research:\n{pubmed_info}\n\nLocal Documents:\n{local_info}"
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response based on all gathered information with quality-aware synthesis"""
        query = state["query"]
        context = state["context"]
        action_type = state["next_action"]
        context_quality = state.get("context_quality_score", 0.5)
        pubmed_quality = state.get("pubmed_quality_score", 0.0)
        local_quality = state.get("local_quality_score", 0.0)
        
        # Adjust response strategy based on context quality
        if context_quality >= 0.8:
            response_strategy = "comprehensive"
        elif context_quality >= 0.6:
            response_strategy = "balanced"
        else:
            response_strategy = "cautious"
        
        # Create context-aware prompt
        system_prompt = f"""
        You are a CrossFit and fitness expert providing evidence-based guidance.
        
        Context Quality Assessment:
        - Overall Context Quality: {context_quality:.2f}
        - PubMed Research Quality: {pubmed_quality:.2f}
        - Local Document Quality: {local_quality:.2f}
        
        Response Strategy: {response_strategy}
        
        Guidelines based on context quality:
        - High quality (0.8+): Provide comprehensive, detailed responses
        - Medium quality (0.6-0.8): Provide balanced responses with appropriate caveats
        - Low quality (<0.6): Provide cautious responses, acknowledge limitations
        
        Always prioritize safety and evidence-based recommendations.
        """
        
        if action_type == "workout":
            system_prompt += """
            Focus on creating a safe, effective workout plan.
            Include proper form cues, progression options, and safety considerations.
            """
        elif action_type == "education":
            system_prompt += """
            Focus on educational content about CrossFit principles, techniques, and science.
            Explain concepts clearly and provide actionable insights.
            """
        elif action_type == "health":
            system_prompt += """
            Focus on health and safety considerations.
            Provide evidence-based health guidance related to CrossFit and fitness.
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"""
            User Query: {query}
            
            Available Context:
            {context}
            
            Please provide a helpful, accurate response based on the available information.
            If context quality is low, acknowledge limitations and suggest seeking additional sources.
            """)
        ])
        
        # Generate response
        response = self.llm.invoke(prompt.format_messages())
        state["response"] = response.content
        
        # Add quality metadata to response
        quality_info = f"""
        
        📊 Response Quality Metrics:
        - Context Quality: {context_quality:.2f}
        - PubMed Quality: {pubmed_quality:.2f}
        - Local Quality: {local_quality:.2f}
        - Strategy: {response_strategy}
        """
        
        print(f"📝 Generated response with {response_strategy} strategy (quality: {context_quality:.2f})")
        
        return state
        
    def _evaluate_helpfulness(self, state: AgentState) -> AgentState:
        """Evaluate if the response is helpful"""
        query = state["query"]
        response = state["response"]
        
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
        
        # Add tracking info
        print(f"🔍 Helpfulness evaluation: {helpfulness_score} (attempt {state['attempt_count']})")
        
        return state
        
    def _route_based_on_helpfulness(self, state: AgentState) -> str:
        """Determine next step based on helpfulness score and attempt count"""
        helpfulness_score = state["helpfulness_score"]
        attempt_count = state["attempt_count"]
        max_attempts = state["max_attempts"]
        
        # If helpful, route to end
        if helpfulness_score == "Y":
            return "end"
        
        # If max attempts reached, route to end
        if attempt_count >= max_attempts:
            return "max_attempts_reached"
        
        # Otherwise, route to refine
        return "refine"
        
    def _refine_response(self, state: AgentState) -> AgentState:
        """Refine the response based on the original query"""
        query = state["query"]
        previous_response = state["response"]
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
        
        # Update state with refined query and increment attempt count
        state["query"] = refined_query
        state["attempt_count"] = attempt_count + 1
        
        # Add refinement info to messages
        state["messages"].append(SystemMessage(content=f"Refining query (attempt {attempt_count + 1}/{state['max_attempts']}): {refined_query}"))
        
        return state
        
    def _finalize_response(self, state: AgentState) -> AgentState:
        """Finalize the response to be returned to the user"""
        helpfulness_score = state["helpfulness_score"]
        response = state["response"]
        attempt_count = state["attempt_count"]
        max_attempts = state["max_attempts"]
        
        # If max attempts reached but still not helpful, add a note
        if helpfulness_score == "N" and attempt_count >= max_attempts:
            final_response = f"""
            {response}
            
            Note: I've made {attempt_count} attempts to provide the most helpful response possible.
            If you need more specific information, please consider refining your question.
            """
            state["response"] = final_response
            
        return state
    
    @traceable(name="pubmed_agent_query")
    def query(self, user_query: str, thread_id: str = "default", max_attempts: int = None) -> Dict[str, Any]:
        """Main entry point for querying the agent with helpfulness evaluation"""
        import time
        start_time = time.time()
        
        try:
            # Use provided max_attempts or default to instance value
            if max_attempts is None:
                max_attempts = self.max_attempts
                
            # Create initial state
            initial_state = {
                "query": user_query,
                "context": "",
                "pubmed_results": [],
                "local_rag_results": [],
                "workout_plan": None,
                "educational_content": None,
                "next_action": "",
                "messages": [],
                "response": "",
                "helpfulness_score": "",
                "attempt_count": 1,
                "max_attempts": max_attempts,
                "context_quality_score": 0.5,
                "pubmed_quality_score": 0.0,
                "local_quality_score": 0.0,
                "local_confidence": 0.0,
                "cache_key": "",
                "cache_hit": False
            }
            
            # Run the graph
            graph = self._build_graph()
            config = {"configurable": {"thread_id": "default"}}
            final_state = graph.invoke(initial_state, config)
            
            # Extract response - use the final response from the state
            # which has been evaluated for helpfulness
            response_content = final_state.get("response", "")
            if not response_content and final_state["messages"]:
                response_content = final_state["messages"][-1].content
            elif not response_content:
                response_content = "I apologize, but I couldn't generate a response at this time."
            
            # Include helpfulness information in the response
            attempts_info = ""
            
            if final_state.get("attempt_count", 1) > 1:
                attempts_info = f"\n\n(Response refined {final_state['attempt_count']-1} times for helpfulness)"
            
            execution_time = time.time() - start_time
            print(f"Query processed in {execution_time:.2f} seconds with {final_state.get('attempt_count', 1)} attempts")
            
            return {
                "response": response_content + attempts_info,
                "pubmed_sources": final_state["pubmed_results"],
                "local_sources": final_state["local_rag_results"],
                "context_used": final_state["context"],
                "helpfulness_score": final_state.get("helpfulness_score", ""),
                "attempts": final_state.get("attempt_count", 1)
            }
            
        except Exception as e:
            print(f"Agent query error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "pubmed_sources": [],
                "local_sources": [],
                "context_used": ""
            }
    
    async def aquery(self, user_query: str, thread_id: str = "default", max_attempts: int = None) -> Dict[str, Any]:
        """Async version of query method"""
        return await asyncio.to_thread(self.query, user_query, thread_id, max_attempts)


# Convenience function for easy import
def create_pubmed_agent(data_path: str = None, initialize_enhanced_rag: bool = True, langsmith_api_key: str = None, **kwargs):
    """Create a PubMed CrossFit Agent instance with optional enhanced RAG initialization"""
    # Pass the LangSmith API key explicitly to ensure tracing works
    if langsmith_api_key:
        kwargs['langsmith_api_key'] = langsmith_api_key
    agent = PubMedCrossFitAgent(**kwargs)
    
    if initialize_enhanced_rag:
        print("🔧 Setting up Enhanced RAG system...")
        
        # Auto-detect data path if not provided
        if data_path is None:
            # Try to find the data directory relative to this file
            current_file = os.path.abspath(__file__)
            # Go up from: api/features/agents/pub_med_agent.py -> api/data
            api_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            data_path = os.path.join(api_dir, "data")
            print(f"📍 Auto-detected data path: {data_path}")
        
        success = agent.initialize_enhanced_rag(data_path)
        if success:
            print("🎉 Agent created with Enhanced RAG capabilities!")
        else:
            print("⚠️  Agent created but Enhanced RAG initialization failed. Using standard RAG.")
    
    return agent