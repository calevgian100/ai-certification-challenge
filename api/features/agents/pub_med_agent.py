import os
import json
import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

# Local imports
from api.features.rag.rag import RAGQueryEngine
from api.features.store.vector_store import QdrantVectorStore
from api.features.processors.local_document_manager import LocalDocumentManager

# PubMed and Bio imports
import requests
from Bio import Entrez
from xml.etree import ElementTree as ET


class AgentState(TypedDict):
    """State for the PubMed CrossFit Agent"""
    messages: Annotated[List, "Messages in the conversation"]
    query: str
    context: str
    pubmed_results: List[Dict[str, Any]]
    local_rag_results: List[Dict[str, Any]]
    workout_plan: Optional[Dict[str, Any]]
    educational_content: Optional[str]
    next_action: str


class PubMedCrossFitAgent:
    """An agentic RAG system that leverages PubMed and local documents for CrossFit guidance"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 langsmith_api_key: str = None,
                 email: str = "user@example.com"):
        
        # Set up API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.langsmith_api_key = langsmith_api_key or os.getenv("langsmith_api_key")
        
        if self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = "crossfit-pubmed-agent"
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=self.openai_api_key
        )
        
        # Initialize RAG components
        self.rag_engine = RAGQueryEngine()
        self.local_doc_manager = LocalDocumentManager()
        
        # Set up PubMed
        Entrez.email = email
        
        # Build the graph
        self.graph = self._build_graph()
    
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
    
    def search_local_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search local documents and Qdrant vector store for relevant information"""
        try:
            # Use existing RAG engine
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
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("search_pubmed", self._search_pubmed_node)
        workflow.add_node("search_local", self._search_local_node)
        workflow.add_node("synthesize_information", self._synthesize_information)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define the flow
        workflow.set_entry_point("analyze_query")
        
        # workflow.add_edge("analyze_query", "search_pubmed")
        # workflow.add_edge("search_pubmed", "search_local")
        workflow.add_edge("analyze_query", "search_local")
        workflow.add_edge("search_local", "search_pubmed")
        workflow.add_edge("search_pubmed", "synthesize_information")
        workflow.add_edge("synthesize_information", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
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
        """Generate the final response based on all gathered information"""
        query = state["query"]
        context = state["context"]
        action_type = state["next_action"]
        
        # Create appropriate prompt based on action type
        if "workout" in action_type:
            system_prompt = """
            You are an expert CrossFit coach. Based on the scientific evidence and local documents provided,
            help the user adapt their workout. Focus on safety, progression, and evidence-based modifications.
            """
        elif "education" in action_type:
            system_prompt = """
            You are a fitness educator. Provide comprehensive, accurate educational content based on the
            scientific evidence and documents provided. Make it accessible and practical.
            """
        else:
            system_prompt = """
            You are an expert CrossFit coach and fitness educator. Provide helpful, evidence-based advice
            based on the scientific research and local documents provided.
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"""
            User Query: {query}
            
            Available Context:
            {context}
            
            Please provide a comprehensive, helpful response based on the available evidence.
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        # Add the response to messages
        if "messages" not in state:
            state["messages"] = []
        
        state["messages"].append(HumanMessage(content=query))
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    @traceable
    def query(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Main entry point for querying the agent"""
        try:
            # Create initial state
            initial_state = {
                "query": user_query,
                "messages": [],
                "context": "",
                "pubmed_results": [],
                "local_rag_results": [],
                "workout_plan": None,
                "educational_content": None,
                "next_action": ""
            }
            
            # Run the graph
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract the response
            if final_state["messages"]:
                response_content = final_state["messages"][-1].content
            else:
                response_content = "I apologize, but I couldn't generate a response at this time."
            
            return {
                "response": response_content,
                "pubmed_sources": final_state["pubmed_results"],
                "local_sources": final_state["local_rag_results"],
                "context_used": final_state["context"]
            }
            
        except Exception as e:
            print(f"Agent query error: {e}")
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "pubmed_sources": [],
                "local_sources": [],
                "context_used": ""
            }
    
    async def aquery(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Async version of query method"""
        return await asyncio.to_thread(self.query, user_query, thread_id)


# Convenience function for easy import
def create_pubmed_agent(**kwargs) -> PubMedCrossFitAgent:
    """Create a PubMed CrossFit Agent instance"""
    return PubMedCrossFitAgent(**kwargs)