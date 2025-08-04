from typing import List, Dict, Any, Optional
from api.features.store.vector_store import QdrantVectorStore
from api.domain.chatmodel import ChatOpenAI
from api.domain.prompts import SystemRolePrompt, UserRolePrompt

# RAG Engine Constants
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant that answers questions based on the provided context."
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
DEFAULT_K = 5

# RAG Response Templates
NO_RESULTS_RESPONSE = "I don't have any relevant information to answer your question."
NO_PDF_CONTENT_RESPONSE = "I don't have any relevant information from your uploaded PDFs to answer this question. Please try a different question related to the PDF content."
LOW_RELEVANCE_RESPONSE = "I don't know (relevance: {relevance_percentage}%)."
ERROR_RESPONSE = "I encountered an error while searching for relevant information. Please try again."

# Prompt Enhancements
PERSONA_REMINDER = "\n\nIMPORTANT: Maintain your trainer persona's expertise level, tone, and characteristics when answering. Your response should clearly reflect your specific trainer persona. Format your response with paragraph breaks after each sentence for better readability. Do not combine multiple sentences into a single paragraph."

# Streaming Constants
STREAM_COMPLETE_MARKER = "__STREAM_COMPLETE__"

# Relevance Thresholds
MIN_RELEVANCE_SCORE = 0.25 
# Context Formatting
CONTEXT_FORMAT = "Document {index} (Source: {source}):\n{text}\n"

class RAGQueryEngine:
    def __init__(self, 
                 collection_name: str = DEFAULT_COLLECTION_NAME, 
                 model_name: str = DEFAULT_MODEL_NAME,
                 k: int = DEFAULT_K):
        self.vector_store = QdrantVectorStore(collection_name=collection_name)
        self.chat_model = ChatOpenAI(model_name=model_name)
        self.k = k
    
    def query(self, query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system with a question"""
        # Search for relevant documents
        search_results = self.vector_store.similarity_search(query, k=self.k)
        
        if not search_results:
            return {
                "answer": NO_RESULTS_RESPONSE,
                "sources": []
            }
        
        # Format context from search results
        context = self._format_context(search_results)
        
        # Create messages for the chat model
        messages = [
            SystemRolePrompt(system_prompt or self.DEFAULT_SYSTEM_PROMPT).create_message(),
            UserRolePrompt(
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ).create_message()
        ]
        
        # Get response from chat model
        response = self.chat_model.run(messages)
        
        # Extract sources
        sources = [{
            "text": result["text"],
            "source": result["metadata"].get("source", "Unknown"),
            "score": result["score"]
        } for result in search_results]
        
        return {
            "answer": response,
            "sources": sources
        }
    
    async def aquery(self, query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system with a question asynchronously"""
        # Search for relevant documents
        search_results = await self.vector_store.asimilarity_search(query, k=self.k)
        
        if not search_results:
            return {
                "answer": NO_RESULTS_RESPONSE,
                "sources": []
            }
        
        # Format context from search results
        context = self._format_context(search_results)
        
        # Create messages for the chat model
        messages = [
            SystemRolePrompt(system_prompt or self.DEFAULT_SYSTEM_PROMPT).create_message(),
            UserRolePrompt(
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ).create_message()
        ]
        
        # Get response from chat model
        response = self.chat_model.run(messages)
        
        # Extract sources
        sources = [{
            "text": result["text"],
            "source": result["metadata"].get("source", "Unknown"),
            "score": result["score"]
        } for result in search_results]
        
        return {
            "answer": response,
            "sources": sources
        }
    
    async def astream_query(self, query: str, system_prompt: Optional[str] = None):
        """Stream the RAG response asynchronously"""
        try:
            # Search for relevant documents
            search_results = await self.vector_store.asimilarity_search(query, k=self.k)
            
            # If no relevant PDF content is found, return a clear message and stop
            if not search_results or len(search_results) == 0:
                yield NO_PDF_CONTENT_RESPONSE
                return
                
            # Check if ALL relevance scores are too low
            max_score = max(result.get('score', 0) for result in search_results)
            avg_score = sum(result.get('score', 0) for result in search_results) / len(search_results)
            
            # Only show the not relevant message if ALL scores are below threshold
            if max_score < MIN_RELEVANCE_SCORE:
                relevance_percentage = int(avg_score * 100)
                yield LOW_RELEVANCE_RESPONSE.format(relevance_percentage=relevance_percentage)
                return
            
            # Format context from search results
            context = self._format_context(search_results)
            
        except Exception as e:
            yield ERROR_RESPONSE
            return
        
        # Use the system prompt from the frontend or default
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Add the persona reminder to the system prompt
        enhanced_system_prompt = system_prompt + PERSONA_REMINDER
        
        # Create messages for the chat model
        messages = [
            SystemRolePrompt(enhanced_system_prompt).create_message(),
            UserRolePrompt(
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ).create_message()
        ]
        
        # Stream response from chat model
        async for chunk in self.chat_model.astream(messages):
            yield chunk
            
        # Add completion marker to signal the end of the stream
        yield STREAM_COMPLETE_MARKER
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into a context string for prompt"""
        # Extract text and source from search results
        formatted_results = []
        for i, result in enumerate(search_results):
            try:
                text = result.get("text", "")
                source = result.get("metadata", {}).get("source", "Unknown")
                formatted_results.append(CONTEXT_FORMAT.format(index=i+1, source=source, text=text))
            except Exception:
                # Skip this result if there's an error
                pass
        
        return "\n\n".join(formatted_results)
