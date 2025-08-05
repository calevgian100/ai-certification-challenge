# CrossFit AI Agent with Enhanced RAG

An intelligent CrossFit coaching assistant powered by advanced Retrieval-Augmented Generation (RAG) and agentic reasoning. The system combines scientific research from PubMed with local CrossFit expertise to provide evidence-based training guidance.

## 🚀 Features

- **Enhanced RAG System** - 95.8% faithfulness, 94.4% context precision (RAGAS validated)
- **Agentic Intelligence** - Multi-source reasoning combining PubMed research + local documents
- **Advanced Retrieval** - Ensemble retriever (semantic + BM25) with smart chunking
- **Full Observability** - LangSmith integration for production monitoring
- **Multi-Modal Support** - Text queries with scientific evidence backing

## 🏗️ Architecture

- **FastAPI Backend** - RESTful API with agent endpoints
- **Enhanced RAG Engine** - Advanced chunking and ensemble retrieval
- **PubMed Integration** - Scientific research retrieval
- **Qdrant Vector Store** - Semantic search capabilities
- **LangSmith Tracing** - End-to-end observability

## 📋 Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- LangSmith API key (optional, for tracing)
- Qdrant Cloud account url and API key

## 🚀 Local Deployment

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-certification-challenge
```

### 2. Install Dependencies

```bash
# Install all dependencies using uv
uv sync
```

### 3. Environment Configuration

Create an `env.yaml` file in the project root:

```yaml
openai_api_key: your_openai_api_key_here
langsmith_api_key: your_langsmith_api_key_here
qdrant_url: your_qdrant_cloud_url
qdrant_api_key: your_qdrant_api_key
```

### 4. Deploy Locally

```bash
# Start the FastAPI server
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Or in case that doesn't work do
# uv run python app.py
```

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Agent Endpoints**: http://localhost:8000/agents/

## 🔧 API Endpoints

### Agent Endpoints

- `POST /agents/query` - General CrossFit questions with enhanced RAG
- `POST /agents/workout-adaptation` - Personalized workout modifications
- `POST /agents/education` - Educational content about CrossFit/fitness

### RAG Endpoints

- `POST /rag/query` - Direct RAG queries
- `POST /upload` - Upload PDF documents
- `GET /pdfs` - List available documents

## 📊 Enhanced RAG Performance

Based on RAGAS evaluation:

| Metric            | Score | Performance |
| ----------------- | ----- | ----------- |
| Faithfulness      | 95.8% | Excellent   |
| Context Precision | 94.4% | Excellent   |
| Context Recall    | 93.5% | Very Good   |
| Answer Relevancy  | 85.0% | Good        |

## 🔍 Retrieval Techniques

- **Semantic Retrieval** - OpenAI embeddings with FAISS
- **BM25 Keyword Matching** - Exact term matching
- **Ensemble Retrieval** - Combined semantic + keyword (default)
- **Advanced Chunking** - 1000 chars with 200 char overlap
- **Content Filtering** - Removes headers/footers/TOC

## 📈 Observability

### LangSmith Integration

Monitor your agent's performance at:

- **Project URL**: https://smith.langchain.com/projects/crossfit-pubmed-agent
- **Traces**: Complete end-to-end query processing
- **Metrics**: Response times, context quality, retrieval performance

## 🧪 Testing

### Test Enhanced RAG System

```bash
# Run comprehensive agent tests
uv run python api/features/agents/test_enhanced_agent.py
```

### RAGAS Evaluation

```bash
# Run RAGAS evaluation on synthetic dataset
uv run python api/evaluator/ragas_eval.py
```

## 📁 Project Structure

```
api/
├── features/
│   ├── agents/          # Agentic reasoning and endpoints
│   ├── rag/            # Enhanced RAG implementation
│   ├── processors/     # Document processing
│   └── observability/  # LangSmith configuration
├── data/               # Local CrossFit documents
├── evaluator/          # RAGAS evaluation system
└── utils/              # Utility functions
```

## 🔑 Key Components

- **PubMedCrossFitAgent** - Main intelligent agent
- **EnhancedRAGQueryEngine** - Advanced retrieval system
- **LangSmithConfig** - Observability and tracing
- **RAGASSDGGenerator** - Synthetic data generation for evaluation

## 🎯 Next Steps

1. **Improve Answer Relevancy** - Target 90%+ from current 85%
2. **Multi-language Support** - Better Spanish query handling
3. **Query Classification** - Tailor responses to user expertise level
4. **Performance Optimization** - Use LangSmith traces to identify bottlenecks

## **AI Certification Challenge Responses**

### **Task 1: Problem Definition**

**Problem Statement:**
Athletes recovering from injuries often receive static rehab plans that are hard to adapt to daily CrossFit training, leading to confusion, setbacks, or ineffective modifications.

**User Impact:**
Based on my experience and interviews, CrossFit athletes are highly engaged, performance-driven individuals who thrive on dynamic, constantly varied training. When they're injured, they're often handed rigid rehab plans (typically in PDF form) that don't account for the complexity of daily WODs, movement substitutions, or how their body feels on a given day. This situation forces them to either skip workouts entirely or attempt unsafe modifications without proper guidance, risking re-injury or slowed recovery.

This is especially frustrating because the athletes expect personalized, coach-like support. The inability to easily integrate their rehab plans into their ongoing training disrupts consistency and motivation.

### **Task 2: Proposed Solution**

**Solution Overview:**
This solution enables users to upload their clinic history or rehab plan PDF directly into the app, allowing their selected AI trainer to instantly understand their injury, limitations, and prescribed recovery movements. Using Retrieval-Augmented Generation (RAG), the AI reads and interprets the document, then dynamically tailors daily workout suggestions, scaling options, and movement substitutions based on both the rehab plan and the user's current training goals. The experience is seamless, once the PDF is uploaded, users can simply ask questions like "What should I do today if I can't squat?" or "Can I still train my upper body while rehabbing my knee?" and receive intelligent, personalized responses.

To the user, this feels like having a dedicated coach who not only remembers their injury but actively helps them train around it. Whether they're following a gym's programming or a custom track within the app, the AI trainer will continuously adjust recommendations to keep them progressing without compromising recovery. It bridges the gap between rehab and training, giving users confidence, clarity, and support throughout the healing process, without needing to interpret medical documents or guess what's safe and what is not.

**Technology Stack:**

- **LLM:** OpenAI GPT-4o-mini for intelligent synthesis and personalized responses
- **Embedding Model:** text-embedding-3-small
- **Orchestration:** LangGraph for agentic workflow management
- **Vector Database:** Qdrant for fast semantic search across knowledge base
- **Monitoring:** LangSmith for production monitoring and optimization insights
- **Evaluation:** RAGAS framework for comprehensive performance assessment
- **User Interface:** FastAPI with web-based frontend using Next.js

**Agentic Reasoning:**
The PubMedCrossFitAgent uses agentic reasoning for multi-source decision making, intelligently combining scientific research from PubMed with local CrossFit expertise, and providing context synthesis that merges scientific evidence with practical training guidance.

### **Task 3: Data Sources & APIs**

**External APIs:**

- **PubMed:** Scientific validation and research-backed recommendations
- **OpenAI:** Intelligent synthesis and personalized responses
- **LangSmith:** Production monitoring and optimization insights

**Data Sources:**

- **Local PDFs:** Clinic history of the athletes and other relevant pdfs
- **Qdrant:** Fast semantic search across local knowledge base

**Chunking Strategy:**
RecursiveCharacterTextSplitter with 1000 character chunks and 200 character overlap. It removes headers, footers and tables of content and the overlap ensures exercise instructions aren't cut mid-sentence.

### **Task 4: End-to-End Prototype**

✅ **Completed:** Built and deployed locally using FastAPI with enhanced RAG capabilities and full LangSmith observability.

### **Task 5: RAGAS Evaluation Results**

| Metric                | Percentage | Performance |
| --------------------- | ---------- | ----------- |
| **Faithfulness**      | 95.8%      | Excellent   |
| **Context Precision** | 94.4%      | Excellent   |
| **Context Recall**    | 93.5%      | Very Good   |
| **Answer Relevancy**  | 85.0%      | Good        |

**Conclusion:** We can conclude that the system is retrieving high-quality, relevant content that enables accurate and faithful responses.

### **Task 6: Advanced Retrieval**

**Chosen Technique:** Ensemble retrieval (semantic + BM25) based on RAGAS evaluation showing 95.8% faithfulness and 94.4% context precision being the highest of them all.

**Retrieval Techniques Assessed:**

- **Semantic Retrieval:** Uses OpenAI embeddings for conceptual understanding
- **BM25 Keyword Matching:** Provides exact keyword matching for specific exercise names
- **Ensemble Retrieval:** Combines both approaches for optimal results
- **Compression Retrieval:** Filters irrelevant content
- **Multi-Query Retrieval:** Generates query variations
- **Parent Document Retrieval:** Provides larger context windows

**Implemented:** Ensemble retrieval (semantic + BM25) based on RAGAS evaluation showing 95.8% faithfulness and 94.4% context precision being the highest of them all.

### **Task 7: Performance Assessment**

**Improvement Results:**
Before:

| Metric                | Percentage | Performance |
| --------------------- | ---------- | ----------- |
| **Faithfulness**      | 74.4%      | Mid         |
| **Context Precision** | 84.3%      | Excellent   |
| **Context Recall**    | 100.0%     | Very Good   |
| **Answer Relevancy**  | 93.3%      | Good        |

After:

| Metric                | Percentage | Performance |
| --------------------- | ---------- | ----------- |
| **Faithfulness**      | 95.8%      | Excellent   |
| **Context Precision** | 94.4%      | Excellent   |
| **Context Recall**    | 93.5%      | Very Good   |
| **Answer Relevancy**  | 85.0%      | Good        |

**Future Improvements:**
Now, although we improved the faithfulness a lot, the answer relevancy has some room for improvement. I want to boost the Answer Relevancy from 85% to, at least, 90%+. Fine-tune prompt engineering to better align responses with user intent and Leverage LangSmith traces to identify slow queries and optimize retrieval speed.

---

**[Loom Video Link HERE](https://www.loom.com/share/b08a09350f1e48f2b51e84a5b02ead9a?sid=ff77a487-98f0-4909-8fe1-be557566d537)**
👆👆👆👆👆👆👆👆👆👆👆👆👆👆

**Built with Enhanced RAG + Agentic Intelligence for Evidence-Based CrossFit Coaching** 🏋️‍♂️
