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
- Qdrant Cloud account (or local Qdrant instance)

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

## 🐛 Troubleshooting

### Common Issues

1. **LangSmith traces not appearing**

   - Ensure `langsmith_api_key` is in `env.yaml`
   - Check server logs for LangSmith initialization messages

2. **Qdrant connection errors**

   - Verify `qdrant_url` and `qdrant_api_key` in `env.yaml`
   - Ensure Qdrant Cloud instance is running

3. **Enhanced RAG initialization fails**
   - Check that PDF files exist in `api/data/` directory
   - Verify OpenAI API key has sufficient credits

### Debug Mode

```bash
# Run with debug logging
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000 --log-level debug
```

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs (when running)
- **RAGAS Evaluation**: See `api/evaluator/ragas_evaluation_results.json`
- **LangSmith Traces**: https://smith.langchain.com/projects/crossfit-pubmed-agent

## 🎯 Next Steps

1. **Improve Answer Relevancy** - Target 90%+ from current 85%
2. **Multi-language Support** - Better Spanish query handling
3. **Query Classification** - Tailor responses to user expertise level
4. **Performance Optimization** - Use LangSmith traces to identify bottlenecks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run python -m pytest`
5. Submit a pull request

## 📄 License

[Your License Here]

---

**Built with Enhanced RAG + Agentic Intelligence for Evidence-Based CrossFit Coaching** 🏋️‍♂️
