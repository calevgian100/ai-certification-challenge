# CrossFit LLM App Stack

```mermaid
graph TB
    subgraph "Data Layer"
        PDFs[PDF Documents<br/>CrossFit Training Guides]
        PubMed[PubMed API<br/>Scientific Articles]
        UserQueries[User Queries<br/>Contextual Data]
    end

    subgraph "Vector Storage"
        Qdrant[Qdrant Cloud<br/>Vector Database]
        Cache[In-Memory Cache<br/>SHA256 Query Hashing]
    end

    subgraph "Processing & Orchestration"
        LangChain[LangChain<br/>🔗 Prompt Engineering<br/>🔗 Document Processing<br/>🔗 Chain Management]
        LangGraph[LangGraph<br/>🔀 Workflow Orchestration<br/>🔀 State Management<br/>🔀 Conditional Routing]
        LangSmith[LangSmith<br/>📊 Observability<br/>📊 Tracing<br/>📊 Evaluation]
    end

    subgraph "LLM & AI Services"
        OpenAI[OpenAI GPT-4o-mini<br/>🧠 Response Generation<br/>🧠 Query Analysis<br/>🧠 Confidence Scoring]
        Embeddings[OpenAI Embeddings<br/>🔢 Text Vectorization<br/>🔢 Semantic Search]
    end

    subgraph "Agent Layer"
        PubMedAgent[PubMed CrossFit Agent<br/>🚀 Smart Caching<br/>🚀 Early Exit Logic<br/>🚀 Quality Scoring]
        HelpfulEval[Helpful Evaluator<br/>✅ Response Quality<br/>✅ Refinement Logic]
    end

    subgraph "API & Interface"
        FastAPI[FastAPI Backend<br/>🌐 REST Endpoints<br/>🌐 Agent Routing]
        NextJS[Next.js Frontend<br/>💬 Chat Interface<br/>📄 PDF Upload]
    end

    subgraph "User Experience"
        User[CrossFit Athletes<br/>👤 Workout Guidance<br/>👤 Injury Prevention<br/>👤 Education]
    end

    %% Data Flow
    PDFs --> Qdrant
    PubMed --> Cache
    UserQueries --> FastAPI

    %% Processing Flow
    Qdrant --> LangChain
    Cache --> LangChain
    LangChain --> LangGraph
    LangGraph --> LangSmith

    %% AI Services
    LangChain --> OpenAI
    LangChain --> Embeddings
    Embeddings --> Qdrant

    %% Agent Flow
    LangGraph --> PubMedAgent
    PubMedAgent --> HelpfulEval
    PubMedAgent --> OpenAI
    PubMedAgent --> PubMed
    PubMedAgent --> Qdrant

    %% Interface Flow
    FastAPI --> PubMedAgent
    NextJS --> FastAPI
    User --> NextJS

    %% Observability
    PubMedAgent --> LangSmith
    FastAPI --> LangSmith

    %% Styling
    classDef data fill:#e8f4f8,stroke:#1976d2,stroke-width:2px
    classDef storage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef processing fill:#e8f5e8,stroke:#388e3c,stroke-width:3px
    classDef llm fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    classDef agent fill:#e1f5fe,stroke:#0277bd,stroke-width:3px
    classDef interface fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef user fill:#f1f8e9,stroke:#689f38,stroke-width:2px

    class PDFs,PubMed,UserQueries data
    class Qdrant,Cache storage
    class LangChain,LangGraph,LangSmith processing
    class OpenAI,Embeddings llm
    class PubMedAgent,HelpfulEval agent
    class FastAPI,NextJS interface
    class User user
```

## CrossFit LLM Application Stack

### **🏗️ Architecture Overview**

This diagram shows how our CrossFit application leverages the modern LLM app stack for intelligent fitness guidance.

### **📊 Stack Components**

#### **Data Layer**

- **PDF Documents**: CrossFit training guides, injury prevention materials
- **PubMed API**: Scientific research articles for evidence-based responses
- **User Queries**: Contextual workout and health questions

#### **Vector Storage**

- **Qdrant Cloud**: Scalable vector database for semantic search
- **In-Memory Cache**: SHA256-based query deduplication for performance

#### **Processing & Orchestration**

- **LangChain**: Prompt engineering, document processing, chain management
- **LangGraph**: Advanced workflow orchestration with conditional routing
- **LangSmith**: Full observability, tracing, and evaluation

#### **LLM & AI Services**

- **OpenAI GPT-4o-mini**: Response generation, analysis, confidence scoring
- **OpenAI Embeddings**: Text vectorization for semantic search

#### **Agent Layer** 🚀

- **PubMed CrossFit Agent**: Optimized with caching, early exit, quality scoring
- **Helpful Evaluator**: Response quality assessment and refinement

#### **API & Interface**

- **FastAPI**: High-performance backend with REST endpoints
- **Next.js**: Modern frontend with chat interface and PDF upload

### **🔄 Data Flow**

1. **Data Ingestion**: PDFs → Vector Storage, PubMed → Cache
2. **Query Processing**: User → Frontend → API → Agent
3. **Intelligent Routing**: Cache Check → Local Search → Confidence Eval → Conditional PubMed
4. **Response Generation**: Context Synthesis → Quality Scoring → LLM → Response
5. **Quality Control**: Helpfulness Evaluation → Refinement → Cache Storage

### **⚡ Performance Optimizations**

- **Smart Caching**: Eliminates redundant API calls
- **Early Exit Logic**: Skips unnecessary external searches
- **Quality-Aware Generation**: Adaptive response strategies
- **Observability**: Full workflow tracing and monitoring

### **🎯 Use Cases**

- **Workout Adaptation**: Personalized exercise modifications
- **Injury Prevention**: Evidence-based safety guidance
- **Education**: Scientific explanations of CrossFit principles
- **Performance Optimization**: Data-driven training advice
