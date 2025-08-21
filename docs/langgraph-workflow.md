# LangGraph Workflow Diagram

```mermaid
graph TD
    Start([User Query]) --> AnalyzeQuery[Analyze Query]
    
    AnalyzeQuery --> SearchLocal[Search Local Documents]
    SearchLocal --> SearchPubMed[Search PubMed]
    SearchPubMed --> Synthesize[Synthesize Information]
    Synthesize --> GenerateResponse[Generate Response]
    
    GenerateResponse --> EvaluateHelpfulness{Evaluate Helpfulness}
    
    EvaluateHelpfulness -->|Score: Y| FinalizeResponse[Finalize Response]
    EvaluateHelpfulness -->|Score: N & Attempts < Max| RefineResponse[Refine Response]
    EvaluateHelpfulness -->|Score: N & Max Attempts| FinalizeResponse
    
    RefineResponse --> AnalyzeQuery
    FinalizeResponse --> End([Return Response])
    
    %% Node Details
    AnalyzeQuery --> |Intent Detection| QueryType{Query Type}
    QueryType -->|Workout| WorkoutFlow[Workout Adaptation Flow]
    QueryType -->|Education| EducationFlow[Education Flow]
    QueryType -->|General| GeneralFlow[General Advice Flow]
    
    WorkoutFlow --> SearchLocal
    EducationFlow --> SearchLocal
    GeneralFlow --> SearchLocal
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#0277bd,stroke-width:3px
    classDef process fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef evaluation fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef flow fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class Start,End startEnd
    class AnalyzeQuery,SearchLocal,SearchPubMed,Synthesize,GenerateResponse,RefineResponse,FinalizeResponse process
    class EvaluateHelpfulness,QueryType decision
    class WorkoutFlow,EducationFlow,GeneralFlow flow
```

## Workflow Details

### Node Functions:

1. **Analyze Query**: Determines query intent and type
2. **Search Local**: Retrieves relevant local document chunks
3. **Search PubMed**: Fetches scientific articles from PubMed API
4. **Synthesize Information**: Combines all sources into context
5. **Generate Response**: Creates initial response using LLM
6. **Evaluate Helpfulness**: Scores response quality (Y/N)
7. **Refine Response**: Improves query and regenerates if needed
8. **Finalize Response**: Prepares final output for user

### Key Features:

- **Helpfulness Loop**: Automatic quality control with up to 3 attempts
- **Multi-source RAG**: Combines local docs + PubMed research
- **Intent-based Routing**: Different flows for workout, education, general queries
- **Memory Management**: LangGraph MemorySaver for conversation state
- **Observability**: LangSmith tracing throughout workflow
