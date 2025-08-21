# Architecture Diagrams

This directory contains visual diagrams of the system architecture.

## Viewing the Diagrams

### Option 1: GitHub/GitLab (Recommended)
Simply view the `.md` files on GitHub/GitLab - they render Mermaid diagrams automatically:
- [Component Architecture](./component-architecture.md)
- [LangGraph Workflow](./langgraph-workflow.md)

### Option 2: VS Code with Mermaid Extension
1. Install the "Mermaid Markdown Syntax Highlighting" extension
2. Install the "Markdown Preview Mermaid Support" extension
3. Open the `.md` files and use `Cmd+Shift+V` to preview

### Option 3: Online Mermaid Editor
1. Copy the mermaid code from the files
2. Paste into [Mermaid Live Editor](https://mermaid.live/)
3. View and export as PNG/SVG

### Option 4: Command Line (with mermaid-cli)
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG images
mmdc -i component-architecture.md -o component-architecture.png
mmdc -i langgraph-workflow.md -o langgraph-workflow.png
```

## Files

- `component-architecture.md` - Overall system component diagram
- `langgraph-workflow.md` - Detailed LangGraph agent workflow
- `README.md` - This file with viewing instructions
