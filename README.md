# 🧠 Cortex-MCP

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-000000?style=for-the-badge&logo=langfuse&logoColor=white)

**Cortex-MCP** is a high-performance autonomous AI orchestrator designed for speed and modularity. It leverages the **Model Context Protocol (MCP)** to standardize tool connections and **Groq LPUs** for lightning-fast inference.

The system features a **Python FastAPI** backend that orchestrates complex multi-step workflows (Search, Wikipedia, Weather, RAG) and streams the reasoning process in real-time to a modern **React** frontend.

## ✨ Features

- **🚀 Ultra-Fast Inference:** Powered by Groq's Llama 3 models running on LPUs (500+ tokens/sec).
- **⚡ Real-Time Streaming:** Character-by-character streaming responses with visible "Thinking" logs.
- **🛠️ Model Context Protocol (MCP):** Modular tool architecture using `fastmcp`.
- **🔗 Tool Chaining:** Capable of complex multi-step reasoning (e.g., Search -> Fact Check -> Analysis).
- **🧠 RAG Knowledge Base:** Ingest websites and query vector memory using ChromaDB.
- **📊 Observability:** Full trace logging with Langfuse.
- **🎨 Modern UI:** Clean, Dark Mode interface built with React and standard CSS (No heavy frameworks).

## 🏗️ Architecture

1.  **Frontend:** React + Vite app that handles NDJSON streams.
2.  **Orchestrator:** FastAPI service that manages the LangGraph agent state.
3.  **MCP Server:** A dedicated process hosting tools (Wikipedia, SerpAPI, Weather, Vector DB).
4.  **LLM Provider:** Groq (Llama 3) or OpenRouter (GPT-OSS).

## 🚀 Getting Started

### Prerequisites

* **Node.js** (v18+)
* **Python** (v3.10+)
* **API Keys:** Groq, OpenAI (for Embeddings), SerpAPI, Langfuse.

### 1. Backend Setup

Navigate to the backend folder and set up the Python environment.

```bash
cd backend
# Create virtual env (optional but recommended)
python -m venv .mcp
# Activate it (Windows: .mcp\Scripts\activate | Mac: source .mcp/bin/activate)

# Install dependencies
```bash
pip install fastapi uvicorn mcp fastmcp langgraph langchain-groq langchain-openai langchain-community langfuse chromadb wikipedia google-search-results sentence-transformers
```

# Create a .env file in backend/:
```bash
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
SERPAPI_API_KEY=...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)
```

## Run the Orchestrator:
```
python orchestrator.py
# Server will start on [http://0.0.0.0:8002](http://0.0.0.0:8002)
```
# 2. Frontend Setup
Open a new terminal, navigate to the frontend folder.

```
cd frontend
npm install
```
# Create a .env file in frontend/:
```
VITE_API_URL=http://localhost:8002/api/chat
```

# Run the UI:
```
npm run dev
# App will run at http://localhost:5173 or 3000
```

## 💡 Usage Examples
Try asking the agent complex, multi-step questions to see the Chain of Thought in action:

# 1.Multi-Step Reasoning:
"Find out which city hosted the 1992 Summer Olympics, find its current mayor, and tell me the weather there right now."

# 2.RAG / Learning:
"Read this page https://en.wikipedia.org/wiki/LangChain and summarize what it does."

# 3.Real-Time Data:
"What is the stock price of NVDA?"

## 📂 Project Structure

```
cortex-mcp/
├── backend/
│   ├── orchestrator.py   # Main API & Agent Loop
│   ├── server.py         # MCP Tool Definitions
│   └── chroma_db_mcp/    # Vector Database Storage
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Chat Interface Logic
│   │   ├── App.css       # Dark Mode Styling
│   │   └── main.jsx
│   └── vite.config.js
└── README.md

```


# 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a Pull Request.

# 📄 License
MIT License.