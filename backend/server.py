# import os
# import logging
# from typing import Annotated
# from dotenv import load_dotenv

# # MCP Imports
# from fastmcp import FastMCP

# # LangChain & Tools
# from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma

# # Observability
# # We wrap the import in try-except to give a clear error if the upgrade failed
# try:
#     from langfuse.decorators import observe
# except ImportError:
#     raise ImportError("Langfuse is outdated. Run: pip install --upgrade langfuse")

# # 1. Setup Logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger("mcp_server")

# # 2. Load Env
# load_dotenv()

# # 3. Initialize FastMCP Server
# mcp = FastMCP(
#     "SuperAgent Tools", 
#     dependencies=["langchain", "chromadb", "openai", "google-search-results", "langfuse"]
# )

# # --- RAG SETUP ---
# DB_DIR = "./chroma_db_mcp"

# # Initialize Embeddings (OpenRouter Config)
# embeddings = OpenAIEmbeddings(
#     model="openai/text-embedding-3-small", 
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# # Initialize Vector DB
# vector_store = Chroma(
#     persist_directory=DB_DIR,
#     embedding_function=embeddings,
#     collection_name="agent_knowledge"
# )

# # --- TOOLS ---

# @mcp.tool()
# @observe(name="get_weather")
# def get_weather(city: str) -> str:
#     """Get the current weather for a specific city."""
#     logger.info(f"Tool Call: Weather for {city}")
#     return f"The weather in {city} is Sunny, 25°C. Wind 10km/h."

# @mcp.tool()
# @observe(name="search_web")
# def search_web(query: str) -> str:
#     """Search the real-time web using Google (SerpApi)."""
#     logger.info(f"Tool Call: Search for {query}")
#     try:
#         search = SerpAPIWrapper()
#         return search.run(query)
#     except Exception as e:
#         return f"Search Error: {str(e)}"

# @mcp.tool()
# @observe(name="search_wikipedia")
# def search_wikipedia(query: str) -> str:
#     """Search Wikipedia for encyclopedic knowledge."""
#     logger.info(f"Tool Call: Wikipedia for {query}")
#     try:
#         wiki = WikipediaAPIWrapper()
#         return wiki.run(query)
#     except Exception as e:
#         return f"Wikipedia Error: {str(e)}"

# @mcp.tool()
# @observe(name="ingest_knowledge")
# def ingest_knowledge(url: str) -> str:
#     """Read a website and learn its content (Save to Vector DB)."""
#     logger.info(f"Tool Call: Ingesting {url}")
#     try:
#         loader = WebBaseLoader(url)
#         docs = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         splits = text_splitter.split_documents(docs)
#         vector_store.add_documents(splits)
#         return f"Successfully learned {len(splits)} chunks of information from {url}."
#     except Exception as e:
#         return f"Ingest Error: {str(e)}"

# @mcp.tool()
# @observe(name="query_knowledge_base")
# def query_knowledge_base(query: str) -> str:
#     """Recall information from the internal knowledge base."""
#     logger.info(f"Tool Call: RAG Query for {query}")
#     try:
#         results = vector_store.similarity_search(query, k=3)
#         if not results:
#             return "No relevant info found in memory."
#         context = "\n\n".join([doc.page_content for doc in results])
#         return context
#     except Exception as e:
#         return f"Retrieval Error: {str(e)}"

# if __name__ == "__main__":
#     mcp.run()





import os
import logging
from dotenv import load_dotenv
# Load Env
load_dotenv()
# MCP Imports
from fastmcp import FastMCP

# LangChain & Tools
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langfuse import observe
# --- CHANGE IMPORTS ---
# Remove: from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings # <--- NEW


# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_server")

# 2. Load Env
# load_dotenv()

# 3. Initialize FastMCP Server
mcp = FastMCP("SuperAgent Tools")

# --- RAG SETUP ---
DB_DIR = "./chroma_db_mcp"

# # Initialize Embeddings (OpenRouter Config)
# embeddings = OpenAIEmbeddings(
#     model="openai/text-embedding-3-small", 
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENAI_API_KEY")
# )
# Initialize Embeddings (Runs locally on your CPU)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Vector DB
vector_store = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name="agent_knowledge"
)

# --- TOOLS ---

@mcp.tool()
@observe(name="get_weather")
def get_weather(city: str) -> str:
    """Get the current weather for a specific city."""
    logger.info(f"Tool Call: Weather for {city}")
    return f"The weather in {city} is Sunny, 25°C. Wind 10km/h."

@mcp.tool()
@observe(name="Search_web")
def search_web(query: str) -> str:
    """Search the real-time web using Google (SerpApi)."""
    logger.info(f"Tool Call: Search for {query}")
    try:
        search = SerpAPIWrapper()
        return search.run(query)
    except Exception as e:
        return f"Search Error: {str(e)}"

@mcp.tool()
@observe(name="search_wikipedia")
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for encyclopedic knowledge."""
    logger.info(f"Tool Call: Wikipedia for {query}")
    try:
        wiki = WikipediaAPIWrapper()
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia Error: {str(e)}"

@mcp.tool()
@observe(name="ingest_knowledge")
def ingest_knowledge(url: str) -> str:
    """Read a website and learn its content (Save to Vector DB)."""
    logger.info(f"Tool Call: Ingesting {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(splits)
        return f"Successfully learned {len(splits)} chunks of information from {url}."
    except Exception as e:
        return f"Ingest Error: {str(e)}"

@mcp.tool()
@observe(name="query_knowledge_base")
def query_knowledge_base(query: str) -> str:
    """Recall information from the internal knowledge base."""
    logger.info(f"Tool Call: RAG Query for {query}")
    try:
        results = vector_store.similarity_search(query, k=3)
        if not results:
            return "No relevant info found in memory."
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        return f"Retrieval Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()