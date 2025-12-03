import os
import logging
import shutil
import uvicorn
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# LangChain Imports for RAG and Search
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Logging
# We configure logging to show timestamps and levels (INFO, ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()] # Print logs to the terminal
)
logger = logging.getLogger(__name__)

# 2. Load Environment Variables
load_dotenv()

# Verify API Keys exist
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY is missing in .env file")
if not os.getenv("SERPAPI_API_KEY"):
    logger.error("SERPAPI_API_KEY is missing in .env file")

# 3. Initialize FastAPI App
app = FastAPI(title="Agent Tools API", description="Endpoints for Weather, Search, and RAG")

# --- GLOBAL VARS FOR RAG ---
DB_DIR = "./chroma_db_temp"
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# --- UPDATED EMBEDDINGS FOR OPENROUTER ---
# embeddings = OpenAIEmbeddings(
#     model="openai/text-embedding-3-small", # OpenRouter requires 'provider/model' format
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENAI_API_KEY")
# )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Vector Store (Persistent)
# We use ChromaDB to store our embeddings locally
vector_store = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name="agent_knowledge"
)
logger.info("ChromaDB Vector Store initialized.")


# --- DATA MODELS ---
# These define what the JSON body of your requests should look like

class SearchRequest(BaseModel):
    query: str

class WeatherRequest(BaseModel):
    city: str

class RagIngestRequest(BaseModel):
    url: str

class RagQueryRequest(BaseModel):
    query: str


# --- ENDPOINT 1: SEARCH (SerpApi) ---

@app.post("/tools/search")
async def search_web(request: SearchRequest):
    """
    Uses SerpApi to search Google.
    Input: {"query": "latest news on AI"}
    """
    logger.info(f"Received Search Request for: {request.query}")
    
    try:
        # Initialize wrapper
        search = SerpAPIWrapper()
        
        # Run search
        result = search.run(request.query)
        
        logger.info("Search completed successfully.")
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 2: WEATHER (Mock for stability) ---

@app.post("/tools/weather")
async def get_weather(request: WeatherRequest):
    """
    Mock Weather Endpoint. 
    (Real OpenWeather requires a separate free API key, let me know if you want that specific code).
    Input: {"city": "New York"}
    """
    logger.info(f"Received Weather Request for city: {request.city}")
    
    # Mock Logic for demonstration
    # In production, you would use `requests.get(f'api.openweathermap.org/...?q={request.city}')`
    fake_temp = 24
    fake_condition = "Sunny"
    
    result = f"The weather in {request.city} is {fake_condition} with a temperature of {fake_temp}°C."
    
    logger.info(f"Returning weather: {result}")
    return {"status": "success", "result": result}


# --- ENDPOINT 3A: RAG INGEST (Upload) ---

@app.post("/rag/ingest")
async def ingest_knowledge(request: RagIngestRequest):
    """
    Scrapes a URL, chunks the text, and saves it to ChromaDB.
    Input: {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}
    """
    logger.info(f"Received RAG Ingest Request for URL: {request.url}")
    
    try:
        # 1. Load Data from URL
        logger.info("Loading URL content...")
        loader = WebBaseLoader(request.url)
        docs = loader.load()
        
        # 2. Split Text into manageable chunks
        logger.info("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # 3. Add to Vector Store
        logger.info(f"Adding {len(splits)} chunks to ChromaDB...")
        vector_store.add_documents(splits)
        
        logger.info("Ingestion complete.")
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(splits)} chunks from {request.url}"
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 3B: RAG RETRIEVE (Query) ---

@app.post("/rag/query")
async def query_knowledge(request: RagQueryRequest):
    """
    Searches ChromaDB for context relevant to the query.
    Input: {"query": "What is the history of AI?"}
    """
    logger.info(f"Received RAG Query: {request.query}")
    
    try:
        # Perform Similarity Search (Top 3 results)
        results = vector_store.similarity_search(request.query, k=3)
        
        if not results:
            logger.warning("No relevant documents found in DB.")
            return {"status": "success", "result": "No information found."}
            
        # Combine results into a single string
        context = "\n\n".join([doc.page_content for doc in results])
        
        logger.info("Retrieval successful.")
        return {"status": "success", "result": context}
        
    except Exception as e:
        logger.error(f"RAG Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the API on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)