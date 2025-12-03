import asyncio
import json
import os
import sys
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Modern Agent Imports (LangGraph & LangChain) ---
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool

# --- Langfuse v3 Import ---
from langfuse.langchain import CallbackHandler 

# --- MCP Imports ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 1. Load Environment
load_dotenv()

# 2. Setup FastAPI
app = FastAPI(title="Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Server Script Path
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")

class ChatRequest(BaseModel):
    query: str

async def run_agent_stream(query: str) -> AsyncGenerator[str, None]:
    """
    Connects to MCP Server -> Loads Tools -> Runs Agent (LangGraph) -> Streams Logs
    """
    
    # Define connection parameters for the MCP Server
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=[SERVER_SCRIPT],   
        env=os.environ.copy()   
    )

    # Start the MCP Client
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # --- DYNAMIC TOOL LOADING ---
            mcp_tools_list = await session.list_tools()
            langchain_tools = []

            for tool_def in mcp_tools_list.tools:
                # Async wrapper for the tool
                async def call_mcp_tool(*args, **kwargs):
                    tool_args = kwargs if kwargs else args[0] if args else {}
                    result = await session.call_tool(tool_def.name, arguments=tool_args)
                    return result.content[0].text

                lc_tool = Tool(
                    name=tool_def.name,
                    description=tool_def.description,
                    func=None, 
                    coroutine=call_mcp_tool 
                )
                langchain_tools.append(lc_tool)

            # --- AGENT SETUP ---
            # KEY FIX: streaming=False to prevent the "Tools not supported" error
            llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct:free", 
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
                streaming=False 
            )

            agent_app = create_react_agent(llm, langchain_tools)

            # --- LANGFUSE SETUP ---
            langfuse_handler = CallbackHandler()

            # --- EXECUTION LOOP (OPTIMIZED) ---
            input_message = HumanMessage(content=query)
            
            final_answer_content = ""

            # We run the agent ONCE here and capture logs + answer as they happen
            async for event in agent_app.astream_events(
                {"messages": [input_message]}, 
                version="v1",
                config={"callbacks": [langfuse_handler]}
            ):
                kind = event["event"]
                
                # 1. Log Tool Decisions
                if kind == "on_tool_start":
                    log_data = {
                        "type": "log",
                        "tool": event['name'],
                        "input": str(event['data'].get('input')),
                        "message": f"Decided to use {event['name']}..."
                    }
                    yield json.dumps(log_data) + "\n"

                # 2. Log Tool Outputs
                elif kind == "on_tool_end":
                    if event['name'] != '__start__': 
                        log_data = {
                            "type": "log",
                            "message": f"Tool Output: {str(event['data'].get('output'))[:200]}..." 
                        }
                        yield json.dumps(log_data) + "\n"
                
                # 3. Capture Final Answer (When the LLM finishes speaking)
                elif kind == "on_chat_model_end":
                    output = event['data']['output']
                    # If the LLM output has content (text) and NO tool calls, it's the answer.
                    if isinstance(output, AIMessage) and output.content:
                        # We accumulate or overwrite the final answer (ReAct agents end with text)
                        final_answer_content = output.content

            # 4. Send the Final Answer to the Frontend
            yield json.dumps({
                "type": "answer",
                "content": final_answer_content
            }) + "\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_agent_stream(request.query), 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    import uvicorn
    # Forced to Port 8002
    uvicorn.run(app, host="0.0.0.0", port=8002)