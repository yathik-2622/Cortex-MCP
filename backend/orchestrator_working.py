import asyncio
import json
import os
import sys
from typing import AsyncGenerator, Callable, Any, Dict, Type

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, create_model, Field
from dotenv import load_dotenv

# --- Modern Agent Imports ---
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

# --- Langfuse ---
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

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")

class ChatRequest(BaseModel):
    query: str

# --- HELPER 1: Convert MCP Schema to Pydantic (THE FIX) ---
def mcp_schema_to_pydantic(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Dynamically converts an MCP JSON Schema into a Pydantic Model.
    This ensures the LLM knows EXACTLY what arguments (city, query, etc.) to send.
    """
    fields = {}
    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})

    for field_name, field_def in properties.items():
        # Determine Python type
        field_type = str
        if field_def.get("type") == "integer":
            field_type = int
        elif field_def.get("type") == "boolean":
            field_type = bool
        
        # Determine if required or optional
        if field_name in required_fields:
            fields[field_name] = (field_type, Field(description=field_def.get("description", "")))
        else:
            fields[field_name] = (field_type | None, Field(default=None, description=field_def.get("description", "")))

    return create_model(f"{name}Input", **fields)

# --- HELPER 2: Tool Execution Wrapper ---
def create_mcp_tool_wrapper(session: ClientSession, tool_name: str) -> Callable:
    async def _tool_wrapper(**kwargs):
        # Unwrap nested kwargs if they exist (Fix for Llama/Groq quirks)
        actual_args = kwargs.get("kwargs", kwargs) if isinstance(kwargs.get("kwargs"), dict) else kwargs
        result = await session.call_tool(tool_name, arguments=actual_args)
        return result.content[0].text
    return _tool_wrapper

async def run_agent_stream(query: str) -> AsyncGenerator[str, None]:
    
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=[SERVER_SCRIPT],   
        env=os.environ.copy()   
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # --- LOAD TOOLS ---
                mcp_tools_list = await session.list_tools()
                langchain_tools = []

                for tool_def in mcp_tools_list.tools:
                    # 1. Create the Function Wrapper
                    tool_wrapper = create_mcp_tool_wrapper(session, tool_def.name)
                    
                    # 2. Create the Argument Schema (CRITICAL STEP)
                    # We look at tool_def.inputSchema and build a Pydantic model
                    arg_schema = mcp_schema_to_pydantic(tool_def.name, tool_def.inputSchema)

                    # 3. Create the Structured Tool with Explicit Schema
                    lc_tool = StructuredTool.from_function(
                        name=tool_def.name,
                        description=tool_def.description,
                        coroutine=tool_wrapper,
                        args_schema=arg_schema # <--- This tells the LLM what args to use
                    )
                    langchain_tools.append(lc_tool)

                # --- AGENT SETUP (Groq) ---
                llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0,
                    streaming=True
                )

                agent_app = create_react_agent(llm, langchain_tools)
                langfuse_handler = CallbackHandler()

                # --- EXECUTION LOOP ---
                input_message = HumanMessage(content=query)
                last_processed_msg_id = None

                async for event in agent_app.astream(
                    {"messages": [input_message]}, 
                    config={"callbacks": [langfuse_handler]},
                    stream_mode="values"
                ):
                    messages = event.get("messages", [])
                    if not messages:
                        continue
                    
                    new_msg = messages[-1]
                    if new_msg.id == last_processed_msg_id:
                        continue
                    last_processed_msg_id = new_msg.id

                    if isinstance(new_msg, AIMessage) and new_msg.tool_calls:
                        for tool_call in new_msg.tool_calls:
                            yield json.dumps({
                                "type": "log",
                                "tool": tool_call['name'],
                                "input": str(tool_call['args']),
                                "message": f"Calling tool: {tool_call['name']}..."
                            }) + "\n"

                    elif isinstance(new_msg, ToolMessage):
                        yield json.dumps({
                            "type": "log",
                            "message": f"Tool Output: {str(new_msg.content)[:200]}..."
                        }) + "\n"
                        
                    elif isinstance(new_msg, AIMessage) and new_msg.content:
                        yield json.dumps({
                            "type": "answer",
                            "content": new_msg.content
                        }) + "\n"

    except Exception as e:
        error_msg = f"Orchestrator Error: {str(e)}"
        if hasattr(e, 'exceptions'):
            error_msg += f"\nSub-errors: {e.exceptions}"
        print(f"CRITICAL ERROR: {error_msg}")
        yield json.dumps({"type": "error", "message": error_msg}) + "\n"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_agent_stream(request.query), 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)