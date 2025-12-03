# import asyncio
# import json
# import os
# import sys
# from typing import AsyncGenerator

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from dotenv import load_dotenv

# # --- Modern Agent Imports (LangGraph) ---
# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# # --- Langfuse v3 ---
# from langfuse.langchain import CallbackHandler 

# # --- MCP Imports ---
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langchain_core.tools import Tool

# # 1. Load Environment
# load_dotenv()

# # 2. Setup FastAPI
# app = FastAPI(title="Orchestrator")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")

# class ChatRequest(BaseModel):
#     query: str

# async def run_agent_stream(query: str) -> AsyncGenerator[str, None]:
#     """
#     Robust Method: Stream the 'Values' (Full State) of the conversation.
#     This ensures we capture the final answer even if the model doesn't stream tokens.
#     """
    
#     # --- MCP SERVER CONNECTION ---
#     server_params = StdioServerParameters(
#         command=sys.executable, 
#         args=[SERVER_SCRIPT],   
#         env=os.environ.copy()   
#     )

#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
            
#             # --- LOAD TOOLS ---
#             mcp_tools_list = await session.list_tools()
#             langchain_tools = []

#             for tool_def in mcp_tools_list.tools:
#                 async def call_mcp_tool(*args, **kwargs):
#                     tool_args = kwargs if kwargs else args[0] if args else {}
#                     result = await session.call_tool(tool_def.name, arguments=tool_args)
#                     return result.content[0].text

#                 lc_tool = Tool(
#                     name=tool_def.name,
#                     description=tool_def.description,
#                     func=None, 
#                     coroutine=call_mcp_tool 
#                 )
#                 langchain_tools.append(lc_tool)

#             # --- AGENT SETUP ---
#             # Using Gemini 2.0 Flash (Recommended for Tools) or Llama 3.3
#             llm = ChatOpenAI(
#                 model="openai/gpt-oss-20b:free", #google/gemini-2.0-flash-exp:free
#                 base_url="https://openrouter.ai/api/v1",
#                 api_key=os.getenv("OPENAI_API_KEY"),
#                 temperature=0,
#                 streaming=False 
#             )

#             agent_app = create_react_agent(llm, langchain_tools)
#             langfuse_handler = CallbackHandler()

#             # --- EXECUTION LOOP (VALUES MODE) ---
#             input_message = HumanMessage(content=query)
            
#             # Keep track of the last message we processed to avoid duplicates
#             last_processed_msg_id = None

#             # stream_mode="values" yields the ENTIRE list of messages at every step.
#             async for event in agent_app.astream(
#                 {"messages": [input_message]}, 
#                 config={"callbacks": [langfuse_handler]},
#                 stream_mode="values"
#             ):
#                 messages = event.get("messages", [])
#                 if not messages:
#                     continue
                
#                 # We only care about the very last message in the chain
#                 new_msg = messages[-1]

#                 # Skip if we've already processed this exact message object
#                 if new_msg.id == last_processed_msg_id:
#                     continue
#                 last_processed_msg_id = new_msg.id

#                 # LOGIC: Check what kind of message it is
                
#                 # 1. It is from the AI
#                 if isinstance(new_msg, AIMessage):
#                     # A. AI is using a Tool
#                     if new_msg.tool_calls:
#                         for tool_call in new_msg.tool_calls:
#                             log_data = {
#                                 "type": "log",
#                                 "tool": tool_call['name'],
#                                 "input": str(tool_call['args']),
#                                 "message": f"Calling tool: {tool_call['name']}..."
#                             }
#                             yield json.dumps(log_data) + "\n"
                    
#                     # B. AI is giving the Final Answer (Content exists and No Tools)
#                     elif new_msg.content:
#                         yield json.dumps({
#                             "type": "answer",
#                             "content": new_msg.content
#                         }) + "\n"

#                 # 2. It is a Tool Output
#                 elif isinstance(new_msg, ToolMessage):
#                     log_data = {
#                         "type": "log",
#                         "message": f"Tool Output: {str(new_msg.content)[:200]}..."
#                     }
#                     yield json.dumps(log_data) + "\n"


# @app.post("/api/chat")
# async def chat(request: ChatRequest):
#     return StreamingResponse(
#         run_agent_stream(request.query), 
#         media_type="application/x-ndjson"
#     )

# if __name__ == "__main__":
#     import uvicorn
#     # Forced to Port 8002
#     uvicorn.run(app, host="0.0.0.0", port=8002)





import asyncio
import json
import os
import sys
from typing import AsyncGenerator, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Modern Agent Imports (LangGraph) ---
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

# --- Langfuse v3 ---
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

# --- HELPER: Fixes the Python Loop Bug ---
def create_mcp_tool_wrapper(session: ClientSession, tool_name: str) -> Callable:
    async def _tool_wrapper(**kwargs):
        # Unwrap 'kwargs' if nested (common LLM quirk)
        actual_args = kwargs.get("kwargs", kwargs) if isinstance(kwargs.get("kwargs"), dict) else kwargs
        result = await session.call_tool(tool_name, arguments=actual_args)
        return result.content[0].text
    return _tool_wrapper

async def run_agent_stream(query: str) -> AsyncGenerator[str, None]:
    """
    Streams agent state. Includes TRY/EXCEPT to catch server crashes gracefully.
    """
    # Define Server Parameters
    server_params = StdioServerParameters(
        command=sys.executable, 
        args=[SERVER_SCRIPT],   
        env=os.environ.copy()   
    )

    try:
        # Start MCP Client
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # --- LOAD TOOLS ---
                mcp_tools_list = await session.list_tools()
                langchain_tools = []

                for tool_def in mcp_tools_list.tools:
                    # Create wrapper using helper to avoid loop variable bug
                    tool_wrapper = create_mcp_tool_wrapper(session, tool_def.name)
                    
                    lc_tool = StructuredTool.from_function(
                        name=tool_def.name,
                        description=tool_def.description,
                        coroutine=tool_wrapper, 
                    )
                    langchain_tools.append(lc_tool)

                # --- AGENT SETUP ---
                llm = ChatOpenAI(
                    model="mistralai/mistral-7b-instruct:free", 
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0,
                    streaming=False 
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

                    # 1. Yield Logs (AI Thinking)
                    if isinstance(new_msg, AIMessage) and new_msg.tool_calls:
                        for tool_call in new_msg.tool_calls:
                            yield json.dumps({
                                "type": "log",
                                "tool": tool_call['name'],
                                "input": str(tool_call['args']),
                                "message": f"Calling tool: {tool_call['name']}..."
                            }) + "\n"

                    # 2. Yield Tool Outputs
                    elif isinstance(new_msg, ToolMessage):
                        yield json.dumps({
                            "type": "log",
                            "message": f"Tool Output: {str(new_msg.content)[:200]}..."
                        }) + "\n"
                        
                    # 3. Yield Final Answer
                    elif isinstance(new_msg, AIMessage) and new_msg.content:
                        yield json.dumps({
                            "type": "answer",
                            "content": new_msg.content
                        }) + "\n"
    except Exception as e:
        # UNWRAP THE ERROR to see why the TaskGroup failed
        error_msg = f"Orchestrator Error: {str(e)}"
        
        # If it's a TaskGroup error, the real error is in the 'exceptions' list
        if hasattr(e, 'exceptions'):
            error_msg += f"\nSub-errors: {e.exceptions}"
        
        print(f"CRITICAL ERROR: {error_msg}") 
        
        yield json.dumps({
            "type": "error", 
            "message": error_msg
        }) + "\n"

    # except Exception as e:
    #     # CATCH CRASHES: Send error to client instead of killing connection
    #     error_msg = f"Orchestrator Error: {str(e)}"
    #     print(f"CRITICAL ERROR: {error_msg}") # Print to server terminal
    #     yield json.dumps({
    #         "type": "error", 
    #         "message": error_msg
    #     }) + "\n"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_agent_stream(request.query), 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)