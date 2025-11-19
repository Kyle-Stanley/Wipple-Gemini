import shutil
import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Import agent and the new chat function
from wip_agent import app as agent_app, chat_with_wip_stream

app = FastAPI()

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage]
    wip_data: Dict[str, Any] # The full JSON context needed for Stateless RAG

# --- SSE Helper Functions ---

def format_sse(event_type: str, data: str) -> str:
    """Formats a message as a Server-Sent Event."""
    return f"event: {event_type}\ndata: {data}\n\n"

async def processing_generator(temp_filename: str):
    """
    Generates SSE events for the analysis process.
    """
    try:
        yield format_sse("status", "Agent Activated. Reading PDF...")
        
        inputs = {"file_path": temp_filename}
        
        for chunk in agent_app.stream(inputs, stream_mode="updates"):
            if "extract" in chunk:
                row_count = len(chunk["extract"]["processed_data"])
                yield format_sse("status", f"Extracted {row_count} rows. Validating data and performing calculations...")
                
            if "analyze" in chunk:
                yield format_sse("status", "Risk Analysis complete. Formatting final output.")
                final_data = chunk["analyze"]["final_json"]
                yield format_sse("result", json.dumps(final_data))
                break 
                
    except Exception as e:
        print(f"STREAMING ERROR: {e}")
        yield format_sse("error", f"An error occurred during analysis: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- Endpoints ---

@app.post("/analyze-wip-stream")
async def analyze_wip_stream(file: UploadFile = File(...)):
    print(f"--- RECEIVING STREAM REQUEST: {file.filename} ---")
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return StreamingResponse(processing_generator(temp_filename), media_type="text/event-stream")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Stateless Chat Endpoint.
    Receives: History + Context (WIP Data) + Query
    Returns: Streaming text response from Gemini 3
    """
    print(f"--- Chat Request: {request.query} ---")
    
    async def stream_generator():
        # Convert Pydantic models to dicts for the agent function
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
        
        # Call the agent generator
        for token in chat_with_wip_stream(history_dicts, request.query, request.wip_data):
            yield token

    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)