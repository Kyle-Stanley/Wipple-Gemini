import shutil
import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Import your existing agent
from wip_agent import app as agent_app

# --- FastAPI Setup ---
app = FastAPI()

# Allow your frontend to talk to this
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wipple.ai", 
        "https://www.wipple.ai", 
        "https://wipple-ai.web.app",
        "http://localhost:8000",
        "*" 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SSE Helper Functions ---

def format_sse(event_type: str, data: str) -> str:
    """Formats a message as a Server-Sent Event."""
    # SSE format: event: type\ndata: message\n\n
    return f"event: {event_type}\ndata: {data}\n\n"

async def processing_generator(temp_filename: str):
    """
    This asynchronous generator executes the LangGraph agent 
    and yields real-time status updates and the final JSON payload.
    """
    try:
        # 1. Send Initial Status
        yield format_sse("status", "Agent Activated. Reading PDF...")
        
        # 2. Run the Graph in Streaming Mode
        inputs = {"file_path": temp_filename}
        
        # LangGraph's .stream() method yields chunks for each completed node
        for chunk in agent_app.stream(inputs, stream_mode="updates"):
            
            # Check which node just finished and send an update
            if "extract" in chunk:
                row_count = len(chunk["extract"]["processed_data"])
                yield format_sse("status", f"Extracted {row_count} rows. Validating data and performing calculations...")
                
            if "analyze" in chunk:
                yield format_sse("status", "Risk Analysis complete. Formatting final output.")
                
                # The final JSON is available in the 'analyze' node output
                final_data = chunk["analyze"]["final_json"]
                
                # 3. Send Final Result
                yield format_sse("result", json.dumps(final_data))
                break 

    except Exception as e:
        print(f"STREAMING ERROR: {e}")
        yield format_sse("error", f"An error occurred during analysis: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- API Endpoint ---

@app.post("/analyze-wip-stream")
async def analyze_wip_stream(file: UploadFile = File(...)):
    print(f"--- RECEIVING STREAM REQUEST: {file.filename} ---")
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return StreamingResponse(
        processing_generator(temp_filename), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    # CRITICAL FIX: Use the PORT environment variable provided by Railway
    # Default to 8000 only if running locally
    port = int(os.environ.get("PORT", 8000))
    
    # Host must be 0.0.0.0 to accept external connections
    uvicorn.run(app, host="0.0.0.0", port=port)