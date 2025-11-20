import shutil
import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from google import genai
from google.genai import types

# Import Agents
from wip_agent import app as wip_app
from bond_agent import app as bond_app

# --- Config ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = ""

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-3-pro-preview"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SSE Helper Functions ---

def format_sse(event_type: str, data: str) -> str:
    """Formats a message as a Server-Sent Event."""
    clean_data = data.replace('\n', ' ') 
    return f"event: {event_type}\ndata: {clean_data}\n\n"

async def detect_document_type(file_path: str) -> str:
    """Reads the first chunk of the file to classify it."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(10000) 
            
        prompt = "Classify this document as either 'WIP' (Construction Work in Progress Schedule / Financial Table) or 'BOND' (Legal Bond Form, Contract, or Surety Document). Return ONLY the word 'WIP' or 'BOND'."
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite-preview-02-05",
            contents=[types.Part.from_bytes(data=chunk, mime_type="application/pdf"), prompt]
        )
        result = response.text.strip().upper()
        if "BOND" in result: return "BOND"
        return "WIP"
    except Exception as e:
        print(f"Router Error: {e}")
        return "WIP"

# --- Stream Handlers (Defined before usage) ---

async def run_wip_stream(inputs):
    """Handles the WIP agent stream events."""
    # Note: We iterate synchronously because LangGraph stream is synchronous by default
    # unless using astream, but this loop works within the async generator.
    for chunk in wip_app.stream(inputs, stream_mode="updates"):
        if "extract" in chunk:
            row_count = len(chunk["extract"]["processed_data"])
            yield format_sse("status", f"Extracted {row_count} rows. Validating math...")
        
        if "analyze" in chunk:
            yield format_sse("status", "Analysis complete.")
            final_data = chunk["analyze"]["final_json"]
            payload = {"type": "WIP", "data": final_data}
            yield format_sse("result_wip", json.dumps(payload))

async def run_bond_stream(inputs):
    """Handles the Bond agent stream events."""
    for chunk in bond_app.stream(inputs, stream_mode="updates"):
        # Step 1: Extract
        if "extract" in chunk:
            yield format_sse("status", "Core extraction complete. Searching legal statutes...")
            partial = {
                "parties": chunk["extract"]["parties"].model_dump(),
                "risks": chunk["extract"]["risks"].model_dump()
            }
            yield format_sse("bond_step_1", json.dumps(partial))
            
        # Step 2: Research
        if "research" in chunk:
            yield format_sse("status", "Statute research complete. Synthesizing opinion...")
            stats = [s.model_dump() for s in chunk["research"]["researched_statutes"]]
            yield format_sse("bond_step_2", json.dumps(stats))

        # Step 3: Opinion
        if "opinion" in chunk:
            yield format_sse("status", "Underwriting opinion generated.")
            op = chunk["opinion"]["opinion"].model_dump()
            yield format_sse("bond_step_3", json.dumps(op))
            
            final = chunk["opinion"]["final_json"]
            yield format_sse("result_bond_final", json.dumps(final))

# --- Main Generator ---

async def processing_generator(temp_filename: str):
    """Orchestrates the detection and agent execution."""
    try:
        yield format_sse("status", "Analyzing document structure...")
        
        # 1. ROUTER
        doc_type = await detect_document_type(temp_filename)
        yield format_sse("status", f"Detected document type: {doc_type}")
        
        inputs = {"file_path": temp_filename}

        # 2. BRANCH: WIP
        if doc_type == "WIP":
            yield format_sse("router", "WIP")
            async for chunk in run_wip_stream(inputs):
                yield chunk

        # 3. BRANCH: BOND
        else:
            yield format_sse("router", "BOND")
            async for chunk in run_bond_stream(inputs):
                yield chunk

    except Exception as e:
        print(f"STREAM ERROR: {e}")
        yield format_sse("error", str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- Endpoints ---

class ChatRequest(BaseModel):
    message: str
    context: str 

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Using standard string concatenation to avoid syntax highlighting issues with triple quotes in some editors
        prompt = (
            "You are a helpful Bond & Construction Financial Analyst.\n"
            "The user is asking a question about a document you just analyzed.\n\n"
            f"DOCUMENT CONTEXT:\n{req.context}\n\n"
            f"USER QUESTION:\n{req.message}\n\n"
            "Answer clearly and concisely based strictly on the provided context."
        )
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return {"reply": response.text}
    except Exception as e:
        return {"reply": "I'm having trouble connecting to the chat service right now."}

@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...)):
    print(f"--- RECEIVING FILE: {file.filename} ---")
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return StreamingResponse(
        processing_generator(temp_filename), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
