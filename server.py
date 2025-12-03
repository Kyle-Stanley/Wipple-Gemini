import shutil
import os
import json
import re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Model client for unified model access
from model_client import get_client, get_supported_models, DEFAULT_MODEL, SUPPORTED_MODELS

# Import Agents
from wip_agent import app as wip_app
from bond_agent import app as bond_app

# --- Config ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = ""

app = FastAPI()

# --- CORS FIX ---
origins = [
    "https://wipple.ai",
    "https://www.wipple.ai",
    "http://localhost:3000",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.railway\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# ------------------------

# --- Helpers ---

def format_sse(event_type: str, data: str) -> str:
    """Formats a message as a Server-Sent Event."""
    clean_data = data.replace('\n', ' ') 
    return f"event: {event_type}\ndata: {clean_data}\n\n"

def clean_json_text(text: str) -> str:
    """Strips markdown code blocks and non-JSON prefixes from LLM response."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    text = text.strip()
    return text

async def detect_document_type(file_path: str, model_name: str = DEFAULT_MODEL) -> str:
    """Reads the ENTIRE file and uses the specified model to classify it using JSON."""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read() 
            
        prompt = """
        Look at this document. Is it mainly a TABLE/SPREADSHEET or is it mainly TEXT/PARAGRAPHS?
        
        - If it's dominated by TABLES with rows and columns of data -> classify as "WIP"
        - If it's dominated by TEXT in paragraphs (like a contract or form) -> classify as "BOND"
        
        Return JSON:
        {
            "document_type": "WIP" or "BOND",
            "reasoning": "One sentence: what you see visually"
        }
        """
        
        client = get_client()
        response = client.generate_content(
            prompt=prompt,
            model_name=model_name,
            pdf_bytes=file_bytes,
            response_mime_type="application/json",
        )
        
        # Clean the LLM output before json.loads()
        raw_text = response.text
        cleaned_text = clean_json_text(raw_text)
        
        data = json.loads(cleaned_text)
        doc_type = data.get("document_type", "WIP").upper()
        
        print(f"\nROUTER: {doc_type}")
        print(f"   Why: {data.get('reasoning', 'N/A')}\n")
        
        if doc_type == "BOND":
            return "BOND"
            
        return "WIP"
        
    except Exception as e:
        print(f"ROUTER ERROR (Defaulting to WIP): {e}")
        return "WIP"

# --- Stream Handlers ---

async def run_wip_stream(inputs):
    """Handles the WIP agent stream events."""
    try:
        for chunk in wip_app.stream(inputs, stream_mode="updates"):
            print(f"WIP STREAM CHUNK KEYS: {chunk.keys()}")
            
            if "extract" in chunk:
                extract_data = chunk["extract"]
                processed = extract_data.get("processed_data", [])
                row_count = len(processed) if processed else 0
                yield format_sse("status", f"Extracted {row_count} rows. Validating math...")
            
            if "analyze" in chunk:
                yield format_sse("status", "Analysis complete. Generating summary...")
            
            if "narrative" in chunk:
                yield format_sse("status", "Summary generated. Preparing output...")
            
            if "output" in chunk:
                output_data = chunk["output"]
                final_data = output_data.get("final_json", {})
                
                if "error" in final_data:
                    print(f"WIP WORKFLOW ERROR: {final_data.get('error')}")
                    if "traceback" in final_data:
                        print(f"TRACEBACK: {final_data.get('traceback')}")
                
                payload = {"type": "WIP", "data": final_data}
                yield format_sse("result_wip", json.dumps(payload))
                
    except Exception as e:
        import traceback
        print(f"RUN_WIP_STREAM ERROR: {e}")
        print(traceback.format_exc())
        yield format_sse("error", str(e))

async def run_bond_stream(inputs):
    """Handles the Bond agent stream events."""
    for chunk in bond_app.stream(inputs, stream_mode="updates"):
        # Step 1: Extract
        if "extract" in chunk:
            yield format_sse("status", "Core extraction complete. Searching legal statutes...")
            partial = {
                "parties": chunk["extract"]["parties"].model_dump() if chunk["extract"].get("parties") else {},
                "risks": chunk["extract"]["risks"].model_dump() if chunk["extract"].get("risks") else {}
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
            if chunk["opinion"].get("opinion"):
                op = chunk["opinion"]["opinion"].model_dump()
                yield format_sse("bond_step_3", json.dumps(op))
            
            final = chunk["opinion"]["final_json"]
            yield format_sse("result_bond_final", json.dumps(final))

# --- Main Generator ---

async def processing_generator(temp_filename: str, model_name: str = DEFAULT_MODEL):
    """Orchestrates the detection and agent execution."""
    try:
        # Send selected model info to frontend
        model_config = SUPPORTED_MODELS.get(model_name)
        model_display = model_config.display_name if model_config else model_name
        yield format_sse("model_selected", json.dumps({"model": model_name, "display_name": model_display}))
        
        yield format_sse("status", f"Analyzing document structure with {model_display}...")
        
        # 1. ROUTER
        doc_type = await detect_document_type(temp_filename, model_name)
        
        # Pass model_name to agents via inputs
        inputs = {"file_path": temp_filename, "model_name": model_name}

        # 2. BRANCH: WIP
        if doc_type == "WIP":
            yield format_sse("status", f"Processing WIP schedule with {model_display}...")
            yield format_sse("router", "WIP")
            async for chunk in run_wip_stream(inputs):
                yield chunk

        # 3. BRANCH: BOND
        else:
            yield format_sse("status", f"Processing bond form with {model_display}...")
            yield format_sse("router", "BOND")
            async for chunk in run_bond_stream(inputs):
                yield chunk

    except Exception as e:
        import traceback
        print(f"PROCESSING GENERATOR ERROR: {e}")
        print(traceback.format_exc())
        yield format_sse("error", str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- Endpoints ---

class ChatRequest(BaseModel):
    message: str
    context: str
    model: Optional[str] = DEFAULT_MODEL

@app.get("/models")
async def list_models():
    """Returns list of available models for frontend dropdown."""
    return {"models": get_supported_models(), "default": DEFAULT_MODEL}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        prompt = (
            "You are a helpful Bond & Construction Financial Analyst.\n"
            "The user is asking a question about a document you just analyzed.\n\n"
            f"DOCUMENT CONTEXT:\n{req.context}\n\n"
            f"USER QUESTION:\n{req.message}\n\n"
            "Answer clearly and concisely based strictly on the provided context."
        )
        
        client = get_client()
        response = client.generate_content(
            prompt=prompt,
            model_name=req.model or DEFAULT_MODEL,
        )
        return {"reply": response.text}
    except Exception as e:
        return {"reply": "I'm having trouble connecting to the chat service right now."}

@app.post("/analyze-stream")
async def analyze_stream(
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL)
):
    """
    Main analysis endpoint.
    
    Args:
        file: The PDF document to analyze
        model: Model key from SUPPORTED_MODELS (e.g., "claude-sonnet-4-5", "gemini-3-pro")
    """
    print(f"--- RECEIVING FILE: {file.filename} ---")
    print(f"--- SELECTED MODEL: {model} ---")
    
    # Validate model selection
    if model not in SUPPORTED_MODELS:
        print(f"WARNING: Unknown model '{model}', falling back to {DEFAULT_MODEL}")
        model = DEFAULT_MODEL
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return StreamingResponse(
        processing_generator(temp_filename, model), 
        media_type="text/event-stream"
    )

# BACKWARD COMPATIBILITY
@app.post("/analyze-wip-stream")
async def analyze_wip_stream_legacy(
    file: UploadFile = File(...),
    model: str = Form(default=DEFAULT_MODEL)
):
    """Legacy endpoint - redirects to new unified endpoint"""
    print(f"--- LEGACY ENDPOINT CALLED: {file.filename} ---")
    
    if model not in SUPPORTED_MODELS:
        model = DEFAULT_MODEL
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return StreamingResponse(
        processing_generator(temp_filename, model), 
        media_type="text/event-stream"
    )

# Debug: Print all registered routes
print("\n=== REGISTERED ROUTES ===")
for route in app.routes:
    if hasattr(route, 'methods') and hasattr(route, 'path'):
        print(f"{route.methods} {route.path}")
print("=========================\n")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
