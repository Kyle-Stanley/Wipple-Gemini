import shutil
import os
import json
import re
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

# 1. UNIFIED MODEL DEFINITION
MODEL_NAME = "gemini-3-pro-preview"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---

def format_sse(event_type: str, data: str) -> str:
    """Formats a message as a Server-Sent Event."""
    clean_data = data.replace('\n', ' ') 
    return f"event: {event_type}\ndata: {clean_data}\n\n"

def clean_json_text(text: str) -> str:
    """Strips markdown code blocks and non-JSON prefixes from LLM response."""
    # Find JSON fences and extract content
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    # Aggressive cleaning for any leading/trailing whitespace or non-JSON characters
    text = text.strip()
    return text

async def detect_document_type(file_path: str) -> str:
    """Reads the ENTIRE file and uses Gemini 3 Pro to classify it using JSON."""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read() 
            
        # SIMPLE PROMPT: Just visual structure
        prompt = """
        Look at this document. Is it mainly a TABLE/SPREADSHEET or is it mainly TEXT/PARAGRAPHS?
        
        - If it's dominated by TABLES with rows and columns of data → classify as "WIP"
        - If it's dominated by TEXT in paragraphs (like a contract or form) → classify as "BOND"
        
        Return JSON:
        {
            "document_type": "WIP" or "BOND",
            "reasoning": "One sentence: what you see visually"
        }
        """
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                temperature=0.0
            )
        )
        
        # CRITICAL FIX: Clean the LLM output before json.loads()
        raw_text = response.text
        cleaned_text = clean_json_text(raw_text)
        
        data = json.loads(cleaned_text)
        doc_type = data.get("document_type", "WIP").upper()
        
        # SIMPLE LOGGING
        print(f"\n🔍 ROUTER: {doc_type}")
        print(f"   Why: {data.get('reasoning', 'N/A')}\n")
        
        if doc_type == "BOND":
            return "BOND"
            
        return "WIP"
        
    except Exception as e:
        print(f"ROUTER ERROR (Defaulting to WIP): {e}")
        # Note: Printing the raw response helps diagnose the parsing failure
        if 'response' in locals():
            print(f"Raw Response causing failure: {response.text}")
        return "WIP"

# --- Stream Handlers (No changes needed here) ---

async def run_wip_stream(inputs):
    """Handles the WIP agent stream events."""
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

# --- Endpoints (No changes needed here) ---

class ChatRequest(BaseModel):
    message: str
    context: str 

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
