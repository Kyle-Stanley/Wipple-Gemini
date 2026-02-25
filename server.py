# =====================
# server.py (UPDATED)
# =====================
from __future__ import annotations

import os
import json
import re
import asyncio
import shutil
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Any

import anyio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from model_client import (
    get_client,
    get_supported_models,
    get_default_model,
    SUPPORTED_MODELS,
    parse_json_safely,
    is_model_available,
)

from wip_agent import app as wip_app
from bond_agent import app as bond_app

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("server")

# -----------------------
# Thread pool (LangGraph runs sync)
# -----------------------
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKER_THREADS", "4")))

# -----------------------
# Upload / temp config
# -----------------------
UPLOAD_TMP_DIR = os.environ.get("UPLOAD_TMP_DIR", "/tmp/wipple_uploads")
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))  # 25MB
TEMP_FILE_TTL_SECONDS = int(os.environ.get("TEMP_FILE_TTL_SECONDS", str(60 * 60)))  # 1 hour
TEMP_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("TEMP_CLEANUP_INTERVAL_SECONDS", str(10 * 60)))  # 10 min

os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

origins = [
    "https://wipple.ai",
    "https://www.wipple.ai",
    "http://localhost:3000",
    "http://localhost:8000",
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

# -----------------------
# Helpers
# -----------------------

def format_sse(event_type: str, data: str) -> str:
    clean_data = (data or "").replace("\n", " ")
    return f"event: {event_type}\ndata: {clean_data}\n\n"

def _is_pdf_magic(header: bytes) -> bool:
    return header.startswith(b"%PDF-")

async def save_upload_pdf_to_temp(upload: UploadFile, request_id: str) -> str:
    """
    Save upload to a safe temp path (ignores user filename).
    Validates:
      - size limit
      - PDF magic bytes
    """
    temp_path = os.path.join(UPLOAD_TMP_DIR, f"{request_id}.pdf")
    size = 0
    first_bytes = b""

    # Stream read to avoid huge in-memory reads
    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                if not first_bytes:
                    first_bytes = chunk[:8]
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail=f"File too large (>{MAX_UPLOAD_SIZE_BYTES} bytes).")
                out.write(chunk)
    finally:
        try:
            await upload.close()
        except Exception:
            pass

    if not _is_pdf_magic(first_bytes):
        # Clean up
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Invalid file: expected a PDF.")

    return temp_path

def cleanup_temp_dir() -> int:
    """Delete temp files older than TTL. Returns number deleted."""
    now = time.time()
    deleted = 0
    try:
        for name in os.listdir(UPLOAD_TMP_DIR):
            path = os.path.join(UPLOAD_TMP_DIR, name)
            if not os.path.isfile(path):
                continue
            try:
                st = os.stat(path)
                age = now - st.st_mtime
                if age > TEMP_FILE_TTL_SECONDS:
                    os.remove(path)
                    deleted += 1
            except Exception:
                continue
    except Exception:
        return 0
    return deleted

@app.on_event("startup")
async def _startup_cleanup_task():
    async def _loop():
        while True:
            await asyncio.sleep(TEMP_CLEANUP_INTERVAL_SECONDS)
            deleted = await anyio.to_thread.run_sync(cleanup_temp_dir)
            if deleted:
                logger.info("Temp cleanup deleted %s files", deleted)
    asyncio.create_task(_loop())

def normalize_model_or_default(model: Optional[str]) -> str:
    default = get_default_model()
    if not model:
        return default
    if model not in SUPPORTED_MODELS:
        return default
    if not is_model_available(model):
        return default
    return model

def extract_pdf_text_snippet(file_path: str, max_pages: int = 3, max_chars: int = 20000) -> str:
    """
    Extract text from the first N pages of a PDF for routing/classification.
    Avoids loading and base64-encoding the entire PDF for the router.

    Returns "" if extraction fails (scanned PDFs, encrypted, etc.).
    """
    try:
        from pypdf import PdfReader  # optional dependency
        reader = PdfReader(file_path)

        # Best-effort decrypt for empty-password PDFs
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # may fail; that's OK
            except Exception:
                return ""

        parts = []
        total = 0
        page_count = min(max_pages, len(reader.pages))
        for i in range(page_count):
            try:
                t = reader.pages[i].extract_text() or ""
            except Exception:
                t = ""
            if t:
                parts.append(t)
                total += len(t)
                if total >= max_chars:
                    break

        return ("\n".join(parts))[:max_chars]
    except Exception:
        return ""


def heuristic_document_type(snippet: str) -> Optional[str]:
    """
    Fast heuristic classifier. Returns "WIP" / "BOND" / None (uncertain).
    """
    if not snippet:
        return None

    s = snippet.lower()

    bond_terms = [
        "obligee", "principal", "surety", "penal sum", "bond", "performance bond", "payment bond",
        "cancellation", "indemnity", "whereas", "hereby", "rider", "obligor",
    ]
    wip_terms = [
        "work in progress", "wip", "schedule of contracts", "contract price", "cost to date",
        "cost to complete", "billed to date", "earned revenue", "gross profit", "percent complete",
        "retainage", "under bill", "over bill",
    ]

    bond_hits = sum(1 for t in bond_terms if t in s)
    wip_hits = sum(1 for t in wip_terms if t in s)

    digit_ratio = sum(ch.isdigit() for ch in snippet) / max(len(snippet), 1)
    lines = snippet.splitlines()
    table_like_lines = 0
    for line in lines[:250]:
        nums = re.findall(r"\d[\d,]*", line)
        if len(nums) >= 3:
            table_like_lines += 1

    # Strong keyword signals
    if bond_hits >= 3 and wip_hits == 0:
        return "BOND"
    if wip_hits >= 3 and bond_hits == 0:
        return "WIP"

    # Structure / numeric density signals
    if table_like_lines >= 6 or digit_ratio > 0.16:
        return "WIP"
    if bond_hits >= 2 and digit_ratio < 0.08 and table_like_lines <= 2:
        return "BOND"

    return None


def detect_document_type_sync(file_path: str, model_name: str) -> str:
    """
    Sync router (safe to run in a thread).
    Tries:
      1) extract first N pages text and classify via heuristics
      2) if uncertain, ask model using text snippet (no PDF bytes)
      3) if text extraction fails (scanned/encrypted), fall back to PDF-bytes routing
    """
    client = get_client()

    try:
        snippet = extract_pdf_text_snippet(file_path, max_pages=3, max_chars=20000)

        # If we got enough text, try heuristics first
        if snippet and len(snippet) >= 400:
            heur = heuristic_document_type(snippet)
            if heur:
                logger.info("ROUTER(heuristic): %s", heur)
                return heur

            prompt = f"""
            Classify the document based on extracted text (first pages).
            - If it's dominated by table/spreadsheet numeric rows/columns -> "WIP"
            - If it's dominated by contract/form paragraphs -> "BOND"

            Return JSON:
            {{
              "document_type": "WIP" or "BOND",
              "reasoning": "One sentence"
            }}

            EXTRACTED TEXT:
            ```text
            {snippet}
            ```
            """

            response = client.generate_content(
                prompt=prompt,
                model_name=model_name,
                response_mime_type="application/json",
                system_prompt="You are a strict router. Output JSON only.",
            )

            data = parse_json_safely(response.text)
            doc_type = (data.get("document_type") or "WIP").upper()
            logger.info("ROUTER(text+llm): %s | Why: %s", doc_type, data.get("reasoning", "N/A"))
            return "BOND" if doc_type == "BOND" else "WIP"

        # If no text extracted (likely scanned), fall back to routing on the PDF bytes
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

        response = client.generate_content(
            prompt=prompt,
            model_name=model_name,
            pdf_bytes=file_bytes,
            response_mime_type="application/json",
            system_prompt="You are a strict router. Output JSON only.",
        )

        data = parse_json_safely(response.text)
        doc_type = (data.get("document_type") or "WIP").upper()
        logger.info("ROUTER(pdf+llm): %s | Why: %s", doc_type, data.get("reasoning", "N/A"))
        return "BOND" if doc_type == "BOND" else "WIP"

    except Exception as e:
        logger.exception("ROUTER ERROR (Defaulting to WIP): %s", e)
        return "WIP"

# -----------------------
# Thread -> async queue bridge
# -----------------------

class ThreadToAsyncQueue:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue[Optional[Tuple[str, str]]] = asyncio.Queue()

    def put(self, item: Optional[Tuple[str, str]]) -> None:
        # Called from worker thread
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)

# -----------------------
# Stream runners
# -----------------------

def run_wip_sync(inputs: dict, outbox: ThreadToAsyncQueue):
    try:
        for chunk in wip_app.stream(inputs, stream_mode="updates"):
            if "extract" in chunk:
                extract_data = chunk["extract"]
                processed = extract_data.get("processed_data", [])
                row_count = len(processed) if processed else 0
                outbox.put(("status", f"Extracted {row_count} rows. Validating math..."))

            if "analyze" in chunk:
                outbox.put(("status", "Analysis complete. Generating summary..."))

            if "narrative" in chunk:
                outbox.put(("status", "Summary generated. Preparing output..."))

            if "output" in chunk:
                output_data = chunk["output"]
                final_data = output_data.get("final_json", {})
                payload = {"type": "WIP", "data": final_data}
                outbox.put(("result_wip", json.dumps(payload)))

    except Exception as e:
        logger.exception("RUN_WIP_STREAM ERROR: %s", e)
        outbox.put(("error", str(e)))
    finally:
        outbox.put(None)

def run_bond_sync(inputs: dict, outbox: ThreadToAsyncQueue):
    try:
        for chunk in bond_app.stream(inputs, stream_mode="updates"):
            if "extract" in chunk:
                outbox.put(("status", "Core extraction complete. Searching legal statutes..."))
                partial = {
                    "parties": chunk["extract"]["parties"].model_dump() if chunk["extract"].get("parties") else {},
                    "risks": chunk["extract"]["risks"].model_dump() if chunk["extract"].get("risks") else {},
                }
                outbox.put(("bond_step_1", json.dumps(partial)))

            if "research" in chunk:
                outbox.put(("status", "Statute research complete. Synthesizing opinion..."))
                stats = [s.model_dump() for s in chunk["research"]["researched_statutes"]]
                outbox.put(("bond_step_2", json.dumps(stats)))

            if "opinion" in chunk:
                outbox.put(("status", "Underwriting opinion generated."))
                if chunk["opinion"].get("opinion"):
                    op = chunk["opinion"]["opinion"].model_dump()
                    outbox.put(("bond_step_3", json.dumps(op)))

                final = chunk["opinion"]["final_json"]
                outbox.put(("result_bond_final", json.dumps(final)))

    except Exception as e:
        logger.exception("RUN_BOND_STREAM ERROR: %s", e)
        outbox.put(("error", str(e)))
    finally:
        outbox.put(None)

async def run_wip_stream(inputs: dict):
    loop = asyncio.get_running_loop()
    outbox = ThreadToAsyncQueue(loop)

    fut = loop.run_in_executor(executor, run_wip_sync, inputs, outbox)

    # Remove file when worker completes (prevents early deletion on client disconnect)
    def _cleanup(_f: Any):
        path = inputs.get("file_path")
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    fut.add_done_callback(_cleanup)

    while True:
        item = await outbox.queue.get()
        if item is None:
            break
        event_type, data = item
        yield format_sse(event_type, data)

async def run_bond_stream(inputs: dict):
    loop = asyncio.get_running_loop()
    outbox = ThreadToAsyncQueue(loop)

    fut = loop.run_in_executor(executor, run_bond_sync, inputs, outbox)

    def _cleanup(_f: Any):
        path = inputs.get("file_path")
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    fut.add_done_callback(_cleanup)

    while True:
        item = await outbox.queue.get()
        if item is None:
            break
        event_type, data = item
        yield format_sse(event_type, data)

# -----------------------
# Main generator
# -----------------------

async def processing_generator(temp_filename: str, model_name: str):
    try:
        model_config = SUPPORTED_MODELS.get(model_name)
        model_display = model_config.display_name if model_config else model_name
        yield format_sse("model_selected", json.dumps({"model": model_name, "display_name": model_display}))

        yield format_sse("status", f"Analyzing document structure with {model_display}...")

        # Router offloaded to a thread (LLM call is blocking)
        doc_type = await anyio.to_thread.run_sync(detect_document_type_sync, temp_filename, model_name)

        inputs = {"file_path": temp_filename, "model_name": model_name}

        if doc_type == "WIP":
            yield format_sse("status", f"Processing WIP schedule with {model_display}...")
            yield format_sse("router", "WIP")
            async for chunk in run_wip_stream(inputs):
                yield chunk
        else:
            yield format_sse("status", f"Processing bond form with {model_display}...")
            yield format_sse("router", "BOND")
            async for chunk in run_bond_stream(inputs):
                yield chunk

    except Exception as e:
        logger.exception("PROCESSING GENERATOR ERROR: %s", e)
        yield format_sse("error", str(e))
        # Do NOT delete file here; worker cleanup + TTL sweep handle it safely.

# -----------------------
# Endpoints
# -----------------------

class ChatRequest(BaseModel):
    message: str
    context: str
    doc_type: Optional[str] = None
    model: Optional[str] = None

@app.get("/models")
async def list_models():
    default = get_default_model()
    return {"models": get_supported_models(include_unavailable=False), "default": default}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Sync endpoint so LLM call doesn't block the event loop.
    """
    try:
        model_name = normalize_model_or_default(req.model)
        doc_type = (req.doc_type or "").upper().strip()
        doc_hint = f"Document type: {doc_type}\n" if doc_type in {"WIP", "BOND"} else ""

        prompt = (
            "You are a helpful Bond & Construction Financial Analyst.\n"
            "The user is asking a question about a document you just analyzed.\n"
            f"{doc_hint}\n"
            f"DOCUMENT CONTEXT:\n{req.context}\n\n"
            f"USER QUESTION:\n{req.message}\n\n"
            "Answer clearly and concisely based strictly on the provided context."
        )

        client = get_client()
        response = client.generate_content(
            prompt=prompt,
            model_name=model_name,
            system_prompt="Be accurate. If the context doesn't support an answer, say so.",
        )
        return {"reply": response.text}
    except Exception:
        return {"reply": "I'm having trouble connecting to the chat service right now."}

@app.post("/analyze-stream")
async def analyze_stream(
    file: UploadFile = File(...),
    model: str = Form(default=""),
):
    request_id = uuid.uuid4().hex[:12]
    model_name = normalize_model_or_default(model)

    logger.info("RECEIVING FILE name=%s model=%s request_id=%s", file.filename, model_name, request_id)

    # Basic content-type check (not authoritative but helpful)
    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Invalid content type. Please upload a PDF.")

    temp_path = await save_upload_pdf_to_temp(file, request_id)

    return StreamingResponse(
        processing_generator(temp_path, model_name),
        media_type="text/event-stream",
    )

# BACKWARD COMPATIBILITY
@app.post("/analyze-wip-stream")
async def analyze_wip_stream_legacy(
    file: UploadFile = File(...),
    model: str = Form(default=""),
):
    # Legacy endpoint uses same unified pipeline now
    return await analyze_stream(file=file, model=model)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
