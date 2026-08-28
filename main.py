import json
import os
import asyncio
import threading
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ingest import ingest_file, ingest_folder, ingest_url
from rag import stream_ask

DOCS_DIR = "./docs"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs("./static", exist_ok=True)

app = FastAPI()


# --- Folder watcher ---
class DocHandler(FileSystemEventHandler):
    SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".htm"}

    def on_created(self, event):
        if not event.is_directory:
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext in self.SUPPORTED:
                print(f"New file detected: {event.src_path}")
                ingest_file(event.src_path)

    def on_modified(self, event):
        self.on_created(event)


def start_watcher():
    observer = Observer()
    observer.schedule(DocHandler(), DOCS_DIR, recursive=False)
    observer.start()
    print(f"Watching {DOCS_DIR}/ for new documents...")


# --- API ---
class ChatRequest(BaseModel):
    question: str
    history: list = Field(default_factory=list)


class IngestUrlRequest(BaseModel):
    url: str


@app.post("/chat")
async def chat(req: ChatRequest):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for token, sources in stream_ask(req.question, req.history):
                if sources is not None:
                    payload = json.dumps({"t": token, "s": sources})
                else:
                    payload = json.dumps({"t": token})
                loop.call_soon_threadsafe(queue.put_nowait, f"data: {payload}\n\n")
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait, f"data: {json.dumps({'error': str(e)})}\n\n"
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, "data: [DONE]\n\n")

    threading.Thread(target=run, daemon=True).start()

    async def event_stream():
        while True:
            chunk = await queue.get()
            yield chunk
            if chunk == "data: [DONE]\n\n":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest/url")
async def ingest_url_endpoint(req: IngestUrlRequest):
    try:
        ingest_url(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")
    return {"status": "ok", "url": req.url}


@app.get("/status")
async def status():
    from ingest import collection
    return {"chunks_indexed": collection.count(), "docs_dir": DOCS_DIR}


# --- Serve UI ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# --- Startup ---
@app.on_event("startup")
async def startup():
    print("Ingesting existing docs on startup...")
    ingest_folder(DOCS_DIR)
    threading.Thread(target=start_watcher, daemon=True).start()