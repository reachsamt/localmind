# LocalMind

A fully local, privacy-first RAG (Retrieval-Augmented Generation) assistant that runs entirely on your machine. Drop in your documents and chat with them — no cloud, no API keys, no data leaving your device.

Built for Apple Silicon Macs using [MLX](https://github.com/ml-explore/mlx) for fast on-device LLM inference.

---

## Features

- **100% local** — LLM inference, embeddings, and vector storage all run on your machine
- **Document chat** — ingest PDFs, Word docs, plain text, and HTML pages
- **Live document watching** — drop a file into `docs/` and it's indexed automatically
- **Streaming responses** — see the answer as it's generated, token by token
- **General knowledge fallback** — asks outside your documents are answered from the model's own knowledge
- **Conversation history** — maintains context across follow-up questions
- **Web UI** — clean chat interface served at `localhost:8000`

---

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4) — required for MLX inference
- Python 3.10+
- ~10 GB free disk space (for the default 14B model)
- 16 GB unified memory recommended (8 GB works with the 7B model)

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/reachsamt/localmind.git
cd localmind

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your documents (PDF, DOCX, TXT, HTML)
cp your-file.pdf docs/

# 5. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Supported Document Types

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |
| Plain text | `.txt` |
| HTML | `.html`, `.htm` |
| Web URL | via API (see below) |

---

## Configuration

All settings live at the top of the relevant source files. No `.env` needed.

**Model (`rag.py`)**

```python
LLM_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"  # ~9 GB
# Faster/lighter alternative for 8 GB machines:
# LLM_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"  # ~5 GB
```

**Retrieval (`rag.py`)**

```python
TOP_K = 4        # number of document chunks retrieved per query
MAX_TOKENS = 512  # max tokens generated per response
```

**Chunking (`ingest.py`)**

```python
CHUNK_SIZE = 500    # characters per chunk
CHUNK_OVERLAP = 50  # overlap between adjacent chunks
```

---

## Security notes

- The Quickstart binds the server to `0.0.0.0:8000`, meaning anyone on your local network can reach it. There is no authentication on any endpoint. If you want to expose LocalMind beyond `localhost`, put it behind a reverse proxy with auth, or change the bind host to `127.0.0.1`.
- `POST /ingest/url` fetches whatever URL you give it from the machine running the server. It refuses to fetch internal/private/loopback addresses (e.g. `localhost`, `169.254.169.254`, RFC1918 ranges) to reduce SSRF risk, but treat this endpoint as trusted-input-only — don't expose it to untrusted users.

---

## Ingesting a URL

```bash
curl -X POST http://localhost:8000/ingest/url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/page"}'
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Stream a chat response (SSE) |
| `GET` | `/status` | Returns chunk count and docs directory |
| `GET` | `/` | Web UI |

**POST /chat** body:
```json
{
  "question": "What is the refund policy?",
  "history": []
}
```

Response is a Server-Sent Events stream. Each event:
```
data: {"t": "token text"}          # streamed token
data: {"t": "...", "s": ["file"]}  # first token also carries sources
data: [DONE]                       # end of stream
```

---

## Bypassing document context

Prefix your message with any of these phrases to skip RAG and answer from general knowledge:

> "don't use documents, ..."  
> "from your knowledge, ..."  
> "answer directly: ..."  
> "ignore docs, ..."

---

## Project Structure

```
localmind/
├── main.py          # FastAPI server, file watcher, SSE endpoint
├── rag.py           # LLM loading, retrieval, prompt building, streaming
├── ingest.py        # Text extraction, chunking, embedding, ChromaDB storage
├── static/
│   └── index.html   # Web chat UI
├── docs/            # Drop your documents here
├── chroma_db/       # Auto-created vector store (gitignored)
├── requirements.txt
└── LICENSE
```

---

## Contributing

Contributions are welcome. Please open an issue before submitting large PRs so we can discuss the approach.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a pull request

---

## License

Apache 2.0 — free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.
