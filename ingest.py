import os
import hashlib
import ipaddress
import socket
from urllib.parse import urlparse, urljoin
import requests
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION = "company_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION)


def file_hash(path):
    """Avoid re-ingesting unchanged files."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in (".html", ".htm"):
        with open(path) as f:
            return BeautifulSoup(f.read(), "html.parser").get_text()
    elif ext == ".txt":
        with open(path) as f:
            return f.read()
    return None


def _assert_public_url(url):
    """Reject URLs that resolve to internal/private/loopback addresses (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")

    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from e

    for addr in addrs:
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"Refusing to fetch internal/private address: {addr}")


def ingest_url(url, max_redirects=5):
    """Ingest a web page or Confluence/Notion export URL.

    Follows redirects manually so each hop's destination can be re-checked
    by _assert_public_url (a redirect to a private address must be blocked
    just like a direct request to one).
    """
    current = url
    for _ in range(max_redirects + 1):
        _assert_public_url(current)
        resp = requests.get(current, timeout=10, allow_redirects=False)
        if not resp.is_redirect:
            break
        location = resp.headers.get("Location")
        if not location:
            raise ValueError("Redirect response missing Location header")
        current = urljoin(current, location)
    else:
        raise ValueError("Too many redirects")

    text = BeautifulSoup(resp.text, "html.parser").get_text()
    _store_chunks(text, source=current)
    print(f"  Ingested URL: {current}")


def ingest_file(path):
    fhash = file_hash(path)

    # Skip if already ingested (check by hash in metadata)
    existing = collection.get(where={"hash": fhash})
    if existing["ids"]:
        print(f"  Skipping (unchanged): {path}")
        return

    # Remove old chunks for this file if it changed
    old = collection.get(where={"source": path})
    if old["ids"]:
        collection.delete(ids=old["ids"])

    text = extract_text(path)
    if not text or not text.strip():
        print(f"  Skipping (no text): {path}")
        return

    _store_chunks(text, source=path, fhash=fhash)
    print(f"  Ingested: {path}")


def _store_chunks(text, source, fhash=""):
    chunks = chunk_text(text)
    if not chunks:
        return
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{source}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "hash": fhash, "chunk": i}
                 for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=chunks, metadatas=metadatas)


def ingest_folder(folder="./docs"):
    """Ingest all supported files in a folder."""
    supported = {".pdf", ".docx", ".txt", ".html", ".htm"}
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported:
            ingest_file(os.path.join(folder, fname))