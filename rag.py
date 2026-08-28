from functools import lru_cache
from mlx_lm import load, stream_generate
from ingest import embedder, collection

LLM_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
TOP_K = 4
MAX_TOKENS = 512

print("Loading LLM... (first run downloads ~9GB)")
model, tokenizer = load(LLM_MODEL)
print("LLM ready.")


@lru_cache(maxsize=256)
def _embed_cached(query: str):
    """Cache embeddings for repeated/identical queries."""
    return tuple(embedder.encode([query]).tolist()[0])


def retrieve(query):
    embedding = [list(_embed_cached(query))]
    results = collection.query(query_embeddings=embedding, n_results=TOP_K)
    docs = results["documents"][0]
    sources = list({m["source"] for m in results["metadatas"][0]})
    return docs, sources


_SKIP_DOCS_PHRASES = (
    "don't use documents", "dont use documents",
    "don't refer", "dont refer",
    "ignore documents", "ignore docs",
    "without documents", "no documents",
    "from your knowledge", "general knowledge",
    "not from documents", "answer directly",
)


def _wants_general(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _SKIP_DOCS_PHRASES)


def _build_prompt(question, history=None):
    skip_docs = _wants_general(question)

    if skip_docs:
        sources = []
        system = (
            "You are LocalMind, a helpful assistant with broad general knowledge. "
            "Answer the question directly and accurately. "
            "Be concise."
        )
        user_content = question
    else:
        context_chunks, sources = retrieve(question)
        context = "\n\n---\n\n".join(context_chunks)
        system = (
            "You are LocalMind, a helpful personal knowledge assistant. "
            "Use the provided document context to answer questions when relevant. "
            "If the context does not cover the question, answer from your general knowledge. "
            "Be concise and accurate."
        )
        user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_content})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt, sources


def stream_ask(question, history=None):
    """
    Generator yielding (token_text, sources) tuples.
    sources is populated only on the first yield, None thereafter.
    """
    prompt, sources = _build_prompt(question, history)
    first = True
    for response in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=MAX_TOKENS):
        token = response.text if hasattr(response, "text") else response
        yield token, (sources if first else None)
        first = False