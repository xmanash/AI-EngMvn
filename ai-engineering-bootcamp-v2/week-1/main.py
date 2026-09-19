"""Week 1 live demo — five stages in one file, built up live in class."""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Keep the model in one place: every vector written or queried uses this model.
EMBEDDING_MODEL = "text-embedding-3-small"
# Match the Pinecone index dimension. text-embedding-3-small supports 256–1536.
EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024"))
RETRIEVAL_TOP_K = 5

GROUNDING_PROMPT_TEMPLATE = """\
Answer the question using only the context below.

Rules:
- Do not use knowledge outside the context.
- Cite the document_id in square brackets for every chunk you use, for example [handbook].
- If the context is insufficient, say "I don't have enough context to answer that question."
  Do not guess, and set sources_needed to true.
- Return the requested structured response.

Question:
{question}

Context:
{context}
"""


@app.get("/")
def root() -> dict:
    """Browser-friendly landing so visiting the Render URL is not a 404."""
    return {
        "service": "week-1 /ask",
        "docs": "/docs",
        "health": "/health",
        "pinecone_health": "/health/pinecone",
        "debug_retrieve": "GET /debug/retrieve?q=your-question",
        "ask": "POST /ask with JSON {question, model?, force_bad?}",
        "ingest": "POST /ingest with JSON {text, document_id, metadata?}",
        "vector_upsert": "POST /vectors/upsert",
        "vector_query": "POST /vectors/query",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

class VectorDocument(BaseModel):
    """One text record to embed and store in Pinecone."""

    id: str | None = None
    text: str = Field(min_length=1)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class VectorUpsertRequest(BaseModel):
    documents: list[VectorDocument] = Field(min_length=1)


class VectorQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class IngestRequest(BaseModel):
    """A document to split, embed, and store as individual vector chunks."""

    document_id: str = Field(min_length=1)
    text: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


def _pinecone_settings() -> tuple[Pinecone, str, str]:
    """Create a Pinecone index client from environment-only configuration."""

    # Hosted dashboards can accidentally preserve pasted line endings.
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    namespace = os.getenv("PINECONE_NAMESPACE", "default").strip() or "default"
    if not api_key or not index_name:
        raise HTTPException(
            status_code=500,
            detail="PINECONE_API_KEY and PINECONE_INDEX_NAME must be configured.",
        )
    return Pinecone(api_key=api_key), index_name, namespace


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text with the same model for both Pinecone writes and searches."""

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def _retrieve_chunks(question: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict[str, Any]]:
    """Embed a question and return its highest-scoring Pinecone chunks."""

    pinecone, index_name, namespace = _pinecone_settings()
    query_embedding = _embed_texts([question])[0]
    results = pinecone.Index(index_name).query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    chunks = []
    for match in results.matches:
        metadata = match.metadata or {}
        chunks.append(
            {
                "id": match.id,
                "score": match.score,
                "document_id": str(metadata.get("document_id", "unknown")),
                "text": str(metadata.get("text", "")),
                "metadata": metadata,
            }
        )
    return chunks


def _build_grounding_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into the context-only generation prompt."""

    context = "\n\n".join(
        f"[chunk_id={chunk['id']} | document_id={chunk['document_id']}]\n{chunk['text']}"
        for chunk in chunks
    )
    return GROUNDING_PROMPT_TEMPLATE.format(
        question=question,
        context=context or "(No relevant context was retrieved.)",
    )


def _ingest_splitter() -> RecursiveCharacterTextSplitter:
    """Build the chunker from environment configuration."""

    try:
        chunk_size = int(os.getenv("INGEST_CHUNK_SIZE", "800"))
        chunk_overlap = int(os.getenv("INGEST_CHUNK_OVERLAP", "100"))
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="INGEST_CHUNK_SIZE and INGEST_CHUNK_OVERLAP must be integers.",
        ) from exc
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=500,
            detail="INGEST_CHUNK_SIZE must be positive and exceed INGEST_CHUNK_OVERLAP.",
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def pinecone_health() -> dict[str, Any]:
    """Confirm both Pinecone's control plane and selected index are reachable."""

    pinecone, index_name, namespace = _pinecone_settings()
    index_description = pinecone.describe_index(index_name)
    stats = pinecone.Index(index_name).describe_index_stats()

    status = getattr(index_description, "status", {})
    if hasattr(status, "to_dict"):
        status = status.to_dict()
    namespaces = getattr(stats, "namespaces", {})
    if hasattr(namespaces, "to_dict"):
        namespaces = namespaces.to_dict()
    namespace_stats = namespaces.get(namespace, {}) if isinstance(namespaces, dict) else {}
    if hasattr(namespace_stats, "to_dict"):
        namespace_stats = namespace_stats.to_dict()

    return {
        "status": "ok",
        "index": index_name,
        "namespace": namespace,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "index_ready": status.get("ready") if isinstance(status, dict) else None,
        "namespace_vector_count": namespace_stats.get("vector_count", 0),
    }


@app.get("/health/pinecone")
def pinecone_health_endpoint() -> dict[str, Any]:
    """HTTP wrapper for `pinecone_health`, suitable for Render diagnostics."""

    try:
        return pinecone_health()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pinecone health check failed")
        raise HTTPException(
            status_code=503, detail="Pinecone index is unreachable or unavailable."
        ) from exc


# curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" \
#   -d "{\"document_id\":\"readme-v1\",\"text\":\"Document text...\",\"metadata\":{\"source\":\"README.md\"}}"
@app.post("/ingest")
def ingest_document(body: IngestRequest) -> dict[str, str | int]:
    """Chunk, embed, and upsert one document into the configured namespace."""

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty or whitespace.")

    try:
        chunks = _ingest_splitter().split_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="text did not produce any indexable chunks.")

        pinecone, index_name, namespace = _pinecone_settings()
        embeddings = _embed_texts(chunks)
        source = str(body.metadata.get("source", ""))
        vectors = [
            {
                "id": f"{body.document_id}:{chunk_index}",
                "values": embedding,
                "metadata": {
                    **body.metadata,
                    "document_id": body.document_id,
                    "chunk_index": chunk_index,
                    "source": source,
                    "text": chunk,
                },
            }
            for chunk_index, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=True)
            )
        ]
        pinecone.Index(index_name).upsert(vectors=vectors, namespace=namespace)
        return {
            "document_id": body.document_id,
            "chunks_indexed": len(vectors),
            "status": "indexed",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to ingest document.") from exc


@app.post("/vectors/upsert")
def upsert_vectors(body: VectorUpsertRequest) -> dict[str, Any]:
    """Embed documents and upsert them into the configured Pinecone namespace."""

    try:
        pinecone, index_name, namespace = _pinecone_settings()
        embeddings = _embed_texts([document.text for document in body.documents])
        vector_ids = [document.id or str(uuid.uuid4()) for document in body.documents]
        vectors = [
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {**document.metadata, "text": document.text},
            }
            for vector_id, document, embedding in zip(
                vector_ids, body.documents, embeddings, strict=True
            )
        ]
        pinecone.Index(index_name).upsert(vectors=vectors, namespace=namespace)
        return {
            "upserted_count": len(vectors),
            "ids": vector_ids,
            "namespace": namespace,
            "embedding_model": EMBEDDING_MODEL,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to embed or upsert vectors.") from exc


@app.get("/debug/retrieve")
def debug_retrieve(q: str = Query(min_length=1)) -> dict[str, Any]:
    """Retrieve the top five chunks for a question without calling an LLM."""

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="q must not be empty or whitespace.")

    try:
        matches = _retrieve_chunks(question)
        return {
            "question": question,
            "top_k": RETRIEVAL_TOP_K,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "matches": matches,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to retrieve vectors.") from exc


@app.post("/vectors/query")
def query_vectors(body: VectorQueryRequest) -> dict[str, Any]:
    """Embed a question and return its nearest Pinecone matches."""

    try:
        pinecone, index_name, namespace = _pinecone_settings()
        query_embedding = _embed_texts([body.question])[0]
        results = pinecone.Index(index_name).query(
            vector=query_embedding,
            top_k=body.top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return {
            "matches": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {},
                }
                for match in results.matches
            ],
            "namespace": namespace,
            "embedding_model": EMBEDDING_MODEL,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to embed or query vectors.") from exc


# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    retrieved_chunk_ids: list[str]
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Retrieve context, then answer with structured output, guardrails, and cost visibility."""

    model = body.model or DEFAULT_MODEL
    last_error: str | None = None
    try:
        retrieved_chunks = _retrieve_chunks(body.question)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to retrieve context.") from exc

    grounding_prompt = _build_grounding_prompt(body.question, retrieved_chunks)
    retrieved_chunk_ids = [str(chunk["id"]) for chunk in retrieved_chunks]

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            start = time.perf_counter()

            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounding_prompt, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounding_prompt, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            return AskResponse(
                answer=answer,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
