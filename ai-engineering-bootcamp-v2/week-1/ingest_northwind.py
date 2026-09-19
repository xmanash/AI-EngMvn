"""Ingest the Northwind sample corpus through this app's POST /ingest endpoint."""

import argparse
import re
from pathlib import Path

import httpx


DOCUMENT_ID_PATTERN = re.compile(r"^Document ID:\s*(\S+)", re.MULTILINE)


def document_id_for(path: Path, text: str) -> str:
    """Use the corpus' declared ID so reruns upsert the same vector IDs."""

    match = DOCUMENT_ID_PATTERN.search(text)
    return match.group(1) if match else path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path(r"C:\AnothwindsFile"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()

    files = sorted(args.docs_dir.glob("doc*.txt"))
    if not files:
        raise SystemExit(f"No Northwind documents found in {args.docs_dir}")

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        for path in files:
            text = path.read_text(encoding="utf-8")
            document_id = document_id_for(path, text)
            response = client.post(
                "/ingest",
                json={
                    "document_id": document_id,
                    "text": text,
                    "metadata": {"source": path.name},
                },
            )
            response.raise_for_status()
            payload = response.json()
            print(f"{payload['document_id']}: {payload['chunks_indexed']} chunks")

        health = client.get("/health/pinecone")
        health.raise_for_status()
        total_chunks = health.json()["namespace_vector_count"]
        print(f"Total chunks in vector store: {total_chunks}")


if __name__ == "__main__":
    main()
