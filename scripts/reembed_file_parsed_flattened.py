"""
scripts/reembed_file_parsed_flattened.py

Re-embeds all documents in the `file_parsed_flattened` collection using
OpenRouter's perplexity/pplx-embed-v1-0.6b model (1024-dim).

Run after updating the MongoDB Atlas vector index dimension from 384 → 1024:

    uv run python scripts/reembed_file_parsed_flattened.py

Optional flags:
    --batch-size N   Number of docs to embed per API call (default: 50)
    --dry-run        Print what would be done without writing to MongoDB
    --org-id ORG     Only re-embed documents for the given org_id

Pre-requisites:
    1. Update the 'knowledge_base_rag' Atlas Vector Search index on
       `file_parsed_flattened` to use numDimensions: 1024.
    2. Set OPENROUTER_API_KEY and MONGO_URI in your .env.local file.
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pymongo import MongoClient, UpdateOne

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "perplexity/pplx-embed-v1-0.6b"
DEFAULT_BATCH_SIZE = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


async def embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """Calls OpenRouter embeddings API for a batch of texts."""
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ── Main ──────────────────────────────────────────────────────────────────────

async def reembed(batch_size: int, dry_run: bool, org_id: str | None):
    load_dotenv(".env.local")

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "kita_db")

    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[db_name]
    collection = db["file_parsed_flattened"]

    openrouter = get_openrouter_client()

    # Build query — optionally scope to a single org
    query: dict = {}
    if org_id:
        query["org_id"] = org_id

    total = collection.count_documents(query)
    print(f"Found {total} document(s) in file_parsed_flattened" +
          (f" for org_id={org_id}" if org_id else "") + ".")

    if total == 0:
        print("Nothing to do.")
        return

    processed = 0
    errors = 0
    cursor = collection.find(query, {"_id": 1, "heading_to_text": 1, "text": 1})

    batch_docs = []
    batch_texts = []

    async def flush_batch():
        nonlocal processed, errors
        if not batch_docs:
            return

        try:
            embeddings = await embed_batch(openrouter, batch_texts)
        except Exception as e:
            print(f"  ERROR embedding batch of {len(batch_docs)}: {e}")
            errors += len(batch_docs)
            batch_docs.clear()
            batch_texts.clear()
            return

        if dry_run:
            print(f"  [dry-run] Would update {len(batch_docs)} doc(s) with 1024-dim embeddings.")
        else:
            ops = [
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "embedding": emb,
                        "updated_at": datetime.now(timezone.utc),
                    }}
                )
                for doc, emb in zip(batch_docs, embeddings)
            ]
            collection.bulk_write(ops, ordered=False)
            print(f"  Updated {len(batch_docs)} doc(s). (total so far: {processed + len(batch_docs)})")

        processed += len(batch_docs)
        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        text = doc.get("heading_to_text") or doc.get("text") or ""
        if not text.strip():
            print(f"  SKIP _id={doc['_id']} — no text field.")
            continue

        batch_docs.append(doc)
        batch_texts.append(text)

        if len(batch_docs) >= batch_size:
            await flush_batch()

    # Flush any remaining docs
    await flush_batch()

    mongo_client.close()

    print(f"\nDone. Processed: {processed}, Errors: {errors}, Total: {total}")
    if dry_run:
        print("(dry-run: no writes were performed)")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed file_parsed_flattened with pplx-embed-v1-0.6b")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Docs per API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing to MongoDB")
    parser.add_argument("--org-id", type=str, default=None,
                        help="Only re-embed docs for this org_id")
    args = parser.parse_args()

    asyncio.run(reembed(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        org_id=args.org_id,
    ))
