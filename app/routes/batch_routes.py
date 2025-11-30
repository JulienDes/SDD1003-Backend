from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pymongo import UpdateOne

from app.database import db
from app.models import Estate
from app.embeddings.description import build_estate_description
from app.embeddings.embedder import process_batch

router_batch = APIRouter()


@router_batch.post("/estates/updateBatchEmbeddings")
async def update_batch_embeddings(BATCH_SIZE: int = 300) -> Dict[str, Any]:
    """
    Generate embeddings for all estates (based on small_description)
    and update documents in MongoDB in batches.
    """

    try:
        collection = db["new_york_updated"]

        # Find documents missing embeddings entirely or set to null
        filter_query = {
            "$or": [
                {"embedding": {"$exists": False}},
                {"embedding": None},
            ]
        }

        # Only load ID for Mongo update and description for embedding
        projection = {
            "_id": 1,
            "description": 1,
        }

        # Retrieve only required fields to reduce memory and bandwidth usage
        cursor = collection.find(filter_query, projection)

        matched = 0  # Count total documents identified for processing
        updated = 0  # Count documents actually modified in DB

        docs_buffer = []  # Accumulates batch of documents to send to embedder

        async for doc in cursor:
            matched += 1
            docs_buffer.append(doc)  # Collect document until batch is full

            if len(docs_buffer) >= BATCH_SIZE:
                # Perform embedding request and bulk Mongo update in a single operation
                updated += await process_batch(docs_buffer, collection)
                docs_buffer.clear()

        # Handle remaining documents not filling a complete batch
        if docs_buffer:
            updated += await process_batch(docs_buffer, collection)
            docs_buffer.clear()

        return {"matched": matched, "updated": updated}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_batch.post("/estates/updateBatchDescription")
async def update_batch_description(BATCH_SIZE: int = 300) -> Dict[str, Any]:
    """Generate and set missing descriptions for all estates that don't have one."""

    try:

        collection = db["new_york_updated"]

        # Select documents missing description
        filter_query = {
            "$or": [
                {"description": {"$exists": False}},
                {"description": None},
                {"description": ""},
            ]
        }

        # Only required fields loaded to build description
        projection = {
            "_id": 1,
            "ADDRESS": 1,
            "TYPE": 1,
            "BEDS": 1,
            "BATH": 1,
            "PROPERTYSQFT": 1,
            "discounted_price": 1,
            "actual_price": 1,
            "STATE": 1,
        }

        # Query streams over database to avoid loading all results into memory
        cursor = collection.find(filter_query, projection)

        matched = 0  # How many documents matched filter criteria
        updated = 0  # How many documents were actually updated

        batch_ops: List[UpdateOne] = []  # Accumulates Mongo update operations

        async for doc in cursor:
            matched += 1

            # Transform raw DB document into typed model for consistent formatting rules
            estate = Estate.model_validate({**doc, "_id": str(doc["_id"])})

            # Generate description text using domain-specific constructor
            description = build_estate_description(estate)

            # Queue description update for this estate
            batch_ops.append(
                UpdateOne(
                    {"_id": doc["_id"]},  # Target estate record
                    {"$set": {"description": description}},  # Write generated text
                )
            )

            # Execute batch write when enough operations are accumulated
            if len(batch_ops) >= BATCH_SIZE:
                result = await collection.bulk_write(batch_ops, ordered=False)
                updated += result.modified_count  # Track updates for reporting
                batch_ops.clear()

        # Final flush for remaining updates
        if batch_ops:
            result = await collection.bulk_write(batch_ops, ordered=False)
            updated += result.modified_count
            batch_ops.clear()

        return {"matched": matched, "updated": updated}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
