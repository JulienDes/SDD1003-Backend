import re
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query
from app.db.database import db
from app.services.embeddings.embedder import get_embedding, parse_query_llm

router_list = APIRouter(tags=["Listing operations"])

# Common projection used across listing and search endpoints
# to keep responses lightweight and consistent.
PROJECTION_ESTATE = {
    "_id": 1,
    "TYPE": 1,
    "BEDS": 1,
    "BATH": 1,
    "PROPERTYSQFT": 1,
    "ADDRESS": 1,
    "STATE": 1,
    "discounted_price": 1,
    "actual_price": 1,
    "url_exterior": 1,
    "url_interior": 1,
    "description": 1,
}


@router_list.get("/estates")
async def get_estates(
    limit: int = Query(50, ge=1, le=1000),
    skip: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Fetch paginated estates from the `new_york` collection.

    Returns a JSON object with:
      - `items`: the current page of documents
      - `total`: total number of documents in the collection
    """
    try:
        collection = db["new_york_updated"]

        # Count all documents to allow frontend pagination (total pages, etc.)
        total = await collection.count_documents({})

        # Fetch a page of documents using skip/limit for pagination
        cursor = collection.find({}, PROJECTION_ESTATE).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        # Normalize Mongo ObjectId to string so the payload is JSON-serializable
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])

        return {"items": docs, "total": total}

    except Exception as e:
        # Surface unexpected server-side failures as HTTP 500 errors
        raise HTTPException(status_code=500, detail=str(e))


@router_list.get("/autocomplete")
async def get_autocomplete(query: str) -> Dict[str, Any]:
    """
    Provide lightweight suggestions for address autocomplete.

    This endpoint is intended for typeahead in the UI:
      - Matches addresses starting with the input query (case-insensitive)
      - Returns a small subset of fields needed for quick display
    """
    try:
        collection = db["new_york_updated"]

        # Use a prefix regex so suggestions are aligned with user input start
        filter_query = {"ADDRESS": {"$regex": f"^{query}", "$options": "i"}}

        cursor = (
            collection.find(
                filter_query,
                {"ADDRESS": 1, "discounted_price": 1, "STATE": 1},
            )
            # Collation ensures case-insensitive, locale-aware matching/sorting
            .collation({"locale": "en", "strength": 2})
            # Limit suggestions to avoid overloading the client
            .limit(10)
        )

        results = await cursor.to_list(length=10)

        # Normalize ObjectId for client consumption
        for r in results:
            if "_id" in r:
                r["_id"] = str(r["_id"])

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_list.get("/estates/search")
async def search_estates_by_name(address: str) -> Dict[str, Any]:
    """
    Text search on the ADDRESS field (case-insensitive).

    Uses a regex to match the address substring, allowing partial matches.
    Designed for more precise searches than the autocomplete endpoint.
    """
    try:
        collection = db["new_york_updated"]

        # Escape user input to avoid unintended regex patterns
        pattern = f"{re.escape(address)}"

        filter_query = {"ADDRESS": {"$regex": pattern, "$options": "i"}}

        # Count results to support UI pagination or result summaries
        total = await collection.count_documents(filter_query)

        # Use collation for consistent, case-insensitive behavior
        cursor = collection.find(filter_query, PROJECTION_ESTATE).collation(
            {"locale": "en", "strength": 2}
        )

        # Cap to 50 results to avoid returning too many documents at once
        docs = await cursor.to_list(length=50)

        # Normalize ObjectId for client usage
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])

        return {"items": docs, "total": total}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Tunable weights for balancing semantic vs numeric relevance in hybrid search
NUM_CANDIDATES_FACTOR = 20
VECTOR_INDEX_NAME = "vectorDescriptionIndex"
VECTOR_WEIGHT = 0.3  # How much semantic similarity contributes to final score
NUMERIC_WEIGHT = 0.7  # How much numeric / structured criteria contribute


@router_list.get("/estates/hybridSearch")
async def hybrid_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Hybrid search combining:
      - semantic similarity (vector search) on 'embedding'
      - numeric constraints (beds, baths, sqft, price)
      - a combined score mixing vector + numeric similarity

    This endpoint aims to align search results with both:
      - the meaning of the user's free-text query
      - the user's implicit numeric preferences (parsed via LLM)
    """
    try:
        collection = db["new_york_updated"]

        # 1. Embed the free-text query for semantic search
        embedding: List[float] = await get_embedding(query)
        if not embedding:
            raise HTTPException(
                status_code=500, detail="Embedding query failed (empty vector)"
            )

        # 2. Use LLM to extract structured numeric criteria from the query
        criteria: Dict[str, Any] = await parse_query_llm(query)

        beds = criteria.get("beds")
        baths = criteria.get("baths")
        sqft = criteria.get("sqft")
        max_price = criteria.get("max_price")

        # 3. Hard filters (MATCH): constrain the candidate pool numerically
        match_stage: Dict[str, Any] = {}

        # Beds / Baths: allow equal or higher values than requested
        # and a small tolerance below (beds - 1, baths - 1)
        if beds is not None:
            match_stage["BEDS"] = {
                "$gte": beds - 1,
            }

        if baths is not None:
            match_stage["BATH"] = {
                "$gte": baths - 1,
            }

        # Square footage: hard lower bound; we don't want smaller properties
        if sqft is not None:
            match_stage["PROPERTYSQFT"] = {"$gte": sqft}

        # Discounted price: enforce budget as an upper bound
        if max_price is not None:
            match_stage["discounted_price"] = {"$lte": max_price}

        # 4. Build the aggregation pipeline, starting with vector search
        num_candidates = limit * NUM_CANDIDATES_FACTOR

        pipeline: List[Dict[str, Any]] = []

        # 4.1 Vector search stage: broad semantic retrieval
        pipeline.append(
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": num_candidates,
                    "limit": num_candidates,
                }
            }
        )

        # 4.2 Apply numeric filters after vector search to narrow down results
        if match_stage:
            pipeline.append({"$match": match_stage})

        # 5. Compute numeric similarity and combine with vector score
        # Base vector score comes from Atlas search metadata
        add_fields_stage: Dict[str, Any] = {
            "vector_score": {"$meta": "vectorSearchScore"}
        }

        numeric_score_terms: List[Any] = []

        # Beds similarity:
        # - does not penalize if the property has more beds than requested
        # - penalizes when beds < requested (distance-based)
        if beds is not None:
            desired_beds = beds
            add_fields_stage["beds_score"] = {
                "$cond": [
                    {"$ne": ["$BEDS", None]},
                    {
                        # 1 / (1 + max(0, desired - actual))
                        # -> score close to 1 when requirement is met or exceeded
                        "$divide": [
                            1.0,
                            {
                                "$add": [
                                    1,
                                    {
                                        "$max": [
                                            0,
                                            {
                                                "$subtract": [
                                                    desired_beds,
                                                    "$BEDS",
                                                ]
                                            },
                                        ]
                                    },
                                ]
                            },
                        ]
                    },
                    0,
                ]
            }
            numeric_score_terms.append("$beds_score")

        # Baths similarity: same logic as beds
        if baths is not None:
            desired_baths = baths
            add_fields_stage["baths_score"] = {
                "$cond": [
                    {"$ne": ["$BATH", None]},
                    {
                        "$divide": [
                            1.0,
                            {
                                "$add": [
                                    1,
                                    {
                                        "$max": [
                                            0,
                                            {
                                                "$subtract": [
                                                    desired_baths,
                                                    "$BATH",
                                                ]
                                            },
                                        ]
                                    },
                                ]
                            },
                        ]
                    },
                    0,
                ]
            }
            numeric_score_terms.append("$baths_score")

        # Sqft similarity:
        # - penalizes only when property is smaller than desired
        if sqft is not None:
            desired_sqft = sqft
            add_fields_stage["sqft_score"] = {
                "$cond": [
                    {"$ne": ["$PROPERTYSQFT", None]},
                    {
                        "$divide": [
                            1.0,
                            {
                                "$add": [
                                    1,
                                    {
                                        "$max": [
                                            0,
                                            {
                                                "$subtract": [
                                                    desired_sqft,
                                                    "$PROPERTYSQFT",
                                                ]
                                            },
                                        ]
                                    },
                                ]
                            },
                        ]
                    },
                    0,
                ]
            }
            numeric_score_terms.append("$sqft_score")

        # Price similarity:
        # - lower price relative to budget is better
        # - normalized by desired_price to keep score stable across budgets
        if max_price is not None:
            desired_price = float(max_price)
            add_fields_stage["price_score"] = {
                "$cond": [
                    {
                        "$and": [
                            {"$ne": ["$discounted_price", None]},
                            {"$gt": [desired_price, 0]},
                        ]
                    },
                    {
                        # 1 / (1 + (price / budget))
                        # -> higher score for cheaper properties (relative to budget)
                        "$divide": [
                            1.0,
                            {
                                "$add": [
                                    1,
                                    {
                                        "$divide": [
                                            "$discounted_price",
                                            desired_price,
                                        ]
                                    },
                                ]
                            },
                        ]
                    },
                    0,
                ]
            }
            numeric_score_terms.append("$price_score")

        # Aggregate all numeric scores into a single term
        # If no numeric criteria are present, fallback to 0
        if numeric_score_terms:
            add_fields_stage["numeric_score"] = {"$add": numeric_score_terms}
        else:
            add_fields_stage["numeric_score"] = 0

        # Hybrid score balances semantic and numeric relevance
        add_fields_stage["hybrid_score"] = {
            "$add": [
                {"$multiply": ["$vector_score", VECTOR_WEIGHT]},
                {"$multiply": ["$numeric_score", NUMERIC_WEIGHT]},
            ]
        }

        pipeline.append({"$addFields": add_fields_stage})

        # 6. Sort by hybrid score and keep top-N
        pipeline.append({"$sort": {"hybrid_score": -1}})
        pipeline.append({"$limit": limit})

        # 7. Final projection: return estate fields + scoring breakdown
        pipeline.append(
            {
                "$project": {
                    **PROJECTION_ESTATE,
                    "vector_score": 1,
                    "numeric_score": 1,
                    "beds_score": 1,
                    "baths_score": 1,
                    "sqft_score": 1,
                    "price_score": 1,
                    "hybrid_score": 1,
                }
            }
        )

        # 8. Execute the aggregation pipeline
        results = await collection.aggregate(pipeline).to_list(length=limit)

        # Normalize ObjectId for client responses
        for r in results:
            if "_id" in r:
                r["_id"] = str(r["_id"])

        return {"results": results, "count": len(results)}

    except HTTPException:
        # Re-raise explicit HTTP errors as-is
        raise
    except Exception as e:
        # Wrap all unexpected failures as a generic server error
        raise HTTPException(status_code=500, detail=str(e))
