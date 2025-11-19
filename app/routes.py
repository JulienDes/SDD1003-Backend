from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict
from app.database import db

router = APIRouter()


@router.get("/estates")
async def get_estates(
    limit: int = Query(50, ge=1, le=1000), skip: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """Fetch paginated estates from the `new_york` collection.

    Returns a JSON object with `items` (list of docs) and `total` (total count).
    """
    try:
        collection = db["new_york"]
        total = int(collection.count_documents({}))
        cursor = collection.find({}).skip(skip).limit(limit)
        docs = list(cursor)
        # Convert ObjectId to string for JSON serialization
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])
        return {"items": docs, "total": total}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
