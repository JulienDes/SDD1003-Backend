from fastapi import APIRouter, HTTPException, Query
import re
from typing import Any, Dict
from app.database import db
from bson import ObjectId
from app.models import Estate

router = APIRouter()


@router.get("/estates")
async def get_estates(
    limit: int = Query(50, ge=1, le=1000),
    skip: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Fetch paginated estates from the `new_york` collection.

    Returns a JSON object with `items` (list of docs) and `total` (total count).
    """
    try:
        collection = db["new_york_updated"]

        total = await collection.count_documents({})

        cursor = collection.find({}).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        # Convert ObjectId to string for JSON serialization
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])

        return {"items": docs, "total": total}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/autocomplete")
async def get_autocomplete(query: str) -> Dict[str, Any]:

    try:
        collection = db["new_york_updated"]
        filter_query = {"ADDRESS": {"$regex": f"^{query}", "$options": "i"}}

        cursor = (
            collection.find(
                filter_query,
                {"ADDRESS": 1, "discounted_price": 1, "STATE": 1},
            )
            .collation({"locale": "en", "strength": 2})
            .limit(10)
        )

        results = await cursor.to_list(length=10)

        for r in results:
            if "_id" in r:
                r["_id"] = str(r["_id"])

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/estates/search")
async def search_estates_by_name(name: str) -> Dict[str, Any]:
    """Search estates by their `Name` field.

    - `name`: the string to search for
    - `exact`: when true, matches the whole Name exactly (case-insensitive);
      when false, matches anywhere in the Name (case-insensitive).

    Returns a JSON object with `items` (list of docs) and `total` (count).
    """
    try:
        collection = db["new_york_updated"]

        pattern = f"{re.escape(name)}"

        filter_query = {"ADDRESS": {"$regex": pattern, "$options": "i"}}

        total = await collection.count_documents(filter_query)

        cursor = collection.find(filter_query).collation(
            {"locale": "en", "strength": 2}
        )

        docs = await cursor.to_list(length=50)

        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])

        return {"items": docs, "total": total}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/estates/{estate_id}")
async def update_estate(estate_id: str, estate: Estate) -> Dict[str, Any]:
    """
    Update an estate document by its ID.

    - `estate_id`: MongoDB _id string
    - `estate`: Estate model (partial or full update)

    Returns the updated document.
    """
    try:
        # Validation de l'ID
        if not ObjectId.is_valid(estate_id):
            raise HTTPException(status_code=400, detail="Invalid estate_id")

        object_id = ObjectId(estate_id)
        collection = db["new_york_updated"]

        # Extraction des données du modèle
        # On ignore les champs None pour autoriser un update partiel
        update_data = {
            k: v
            for k, v in estate.model_dump(by_alias=True, exclude_unset=True).items()
            if v is not None and k != "_id"
        }

        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        # Update
        result = await collection.update_one({"_id": object_id}, {"$set": update_data})

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Estate not found")

        # Retour du document mis à jour
        updated = await collection.find_one({"_id": object_id})
        updated["_id"] = str(updated["_id"])

        return {"item": updated, "updated_fields": list(update_data.keys())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
