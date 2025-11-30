from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from app.database import db
from bson import ObjectId
from app.models import Estate
from app.embeddings.description import build_estate_description

router_crud = APIRouter()


@router_crud.delete("/estates/{estate_id}")
async def delete_estate(estate_id: str) -> Dict[str, Any]:
    """Delete an estate by its Mongo _id."""
    try:
        # Ensure the provided string can represent a Mongo ObjectId,
        # so the API fails early on malformed identifiers.
        if not ObjectId.is_valid(estate_id):
            raise HTTPException(status_code=400, detail="Invalid estate_id")

        object_id = ObjectId(estate_id)
        collection = db["new_york_updated"]

        # Remove the document matching the given ID,
        # guaranteeing we only delete an explicit target.
        result = await collection.delete_one({"_id": object_id})

        if result.deleted_count == 0:
            # Signal that the resource was not found,
            # preserving REST semantics.
            raise HTTPException(status_code=404, detail="Estate not found")

        return {"message": "Estate deleted"}
    except HTTPException:
        raise
    except Exception as e:
        # Wrap unexpected errors to avoid leaking internal details.
        raise HTTPException(status_code=500, detail=str(e))


@router_crud.put("/estates/{estate_id}")
async def update_estate(estate_id: str, estate: Estate) -> Dict[str, Any]:
    """
    Update an estate document by its ID.

    - `estate_id`: MongoDB _id string
    - `estate`: Estate model (partial or full update)

    Returns the updated document.
    """
    try:
        # Validate the identifier early to prevent malformed queries.
        if not ObjectId.is_valid(estate_id):
            raise HTTPException(status_code=400, detail="Invalid estate_id")

        object_id = ObjectId(estate_id)
        collection = db["new_york_updated"]

        # Retrieve the current document to ensure the target exists
        # and to support partial updates via merging.
        existing = await collection.find_one({"_id": object_id})
        if not existing:
            raise HTTPException(status_code=404, detail="Estate not found")

        # Extract only user-provided fields to avoid overwriting unspecified data,
        # enabling clean partial updates.
        incoming = estate.model_dump(by_alias=True, exclude_unset=True)
        update_data = {
            k: v for k, v in incoming.items() if v is not None and k != "_id"
        }

        # Reject updates that bring no actual change,
        # ensuring the API communicates meaningful intent.
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        # Prepare a merged representation of old + new data
        # so the description can be rebuilt from a consistent state.
        existing_for_model = {**existing}
        if "_id" in existing_for_model:
            existing_for_model["_id"] = str(existing_for_model["_id"])

        merged_doc = {**existing_for_model, **incoming}

        # Validate the merged document to maintain schema integrity
        # before committing any change.
        merged_estate = Estate.model_validate(merged_doc)

        # Regenerate the full description so the embedding layer
        # always works with updated and consistent text.
        update_data["description"] = build_estate_description(merged_estate)

        # Apply the update atomically, ensuring only the intended fields change.
        result = await collection.update_one({"_id": object_id}, {"$set": update_data})

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Estate not found")

        # Return the updated version so clients always receive
        # authoritative state directly from storage.
        updated = await collection.find_one({"_id": object_id})
        updated["_id"] = str(updated["_id"])

        return {"item": updated, "updated_fields": list(update_data.keys())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_crud.get("/estates/{estate_id}")
async def get_estate_by_id(estate_id: str) -> Dict[str, Any]:
    """Fetch a single estate document by its Mongo _id."""
    try:
        # Validate the identifier to avoid running unnecessary database operations.
        if not ObjectId.is_valid(estate_id):
            raise HTTPException(status_code=400, detail="Invalid estate_id")

        object_id = ObjectId(estate_id)
        collection = db["new_york_updated"]

        # Retrieve the target document to provide full details to the client.
        doc = await collection.find_one({"_id": object_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Estate not found")

        # Convert _id to string so the payload is JSON-compatible.
        doc["_id"] = str(doc["_id"])
        return {"item": doc}

    except HTTPException:
        raise
    except Exception as e:
        # Convert unexpected failures into a uniform API error.
        raise HTTPException(status_code=500, detail=str(e))


@router_crud.post("/estates", status_code=201)
async def create_estate(estate: Estate) -> Dict[str, Any]:
    """Create a new estate document."""
    try:
        collection = db["new_york_updated"]

        # Transform the Pydantic model into a DB-ready structure,
        # excluding empty values to avoid storing irrelevant fields.
        estate_data = estate.model_dump(by_alias=True, exclude_none=True)

        # Remove user-provided _id to ensure Mongo generates a unique one.
        estate_data.pop("_id", None)

        # Build a synthetic description up front
        # so all downstream embedding operations have consistent input.
        estate_data["description"] = build_estate_description(estate)

        # Insert the new entity and retrieve it to confirm the final stored state.
        result = await collection.insert_one(estate_data)
        created = await collection.find_one({"_id": result.inserted_id})

        if not created:
            # Ensure clients never receive ambiguous outcomes after creation.
            raise HTTPException(
                status_code=500, detail="Failed to fetch created estate"
            )

        # Convert _id to string to ensure JSON compatibility.
        created["_id"] = str(created["_id"])
        return {"item": created}
    except HTTPException:
        raise
    except Exception as e:
        # Catch and surface any unexpected situation in a consistent way.
        raise HTTPException(status_code=500, detail=str(e))
