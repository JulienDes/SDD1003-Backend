import json
import os
from typing import List

from openai import AsyncOpenAI
from pymongo import UpdateOne
from dotenv import load_dotenv
from app.models import SearchFilters

load_dotenv()


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def process_batch(docs_batch, collection):
    """Generate embeddings for a batch of documents and bulk update MongoDB."""

    # Collects document descriptions to prepare a batch embedding request.
    descriptions = [doc.get("description", "") or "" for doc in docs_batch]

    # Requests vector embeddings for all descriptions in a single API call to reduce overhead.
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=descriptions,
    )

    # Préparer les opérations Mongo
    ops: List[UpdateOne] = []

    # Builds a list of MongoDB update operations to attach the generated embeddings to each doc.
    for doc, emb in zip(docs_batch, response.data):

        # Associates each embedding with the corresponding MongoDB document.
        ops.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"embedding": emb.embedding}},
            )
        )

    # Executes all update operations at once to improve performance and reduce network calls.
    result = await collection.bulk_write(ops, ordered=False)
    return result.modified_count


async def get_embedding(text: str) -> List[float]:
    """Returns an embedding for a given text or raises an error if the text is empty."""

    # Ensures that an empty or whitespace-only input does not trigger a useless embedding request.
    if not text or not text.strip():
        raise ValueError("The provided text is empty.")

    # Retrieves a vector embedding representing the semantic meaning of the text.
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    # Returns the numeric embedding array from the model's response.
    return response.data[0].embedding


async def parse_query_llm(query: str) -> dict:
    """
    Uses an LLM to extract structured search filters from a natural-language real-estate query.
    Returns only normalized numerical values or null when missing.
    """

    # Defines assistant behavior to ensure output focuses solely on extraction, not interpretation.
    system_message = (
        "You are a helpful assistant that extracts real estate search criteria."
    )

    # Instructs the model to return strict JSON with the expected schema and parsing constraints.
    system_prompt = (
        "You are an assistant that extracts real estate search criteria from a sentence.\n"
        "You MUST return only a valid JSON object with exactly the following keys:\n"
        "{\n"
        '  "beds": number | null,\n'
        '  "baths": number | null,\n'
        '  "sqft": number | null,\n'
        '  "max_price": integer | null\n'
        "}\n\n"
        "Extraction rules:\n"
        "- beds: number of bedrooms (e.g. '3 beds', '3 bedroom' -> 3).\n"
        "- baths: number of bathrooms, can be decimal (e.g. '1.5 baths' -> 1.5).\n"
        "- sqft: area in square feet/sqft (e.g. '1200 sqft', '1,000 square feet' -> 1200).\n"
        "- max_price: maximum budget in whole US dollars.\n"
        "  - '700k' -> 700000\n"
        "  - '1.2m' -> 1200000\n"
        "  - 'budget 800000' -> 800000\n"
        "- If a piece of information is not present, set its value to null.\n"
        "- Do not include any other keys or text; return only the JSON object.\n\n"
        f"User query to parse: {query}"
    )

    # Sends user instructions and schema definition so the model can produce typed output.
    response = await client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": system_prompt},
        ],
        temperature=0.0,
        text_format=SearchFilters,
    )

    # Retrieves the raw JSON string produced by the model.
    content = response.output_text

    # Loads the JSON object into Python for further normalization.
    data = json.loads(content)

    beds = data.get("beds")
    baths = data.get("baths")
    sqft = data.get("sqft")
    max_price = data.get("max_price")

    # Returns a clean dictionary usable directly for filtering logic.
    return {
        "beds": float(beds) if beds is not None else None,
        "baths": float(baths) if baths is not None else None,
        "sqft": float(sqft) if sqft is not None else None,
        "max_price": int(max_price) if max_price is not None else None,
    }
