from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class Estate(BaseModel):
    """Normalized estate as exposed by the API."""

    id: Optional[str] = Field(None, alias="_id")
    TYPE: Optional[str] = None
    BEDS: Optional[float] = None
    BATH: Optional[float] = None
    PROPERTYSQFT: Optional[float] = None
    ADDRESS: Optional[str] = None
    STATE: Optional[str] = None
    discounted_price: Optional[int] = None
    actual_price: Optional[int] = None
    url_exterior: Optional[str] = None
    url_interior: Optional[str] = None
    description: Optional[str] = None

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "691b9a39a07ee66c4bdd535e",
                "TYPE": "Co-op for sale",
                "BEDS": 3,
                "BATH": 1,
                "PROPERTYSQFT": 325,
                "ADDRESS": "9430 Ridge Blvd Apt 6D",
                "STATE": "Brooklyn, NY 11209",
                "discounted_price": 187500,
                "actual_price": 199000,
                "url_exterior": "https://example.com/exterior.jpg",
                "url_interior": "https://example.com/interior.jpg",
                "description": "The estate at 9430 Ridge Blvd Apt 6D is a co-op with 3 beds, 1 bath, and 325 sqft.",
            }
        }


class SearchFilters(BaseModel):
    beds: Optional[float] = None
    baths: Optional[float] = None
    sqft: Optional[float] = None
    max_price: Optional[int] = None


class PredictRequest(BaseModel):
    type: str
    beds: int
    bath: int
    propertysqft: float
    state: str


class ModelName(str, Enum):
    linear_regression = "linear-regression"
    random_forest = "random-forest"
    knn = "knn"
