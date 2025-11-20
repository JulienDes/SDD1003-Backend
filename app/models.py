from typing import Optional

from pydantic import BaseModel, Field


class Estate(BaseModel):
    """Normalized estate as exposed by the API."""

    id: Optional[str] = Field(None, alias="_id")
    TYPE: Optional[str] = None
    BEDS: Optional[int] = None
    BATH: Optional[int] = None
    PROPERTYSQFT: Optional[int] = None
    ADDRESS: Optional[str] = None
    STATE: Optional[str] = None
    discounted_price: Optional[int] = None
    actual_price: Optional[int] = None
    url_exterior: Optional[str] = None
    url_interior: Optional[str] = None

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
            }
        }
