from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class PropertyType(str, Enum):
    Studio = "Studio"
    CondoApartment = "Condo / Apartment"
    Loft = "Loft"
    TownhouseRowHouse = "Townhouse / Row House"
    SingleFamilyHome = "Single-Family Home"
    LuxuryHome = "Luxury Home"


class Estate(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    Name: Optional[str] = Field(None, alias="Name")
    City: Optional[str] = Field(None, alias="City")
    Neighbourhood: Optional[str] = Field(None, alias="Neighbourhood")
    Year_Built: Optional[str] = Field(None, alias="Year Built")
    Beds: Optional[str] = Field(None, alias="Beds")
    Baths: Optional[str] = Field(None, alias="Baths")
    AC: Optional[str] = Field(None, alias="A/C")
    area: Optional[int] = Field(None, alias="area")
    actual_price: Optional[int] = Field(None, alias="actual_price")
    discounted_price: Optional[int] = Field(None, alias="discounted_price")
    url_exterior: Optional[str] = Field(None, alias="url_exterior")
    url_interior: Optional[str] = Field(None, alias="url_interior")
    property_type: Optional[PropertyType] = Field(None, alias="property_type")

    class Config:
        # Allow using field names instead of aliases when creating instances
        validate_by_name = True
        fields = {
            "Year_Built": {"alias": "Year Built"},
        }
        json_schema_extra = {
            "example": {
                "id": "691b9a39a07ee66c4bdd535e",
                "Name": "433 Monroe St",
                "City": "Brooklyn, NY 11221",
                "Neighbourhood": "Bedford Stuyvesant",
                "Year Built": "Year Built: 1899",
                "Beds": "5 Beds",
                "Baths": "4 Baths",
                "A/C": "Heating & Cooling",
                "area": 2400,
                "actual_price": 2195000,
                "discounted_price": 1995000,
            }
        }
