from app.schemas.models import Estate


def build_estate_description(estate: Estate) -> str:
    """
    Generate a full natural-language description for an estate document.
    Any missing fields are safely replaced with 'unknown'.
    """

    address = estate.ADDRESS or "an unknown address"
    est_type = estate.TYPE or "property"
    beds = estate.BEDS or "an unknown number of bedrooms"
    baths = estate.BATH or "an unknown number of bathrooms"
    sqft = estate.PROPERTYSQFT or "an unknown size"
    discounted = estate.discounted_price or "an unknown price"
    actual = estate.actual_price or "an unknown previous price"

    # Clean state
    raw_state = estate.STATE
    state = clean_state(raw_state) or "an unknown location"

    # Extra description from mapping
    state_extra = pick_state_description(state)
    state_extra_text = f" {state_extra}." if state_extra else ""

    return (
        f"The estate at {address} is a {est_type}. "
        f"It has {beds} beds, {baths} baths, and a total size of {sqft} sqft. "
        f"The price is now {discounted} and was {actual} before the reduction. "
        f"It is located in {state}.{state_extra_text}"
    )


def clean_state(state: str) -> str:
    """
    Keep only the part of the state before any comma.
    Example: 'Middle Village, NY 11379' -> 'Middle Village'
    """
    if not state:
        return None
    return state.split(",")[0].strip()


def pick_state_description(state: str) -> str:
    """
    Pick the first available description for the state.
    You can make this random if you prefer.
    """
    if not state:
        return ""

    descriptions = STATE_DESCRIPTIONS.get(state)
    if not descriptions:
        return ""  # No extra description available

    return descriptions[0]  # or use random.choice(descriptions)


STATE_DESCRIPTIONS = {
    "Brooklyn": [
        "a vibrant mix of creativity and residential calm, ideal for young families seeking a strong community feel, small businesses, and a lively urban atmosphere"
    ],
    "New York": [
        "a central area blending cultural diversity with direct access to major business hubs, perfect for professionals who want to stay close to key economic centers"
    ],
    "Staten Island": [
        "a quiet, suburban environment surrounded by nature, ideal for families looking for more space, peaceful streets, and a slower-paced lifestyle"
    ],
    "Bronx": [
        "a lively borough with a strong cultural identity, suitable for those seeking a more affordable cost of living while remaining close to Manhattan"
    ],
    "Manhattan": [
        "in the heart of New York City, ideal for young couples or professionals who enjoy fast-paced living, premium amenities, and dense cultural opportunities"
    ],
    "Flushing": [
        "a bustling multicultural neighborhood known for its food scene, great for students and young workers who enjoy an energetic and diverse environment"
    ],
    "Forest Hills": [
        "a polished residential area popular with families and couples who want a quiet atmosphere with convenient transportation access"
    ],
    "Queens": [
        "a diverse borough offering a balanced mix of residential comfort and commercial activity, great for those looking for a more affordable alternative to Manhattan"
    ],
    "Jamaica": [
        "a transitioning neighborhood with excellent transit connections, well-suited for young professionals seeking mobility and cultural diversity"
    ],
    "Bayside": [
        "a high-end residential area ideal for families looking for quiet streets, strong schools, and a more suburban lifestyle"
    ],
    "Jackson Heights": [
        "a dense and culturally rich neighborhood, perfect for people who value international diversity, local shops, and a very active community"
    ],
    "Rego Park": [
        "a mix of residential buildings and shopping centers, ideal for families and young adults who want easy access to everyday services"
    ],
    "Elmhurst": [
        "an affordable, multicultural area appreciated by workers seeking lower living costs and a wide variety of restaurants"
    ],
    "Howard Beach": [
        "a peaceful waterside neighborhood, attractive to families who want a quiet environment somewhat removed from busy city centers"
    ],
    "Whitestone": [
        "a residential area with abundant green spaces, ideal for those who want a calm, suburban feel while remaining within New York City"
    ],
    "Woodside": [
        "a friendly neighborhood with strong community spirit, perfect for young workers looking for diversity and convenient transportation"
    ],
    "Astoria": [
        "a creative, cultural neighborhood popular with young professionals, appreciated for its art scene, cafes, and vibrant food culture"
    ],
    "Corona": [
        "a family-oriented and affordable area, well-suited for households looking for easy access to shops, parks, and community amenities"
    ],
    "Floral Park": [
        "a small suburban-style neighborhood at the city’s edge, ideal for families seeking tranquility and spacious housing"
    ],
    "East Elmhurst": [
        "a quiet residential area near major transportation routes, great for workers who travel often and want a peaceful home base"
    ],
}
