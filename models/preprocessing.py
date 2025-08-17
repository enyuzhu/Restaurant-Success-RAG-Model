import re
import pandas as pd
import json
from pydantic import BaseModel, Field
from typing import List

class SWOTSubFactor(BaseModel):
    name: str                     
    explanation: str             
    score: float               

class SWOTCategory(BaseModel):
    category: str                 
    explanation: str              
    sub_factors: List[SWOTSubFactor]  
    total_score: float           

class SWOTAnalysis(BaseModel):
    strengths: SWOTCategory
    weaknesses: SWOTCategory
    opportunities: SWOTCategory
    threats: SWOTCategory
    success_score: float = Field(..., alias="Success Score")

def extract_swot_features(swot_json):
    try:
        if not swot_json:
            raise ValueError("Empty SWOT JSON string")

        # If it's a string, sanitize both leading and trailing garbage
        if isinstance(swot_json, str):
            # Extract the first full JSON block from the string
            match = re.search(r"\{.*\}", swot_json.strip(), re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in string")
            swot_clean = match.group(0)
            swot_model = SWOTAnalysis(**json.loads(swot_clean))

        elif isinstance(swot_json, dict):
            swot_model = SWOTAnalysis(**swot_json)
    
        else:
            raise TypeError("Unsupported type for SWOT input")

        features = []
        expected_subfactors = 3

        for category in [
            swot_model.strengths,
            swot_model.weaknesses,
            swot_model.opportunities,
            swot_model.threats,
        ]:
            features.append(category.total_score)
            sub_scores = [sf.score for sf in category.sub_factors]
            sub_scores += [0.0] * (expected_subfactors - len(sub_scores))
            sub_scores = sub_scores[:expected_subfactors]
            features.extend(sub_scores)

        assert len(features) == 16, f"Expected 16 features, got {len(features)}"
        return features

    except Exception as e:
        print("Failed to extract SWOT features:", e)
        return [0.0] * 16


def parse_inputs(inputs):
    """Clean and structure user-provided fields."""
    return {
        "location": inputs.get("location", "").strip(),
        "cuisine": inputs.get("cuisine", "").strip().title(),
        "price": inputs.get("price", "").strip(),
        "payments": [x.strip().lower() for x in inputs.get("payments", "").split(",") if x.strip()],
        "hours": inputs.get("hours", "").strip(),
        "offerings": [x.strip() for x in inputs.get("offerings", "").split(",") if x.strip()],
        "recommended_dishes": [x.strip() for x in inputs.get("recommended_dishes", "").split(",") if x.strip()],
        "accessibility": inputs.get("accessibility", "").strip(),
        "service_options": [x.strip().lower() for x in inputs.get("service_options", "").split(",") if x.strip()],
        "highlights": inputs.get("highlights", "").strip(),
        "amenities": [x.strip().lower() for x in inputs.get("amenities", "").split(",") if x.strip()],
        "atmosphere": inputs.get("atmosphere", "").strip(),
        "crowd": [x.strip().lower() for x in inputs.get("crowd", "").split(",") if x.strip()],
        "dining_options": inputs.get("dining_options", "").strip(),
        "planning": inputs.get("planning", "").strip(),
        "children": inputs.get("children", "").strip(),
        "pets": inputs.get("pets", "").strip(),
    }

def normalize_price(price_str):
    """Convert '$20-30' or '20-30' into a float (average)."""
    if not price_str:
        return 0.0
    try:
        # remove $ and spaces
        cleaned = price_str.replace("$", "").strip()
        # match numbers
        numbers = re.findall(r"\d+", cleaned)
        numbers = [float(n) for n in numbers]
        if len(numbers) == 1:
            return numbers[0]
        elif len(numbers) >= 2:
            return sum(numbers[:2]) / 2.0  # take average of first two
        else:
            return 0.0
    except Exception:
        return 0.0
    
def parse_rag(inputs):
    def normalize_list(value):
        if isinstance(value, list):
            return [str(x).strip().lower() for x in value if str(x).strip()]
        elif isinstance(value, str):
            return [x.strip().lower() for x in value.split(",") if x.strip()]
        return []

    return {
        "location": str(inputs.get("location", "")).strip(),
        "cuisine": str(inputs.get("cuisine", "")).strip().title(),
        "price": normalize_price(inputs.get("price", "")),   # <-- FIX HERE
        "payments": normalize_list(inputs.get("payments", "")),
        "hours": str(inputs.get("hours", "")).strip(),
        "offerings": normalize_list(inputs.get("offerings", "")),
        "recommended_dishes": normalize_list(inputs.get("recommended_dishes", "")),
        "accessibility": str(inputs.get("accessibility", "")).strip(),
        "service_options": normalize_list(inputs.get("service_options", "")),
        "highlights": str(inputs.get("highlights", "")).strip(),
        "amenities": normalize_list(inputs.get("amenities", "")),
        "atmosphere": str(inputs.get("atmosphere", "")).strip(),
        "crowd": normalize_list(inputs.get("crowd", "")),
        "dining_options": str(inputs.get("dining_options", "")).strip(),
        "planning": str(inputs.get("planning", "")).strip(),
        "children": str(inputs.get("children", "")).strip(),
        "pets": str(inputs.get("pets", "")).strip(),
    }

def parse_row_to_input(row):
    def safe_split(val):
        return [x.strip() for x in str(val).split(",") if x.strip()]

    def parse_location(val):
        try:
            lat, lon = val.strip("()").split(",")
            return f"{float(lat)}, {float(lon)}"
        except(ValueError, AttributeError):
            return ""

    return {
        "location": parse_location(str(row.get("latitude, longitude", ""))),
        "cuisine": str(row.get("main_category", "")).strip().title(),
        "price": str(row.get("average_price", "")).strip(),
        "payments": [x.lower() for x in safe_split(row.get("Payments", ""))],
        "hours": str(row.get("open_hours", "")).strip(),
        "offerings": safe_split(row.get("Offerings", "")),
        "recommended_dishes": safe_split(row.get("Recommended dishes", "")),
        "accessibility": str(row.get("Accessibility", "")).strip(),
        "service_options": [x.lower() for x in safe_split(row.get("Service options", ""))],
        "highlights": str(row.get("Highlights", "")).strip(),
        "amenities": [x.lower() for x in safe_split(row.get("Amenities", ""))],
        "atmosphere": str(row.get("Atmosphere", "")).strip(),
        "crowd": [x.lower() for x in safe_split(row.get("Crowd", ""))],
        "dining_options": str(row.get("Dining options", "")).strip(),
        "planning": str(row.get("Planning", "")).strip(),
        "children": str(row.get("Children", "")).strip(),
        "pets": str(row.get("Pets", "")).strip(),
    }

def extract_features(structured_input):
    # Location
    lat, lon = map(float, structured_input.get("location", "0,0").split(","))

    # Cuisine one-hot example
    known_cuisines = ["japanese", "chinese", "indian", "italian", "thai"]
    cuisine = structured_input.get("cuisine", "").lower()
    cuisine_vec = [1 if cuisine == c else 0 for c in known_cuisines]

    # Price
    price = float(structured_input.get("price", 0))

    # Payments count
    payments = structured_input.get("payments", "")
    payment_count = len(str(payments).split(","))

    # Hours: average daily hours
    hours_str = structured_input.get("hours", "")
    total_hours = 0
    matches = re.findall(r"\[(\d+), (\d+), (\d+), (\d+)\]", str(hours_str))
    for m in matches:
        start = int(m[0]) * 60 + int(m[1])
        end = int(m[2]) * 60 + int(m[3])
        total_hours += (end - start) / 60
    avg_daily_hours = total_hours / 7 if total_hours > 0 else 0

    # Other categorical fields → counts
    def count_tokens(field):
        val = structured_input.get(field, "")
        return len(re.findall(r"\w+", str(val)))

    feature_vector = [
        lat,
        lon,
        price,
        payment_count,
        avg_daily_hours,
        count_tokens("offerings"),
        count_tokens("recommended_dishes"),
        count_tokens("accessibility"),
        count_tokens("service_options"),
        count_tokens("highlights"),
        count_tokens("amenities"),
        count_tokens("atmosphere"),
        count_tokens("crowd"),
        count_tokens("dining_options"),
        count_tokens("planning"),
        count_tokens("children"),
        count_tokens("pets"),
        *cuisine_vec  # adds one-hot vector at the end
    ]

    return feature_vector

def extract_features_from_df(df):
    return pd.DataFrame([
        extract_features(parse_row_to_input(row))
        for _, row in df.iterrows()
    ])
