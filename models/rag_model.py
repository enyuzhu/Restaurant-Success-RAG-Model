import os
import json
import re
import joblib
from openai import OpenAI
from langchain.schema import Document
from models.faiss_index import build_faiss_index, search_similar_documents
from utils.data_loader import load_all_restaurants
from models.locations import get_nearby_restaurants, extract_neighborhood_context
from dotenv import load_dotenv
from utils.singapore import population_response, construction_response, get_planning_area
from models.preprocessing import extract_features, extract_swot_features
from pydantic import BaseModel, Field
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

residual_model = joblib.load("/Users/amyyz/Documents/NUS/Official Demo/data/residual_corrector.pkl")

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


load_dotenv()

# Load all restaurants once
all_restaurants_df = load_all_restaurants()

def format_dict_as_string(d):
    return "\n".join(f"{k}: {v}" for k, v in d.items())

def extract_success_score_from_swot_text(text: str):
    try:
        if not text.strip():
            print("Empty GPT output.")
            return None

        # Strip markdown if present
        if text.strip().startswith("```"):
            text = re.sub(r"```(?:json)?", "", text).strip("` \n")

        json_data = json.loads(text)
        parsed = SWOTAnalysis(**json_data)
        return parsed.success_score

    except Exception as e:
        print("Parsing error:", e)
        return None


def coordinates_prompt(structured_input, retrieved_docs):
    context_strings = [doc.page_content for doc in retrieved_docs]
    vector_context = "\n".join(context_strings)
    # print('Most Similar Restaurants:\n',vector_context)

    # Parse the string into name + lat/lon
    lines = vector_context.splitlines()
    parsed = []
    current_name = None

    for line in lines:
        line = line.strip()
        if line.startswith("name: "):
            current_name = line.replace("name:", "").strip()
        elif line.startswith("latitude,longitude:"):
            latlon_str = line.replace("latitude,longitude:", "").strip().strip("()")
            try:
                lat, lon = map(float, latlon_str.split(","))
                if current_name:
                    parsed.append({
                        "name": current_name,
                        "latitude": lat,
                        "longitude": lon
                    })
                    current_name = None  # Reset
            except Exception as e:
                print("Could not parse lat/lon:", latlon_str, e)

    neighborhood_vector_context = {}
    for row in parsed:
        latlon = (row["latitude"], row["longitude"])
        nearby_df = get_nearby_restaurants(latlon, all_restaurants_df)
        neighborhood_vector_context[row["name"]] = extract_neighborhood_context(nearby_df)

    return f"""
You are a geo-aware AI model tasked with predicting ideal locations in Singapore for a restaurant with specified characteristics. Your goal is to match these characteristics with
economic patterns, neighborhood contexts, and success signals (e.g., popularity, review trends, competition density) — and determine whether such restaurants already exist there.


INPUT:
A structured description of a hypothetical restaurant:
{format_dict_as_string(structured_input)}


TASK: IDEAL LOCATION SEARCH & VALIDATION


1. Trait Matching:
- Analyze the input restaurant's features (cuisine, price, crowd type, amenities, etc.).
- Match them to patterns in existing restaurants across Singapore that are the most similar {vector_context} and find patterns in their neighborhood context {neighborhood_vector_context}, 
demographics and population {population_response}, and construction {construction_response}. Avoid coordinates to similar restaurants that have high ratings and reviews but planning 
areas are fine.


2. Ideal Area Suggestion:
- Suggest the top 3 planning areas (e.g., "Bukit Merah", "Downtown Core", "Tampines") where:
   - Restaurants with similar traits are popular or highly rated.
   - There is moderate to low competition.
   - Demographics align well with the input.


3. Best-Fit Coordinates:
- For each suggested area, propose approximate coordinates (up to 6 decimal places) within that planning area where success likelihood
is high — based on high population density and demographics that would like the restarurant's cuisine {population_response} and construction data for growth {construction_response}.


EXPECTED OUTPUT FORMAT:


Suggested Planning Areas for Input Traits:
1. Bugis (Central Area) — High density, tourism traffic, strong match for casual, mid-priced Asian fusion eateries.
2. Tiong Bahru — Residential, strong coffee culture, fits aesthetic/amenity-heavy profiles.
3. Pasir Ris — Underserved area for high-end vegan dining, fits input "family-friendly, nature-focused" theme.


Best-Fit Coordinates:
- (1.301234, 103.854321)
- (1.321111, 103.888888)
- (1.377777, 103.949494)

"""


def format_prompt(structured_input, retrieved_docs):
    context_strings = [doc.page_content for doc in retrieved_docs]
    vector_context = "\n".join(context_strings)
    # print('Most Similar Restaurants: \n',vector_context)

    lat, lon = map(float, structured_input["location"].split(","))

    # get area of lat, long
    area = get_planning_area(lat, lon)
    # print('Area: ', area)

    # get demographics data
    records = population_response['result']['records']
    area_records = [record for record in records if area in record['Number']]
    demographic_population_of_area = []
    for rec in area_records:
        demographic_population_of_area.append(rec)
    # print('Demographics and Population of Area: ', demographic_population_of_area)

    try:
        lat, lon = map(float, structured_input["location"].split(","))
        nearby_df = get_nearby_restaurants((lat, lon), all_restaurants_df)
        neighborhood_context = extract_neighborhood_context(nearby_df)
 
        # print('Neighborhood Context: ', neighborhood_context)

        # search most similar from those close by
        query_str = format_dict_as_string(structured_input)
        nearby_competitors_documents = []
        for _, row in nearby_df.iterrows():
            doc_dict = {
                "name": row.get("name", []),
                "cuisine": row.get("main_category", []),
                "rating": row.get("rating", []),
                "price": row.get("average_price", []),
                "review count": row.get("reviews", []),
                "address": row.get("address", []),
                "open hours": row.get("open_hours", []),
                "categories": row.get("categories", []),
                "latitude,longitude":row.get("latitude, longitude", []),
                "atmosphere rating": row.get("average_atmosphere_score",[]),
                "service rating": row.get("average_service_score",[]),
                "food rating": row.get("average_food_score",[]),
                "service options": row.get("Service options", []),
                "offerings": row.get("Offerings",[]),
                "dining options": row.get("Dining options",[]),
                "crowd": row.get("Crowd",[]),
                "children": row.get("Children",[]),
                "accessibility": row.get("Accessibility",[]),
                "amenities": row.get("Amenities",[]),
                "payments": row.get("Payments",[]),
                "planning": row.get("Planning",[]),
                "pets": row.get("Pets",[])
            }
        nearby_competitors_documents.append(Document(page_content=format_dict_as_string(doc_dict), metadata={"place_id": row.get("place_id", "")}))
        nearby_competitors_db = build_faiss_index(nearby_competitors_documents)
        competitors = search_similar_documents(query_str, nearby_competitors_db)
        # print('Competitors: ', competitors)    

    except:
        neighborhood_context = "This restaurant is in a relatively underserved area."
        competitors = "No competitors found."

    return f"""
You are an AI model tasked with evaluating a restaurant’s potential success in Singapore.


Use the following inputs:


1. Restaurant Attributes:
{format_dict_as_string(structured_input)}


2. Neighborhood Trends (based on geolocation):
{neighborhood_context} and the competitors in the area most similar to the restaurant {competitors}


3. Most similar restaurants based on similarity search for a measurement of how well it would perform in Singapore,
these should NOT be listed as competitors but rather listed as a basis of similar restaurants in Singapore:
{vector_context}


4. Location Feasibility and Growth based on the area: {area} given the population. If there is a low population density and little to no restaurants, give it a low score. 
If no one lives in the area {demographic_population_of_area} and there are no neighboring restaurants due to its isolated location {neighborhood_context}, give the final score a 0 by default.


Your task is to perform a SWOT analysis:
- Strengths: 1) Unique offerings and atmosphere compared to competitors {competitors}, 2) accessibility and amenities, 3) dining options and service flexibility.
If a sub-factor has no meaningful weaknesses or improvements needed, assign it a score of 10. Treat this as a perfect score. 
Avoid unnecessarily lowering strengths that clearly meet all expectations.
Give a score out of 10 to each category and equal weight to the 3 categories.
- Weaknesses: 1) Price points, 2) operational hours (include neighborhood average and different operational hours mean more customers, which is a HIGHER score), 3) restaurant density with a higher score for more restaurants in the area 
due to clustering {neighborhood_context}. Don't include competitors.
If a sub-factor has no meaningful weaknesses or improvements needed, assign it a score of 10. Treat this as a perfect score. 
Avoid unnecessarily lowering weaknesses that clearly meet all expectations.
Give a score out of 10 to each category and equal weight to the 3 categories. 
- Opportunities: 1) Growing population density {population_response}, 2) growing construction {construction_response}, 3) underserved cuisine 
relative to the area meaning no other restaurants of the same cuisine in area {neighborhood_context}. 
If a sub-factor has no meaningful weaknesses or improvements needed, assign it a score of 10. Treat this as a perfect score. 
Avoid unnecessarily lowering opportunities that clearly meet all expectations.
Give a score out of 10 to each category and equal weight to the 3 categories. 
- Threats: 1) High local competition and include only competitors {competitors} in the area {area} with their details (list competitors and their exact details). If there are no competitors,
include details of restaurants in the area and their details {neighborhood_context}. 2) Strong saturation of this restauran't cuisine in the area (highest score if cuisine doesn't exist in neighborhood) {neighborhood_context}. 
If a sub-factor has no meaningful weaknesses or improvements needed, assign it a score of 10. Treat this as a perfect score. 
Avoid unnecessarily lowering threats that clearly meet all expectations.
Give a score out of 10 to each category and equal weight to the 2 categories.

Finally, assign a success score from 0 to 100 with each category in SWOT. Make sure this is in the format "Success Score: " with it rounded to 3 decimals.

Return JSON in the structure matching SWOTSubFactor and SWOTCategory schema. Include overall "Success Score" out of 100 with key name exactly "Success Score".
Return JSON in the structure matching this format:

{{
  "strengths": {{
    "category": "Strengths",
    "explanation": "Overall summary of the strengths...",
    "sub_factors": [
      {{
        "name": "Unique offerings and atmosphere",
        "explanation": "Explain why this is a strength",
        "score": 7.5
      }},
      {{
        "name": "Accessibility and amenities",
        "explanation": "Accessibility explanation",
        "score": 6.5
      }}
    ],
    "total_score": 7.0
  }},
  "weaknesses": {{ ... }},
  "opportunities": {{ ... }},
  "threats": {{ ... }},
  "Success Score": 72.75
}}

Do not return anything outside this JSON structure. No markdown, headers, or extra text.

"""
client = OpenAI() 

def call_gpt4o(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


def coordinates_pipeline(structured_input):
    query_str = format_dict_as_string(structured_input)

    documents = []
    for _, row in all_restaurants_df.iterrows():
        doc_dict = {
            "name": row.get("name", []),
            "cuisine": row.get("main_category", []),
            "rating": row.get("rating", []),
            "price": row.get("Price per person", []),
            "review count": row.get("reviews", []),
            "address": row.get("address", []),
            "open hours": row.get("open_hours", []),
            "categories": row.get("categories", []),
            "latitude,longitude":row.get("latitude, longitude", []),
            "atmosphere rating": row.get("average_atmosphere_score",[]),
            "service rating": row.get("average_service_score",[]),
            "food rating": row.get("average_food_score",[]),
            "service options": row.get("Service options", []),
            "offerings": row.get("Offerings",[]),
            "dining options": row.get("Dining options",[]),
            "crowd": row.get("Crowd",[]),
            "children": row.get("Children",[]),
            "accessibility": row.get("Accessibility",[]),
            "amenities": row.get("Amenities",[]),
            "payments": row.get("Payments",[]),
            "planning": row.get("Planning",[]),
            "pets": row.get("Pets",[])
        }
        documents.append(Document(page_content=format_dict_as_string(doc_dict), metadata={"place_id": row.get("place_id", "")}))

    db = build_faiss_index(documents)
    retrieved_docs = search_similar_documents(query_str, db)

    prompt = coordinates_prompt(structured_input, retrieved_docs)
    return call_gpt4o(prompt)

# === LOAD FAISS INDEX ===
faiss_db_path = "/Users/amyyz/Documents/NUS/Official Demo/data/faiss_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_index = FAISS.load_local(
    faiss_db_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

def run_rag_pipeline(structured_input, db = faiss_index):
    query_str = format_dict_as_string(structured_input)
    retrieved_docs = search_similar_documents(query_str, db)

    prompt = format_prompt(structured_input, retrieved_docs)
    output = call_gpt4o(prompt)

    predicted_score = extract_success_score_from_swot_text(output)

    if predicted_score is None:
        return None 

    # Apply residual correction
    try:
        structured_features = extract_features(structured_input)
        swot_features = extract_swot_features(output)
        combined_features = structured_features + swot_features
        correction = residual_model.predict([combined_features])[0]
        # print("correction is:", correction)        
        adjusted_score = predicted_score + correction
    except Exception as e:
        print("Residual correction failed:", e)
        adjusted_score = predicted_score 

    return f"SWOT Analysis:\n{output}\nResidual-adjusted Success Score: {float(adjusted_score):.3f}"

