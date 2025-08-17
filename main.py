import os
import re
from copy import deepcopy
from models.rag_model import run_rag_pipeline, coordinates_pipeline
from models.preprocessing import parse_inputs, parse_rag

os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid parallelism warning

if __name__ == "__main__":
    print("Please enter the following restaurant attributes:")

    user_input = {
        # "location": input("Location (latitude, longitude): "), # uncomment this line if you want to input your own location
        "cuisine": input("Cuisine (e.g., Korean, Thai): "),
        "price": input("Spending per person (e.g., $10–20): "),
        "payments": input("Accepted payments (comma-separated: cash, credit card, NFC, etc.): "),
        "hours": input("Operating hours (e.g., daily from 11 AM to 10 PM): "),
        "offerings": input("Offerings (comma-separated: alcohol, vegetarian): "),
        "recommended_dishes": input("Recommended dishes (e.g. truffle fries, etc.): "),
        "accessibility": input("Accessibility options (e.g., wheelchair accessible): "),
        "service_options": input("Service options (comma-separated: dine-in, takeaway, delivery): "),
        "highlights": input("Highlights (e.g., rooftop seating, lake view): "),
        "amenities": input("Amenities (comma-separated: Wi-Fi, parking, bar on site, etc.): "),
        "atmosphere": input("Atmosphere (e.g., romantic, casual): "),
        "crowd": input("Crowd type (e.g., family-friendly, groups, tourist-friendly): "),
        "dining_options": input("Dining options (e.g., lunch, dinner, late-night): "),
        "planning": input("Planning features (e.g., accepts reservations, walk-ins): "),
        "children": input("Child-friendliness (e.g., good for kids, high chairs available): "), 
        "pets": input("Pets (dogs allowed inside/outside etc): ")
    }
    structured_input = parse_inputs(user_input)

    # # uncomment this block if you want to input your own location
    # swot_output = run_rag_pipeline(parse_rag(structured_input))
    # print("\n--- SWOT ANALYSIS: ---\n")
    # print(swot_output)

    # comment out the rest of the code below if you want to input your own location
    coordinates_output = coordinates_pipeline(structured_input)
    print("\n--- COORDINATES: ---\n")
    print(coordinates_output)

def extract_best_fit_coords(text):
    section = text
    m = re.search(r"Best-Fit Coordinates:?(.*)", text, flags=re.S|re.I)
    if m:
        section = m.group(1)

    pattern = re.compile(
        r"(?:\*\*(?P<area_bold>[^:*]+)\*\*:|\b(?P<area_plain>[A-Za-z][\w\s&'()-]+):)?"
        r"\s*\(\s*(?P<lat>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<lon>[+-]?\d+(?:\.\d+)?)\s*\)",
        flags=re.I
    )

    results = []
    for g in pattern.finditer(section):
        area = (g.group("area_bold") or g.group("area_plain") or "").strip()
        coord_str = f"{g.group('lat')}, {g.group('lon')}"
        results.append((area, coord_str))

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for area, coord in results:
        if coord not in seen:
            seen.add(coord)
            deduped.append((area, coord))
    return deduped

coords = extract_best_fit_coords(coordinates_output)  

outputs = []
for area, coord in coords:
    this_inputs = deepcopy(structured_input)
    this_inputs['location'] = coord
    parsed = parse_rag(this_inputs)    
    swot_output = run_rag_pipeline(parsed)
    outputs.append((area, coord, swot_output))

for area, coord, swot in outputs:
    print(f"\n=== {area} — {coord} ===")
    print("\n--- SWOT ANALYSIS: ---\n")
    print(swot)

   
