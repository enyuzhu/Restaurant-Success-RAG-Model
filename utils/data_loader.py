import pandas as pd
import re
import json
import os
import numpy as np
from utils.singapore import get_planning_area
from datetime import datetime

def extract_lat_lng(link):
    # Match pattern like @lat,lng,
    match = re.search(r'!3d([-.\d]+)!4d([-.\d]+)', link)
    if match:
        lat, lng = match.groups()
        return float(lat), float(lng)
    else:
        return None, None


def load_all_restaurants(places_path='/Users/amyyz/Documents/NUS/Official Demo/data/places.csv', reviews_path='/Users/amyyz/Documents/NUS/Official Demo/data/all_reviews.csv', about_path='/Users/amyyz/Documents/NUS/Official Demo/data/About'):
    # Load places.csv
    places_df = pd.read_csv(places_path)
    excluded_categories = [
    "Shopping mall", "Hairdresser", "Pet Shop", "Art cafe", "Community center", "Wine store",
    "Observation deck", "Outlet mall", "Tea store", "Event venue", "Caterer", "Club",
    "Sportswear store", "Condominium complex", "Educational institution", "Shop", "Toy store",
    "Travel lounge", "Media company", "Barbecue area", "Recreation center",
    "Security system supplier", "Heritage building", "Real estate developer",
    "Food manufacturing supply", "Food processing company", "Fruit and vegetable store",
    "Delivery service", "Shared-use commercial kitchen", "Food products supplier",
    "Food and beverage exporter", "Vegetable wholesale market", "Restaurant supply store",
    "Bicycle repair shop", "Department store", "Holding company", "Importer and Exporter",
    "Industrial equipment supplier", "Marketing agency", "Hotel", "Duty free store",
    "Plus size clothing store", "Wedding venue", "Art studio", "Education center", "Sailing club",
    "Beauty salon", "General store", "Supermarket", "Optician", "Optometrist",
    "Children's clothing store"
]
    # Apply the filter
    places_df = places_df[~places_df['main_category'].isin(excluded_categories)]

    # Get lat, lon
    places_df["latitude, longitude"] = places_df["link"].apply(extract_lat_lng)

    # get neighborhood
    places_df["neighborhood"] = places_df["latitude, longitude"].apply(
    lambda coord: get_planning_area(*coord))

    def parse_price_range(price_str): # parses price per person like $10-20
        if not isinstance(price_str, str):
            return None
        try:
            # Match numbers like $10–20 or $1–10
            matches = re.findall(r"\d+", price_str)
            if len(matches) == 2:
                min_price = int(matches[0])
                max_price = int(matches[1])
                avg_price = (min_price + max_price) / 2
                return avg_price
            elif len(matches) == 1:
                val = int(matches[0])
                return val
        except:
            pass
        return None
    
    try:
        reviews_df = pd.read_csv(reviews_path, low_memory=False)

        numeric_cols = ['Atmosphere', 'Food', 'Service',]
        reviews_df[numeric_cols] = reviews_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        reviews_df["Price per person"] = reviews_df["Price per person"].apply(lambda x: pd.Series(parse_price_range(x)))

        rating_summary = reviews_df.groupby('place_id').agg(
            average_atmosphere_score=('Atmosphere', 'mean'),
            average_food_score=('Food', 'mean'),
            average_service_score=('Service', 'mean'),
            average_price=('Price per person', 'mean')
        ).reset_index()

        places_df['place_id'] = places_df['place_id'].astype(str).str.strip()

        # get all recommended dishes
        df_clean = reviews_df[reviews_df['Recommended dishes'].str.strip().astype(bool)].copy()
        recommended = df_clean.groupby('place_id')['Recommended dishes'] \
            .apply(lambda dishes: ', '.join(str(dish) for dish in dishes.dropna())).reset_index()

        places_df = places_df.merge(rating_summary, on="place_id", how="left")
        places_df = places_df.merge(recommended, on="place_id", how="left")

        # Make sure 'date' is datetime before grouping
        reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")

        # Remove rows where 'date' couldn't be parsed
        reviews_df = reviews_df.dropna(subset=["date"])

        # Compute average (mean) review date per place_id
        average_review_dates = (
            reviews_df.groupby("place_id")["date"]
            .mean()
            .reset_index()
            .rename(columns={"date": "average_review_date"})
        )

        # Calculate days since the average review date
        average_review_dates["average_review_date"] = pd.to_datetime(average_review_dates["average_review_date"], errors="coerce")
        today = pd.to_datetime(datetime.today().date())
        average_review_dates["days_since_average_review_date"] = (
            today - average_review_dates["average_review_date"]
        ).dt.days


    except Exception as e:
        print("Warning: could not merge review data:", e)

    about_data = []
    for filename in os.listdir(about_path):
        if filename.endswith(".json"):
            path = os.path.join(about_path, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    place_id = data.get("place_id")
                    about = data.get("About", {})

                    row = {"place_id": place_id} # new dictionary
                    for key, value in about.items():
                        row[key] = ", ".join(value)  # Convert list to string
                    about_data.append(row)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    about_df = pd.DataFrame(about_data)
    places_df = places_df.merge(about_df, on="place_id", how="left")
    # Merge into places_df
    places_df = places_df.merge(average_review_dates, on="place_id", how="left")
    places_df = places_df.dropna(subset=["average_review_date"])

    # get normalized success score
    max_reviews = places_df["reviews"].max()
    max_days = places_df["days_since_average_review_date"].max()

    def compute_success_score(row):
        rating_score = (row["rating"] / 5) * 100
        review_score = np.log1p(row["reviews"]) / np.log1p(max_reviews) * 100
        recency_score = (1 - row["days_since_average_review_date"] / max_days) * 100

        score = (
            0.5 * rating_score +
            0.3 * review_score +
            0.2 * recency_score
        )

        return round(min(100, max(50, score)), 2)

    places_df["success_score"] = places_df.apply(compute_success_score, axis=1)

    return places_df

load_all_restaurants().to_csv("/Users/amyyz/Documents/NUS/Official Demo/data/valid_training_data.csv", index=False)
