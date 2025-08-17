import pandas as pd
from geopy.distance import geodesic
import ast

def get_nearby_restaurants(target_location, all_restaurants_df, radius_km=0.5):
    """Return restaurants within a radius (km) of the target_location."""
    lat1, lon1 = target_location
    nearby = []

    for _, row in all_restaurants_df.iterrows():
        distance = geodesic((lat1, lon1), row['latitude, longitude']).km
        if distance <= radius_km:
            nearby.append(row)
    return pd.DataFrame(nearby)

def extract_neighborhood_context(nearby_df):
    """Summarize local cuisine density, average prices, etc."""
    if nearby_df.empty:
        return "This restaurant is in a relatively underserved area."
    
    def compute_start_end_times(open_hours_str):
        try:
            hours_dict = ast.literal_eval(open_hours_str)
            daily_times = {}
            for day, time in hours_dict.items():
                if len(time) == 4:
                    start_hour, start_min, end_hour, end_min = time
                    start_minutes = start_hour * 60 + start_min
                    end_minutes = end_hour * 60 + end_min
                    daily_times[day] = (start_minutes, end_minutes)
            return daily_times
        except:
            return {}

    # Parse daily open/close times for each restaurant
    all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    start_times = {day: [] for day in all_days}
    end_times = {day: [] for day in all_days}

    times_list = nearby_df["open_hours"].apply(compute_start_end_times)

    for times in times_list:
        for day in all_days:
            if day in times:
                start, end = times[day]
                start_times[day].append(start)
                end_times[day].append(end)

    def avg_minutes_to_hhmm(minutes):
        if not minutes:
            return "N/A"
        avg = sum(minutes) / len(minutes)
        h = int(avg) // 60
        m = int(avg) % 60
        return f"{h:02d}:{m:02d}"

    avg_operational_hours = {
        day: {
            "avg_start": avg_minutes_to_hhmm(start_times[day]),
            "avg_end": avg_minutes_to_hhmm(end_times[day])
        }
        for day in all_days
    }

    summary = {
        "Cuisine Diversity": nearby_df['main_category'].value_counts().to_dict(),
        "Review Density": nearby_df['reviews'].mean(),
        "Average Price Level": nearby_df['average_price'].mean(),
        "Average Operational Hours Per Day": avg_operational_hours
    }
    return summary

