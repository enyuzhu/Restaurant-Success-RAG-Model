import pandas as pd

valid_training_path = '/Users/amyyz/Documents/NUS/Official Demo/data/valid_training_data.csv'
training_csv = pd.read_csv(valid_training_path)

# List of required fields (must be present and non-null)
required_fields = [
    "latitude, longitude",
    "main_category",
    "average_price",
    "Payments",
    "open_hours",
    "Offerings",
    "Recommended dishes",
    "Accessibility",
    "Service options",
    "Highlights",
    "Amenities",
    "Atmosphere",
    "Crowd",
    "Planning",
    "Children",
    "Pets",
]

# Filter out rows where any of the required fields is missing
filtered_df = training_csv.dropna(subset=required_fields)

# Optional: save filtered data
filtered_df.to_csv("/Users/amyyz/Documents/NUS/Official Demo/data/filtered_training_data.csv", index=False)
