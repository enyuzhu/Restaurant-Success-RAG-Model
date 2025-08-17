import requests
import geopandas as gpd
from shapely.geometry import Point

# population per area with ethnicity breakdown 2008-2023    
population_dataset_id = "d_e7ae90176a68945837ad67892b898466"
population_url = "https://data.gov.sg/api/action/datastore_search?resource_id="  + population_dataset_id
        
population_response = requests.get(population_url).json()

# construction development 2008-2023         
construction_dataset_id = "d_9bbcd0c9b0351c7f41c9bfdcdc746668"
construction_url = "https://data.gov.sg/api/action/datastore_search?resource_id="  + construction_dataset_id
        
construction_response = requests.get(construction_url).json()
# print(construction_response.json())

# GeoJSON Singapore Handling Points
planning_areas = gpd.read_file("/Users/amyyz/Documents/NUS/Official Demo/data/district_and_planning_area.geojson")

# Check if a lat/lon falls within a planning area
def get_planning_area(lat, lon):
    point = Point(lon, lat) 
    for _, row in planning_areas.iterrows():
        if row['geometry'].contains(point):
            return row['planning_area'] 
    return None

