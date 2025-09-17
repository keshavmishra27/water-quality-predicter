from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.neighbors import BallTree
from geopy.geocoders import Nominatim

# Sample dataset structure
# df = pd.DataFrame({
#     "district": ["A", "B", "C", ...],
#     "lat": [18.52, ...],
#     "lon": [73.85, ...],
#     "lead": [...],
#     "mercury": [...],
#     "arsenic": [...]
# })


def get_nearby_district_averages(df, location_str, metal_columns, k=3):
    """
    Finds k nearest districts to a user-entered location and returns 
    the average heavy metal content from those districts.

    Args:
      df: pandas DataFrame with 'district', 'lat', 'lon', and heavy metal columns
      location_str: string name of district/village
      metal_columns: list, e.g. ["lead", "mercury", "arsenic"]
      k: neighbors to find (default 3)
    Returns:
      average_dict: dictionary of metal averages
      selected_districts: list of nearest district names
    """
    # Geocode entered location to lat/lon
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(location_str)
    if location is None:
        raise ValueError("Location not found")
    user_coords = [[location.latitude, location.longitude]]
    
    # KNN search using haversine distance (requires lat/lon in radians)
    coords_radians = np.deg2rad(df[['lat', 'lon']].values)
    user_coords_radians = np.deg2rad(user_coords)
    tree = BallTree(coords_radians, metric='haversine')
    dist, idx = tree.query(user_coords_radians, k=k)
    idx = idx[0]
    selected_districts = df.iloc[idx]['district'].tolist()
    
    # Extract metal values from those districts
    metal_data = df.iloc[idx][metal_columns]
    average_dict = metal_data.mean().to_dict()
    return average_dict, selected_districts

# Example usage:
# metals = ["lead", "mercury", "arsenic"]
# averages, districts = get_nearby_district_averages(df, "Some village name", metals)
# print("Averages:", averages)
# print("Used districts:", districts)
