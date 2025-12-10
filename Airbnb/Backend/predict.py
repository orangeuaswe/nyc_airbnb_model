import pandas as pd
import numpy as np

def prepare_features(data: dict, km):
    """
    Convert incoming request → full feature row matching training schema.
    """

    lat = data["latitude"]
    lon = data["longitude"]

    df = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "minimum_nights": data["minimum_nights"],
        "number_of_reviews": 0,   
        "availability_365": data["availability_365"],
        "calculated_host_listings_count": 1,
        "sentiment_compound": 0.0,  
        "room_type": data["room_type"],
        "neighbourhood_group": infer_borough(lat, lon),
    }])

    df["geo_cluster"] = km.predict(df[["latitude", "longitude"]])

    return df


def infer_borough(lat, lon):
    """
    Approximate borough based on coordinates.
    Super rough but good enough for model consistency.
    """

    # Manhattan
    if 40.69 < lat < 40.88 and -74.03 < lon < -73.91:
        return "Manhattan"

    # Brooklyn
    if 40.56 < lat < 40.73 and -74.05 < lon < -73.85:
        return "Brooklyn"

    # Queens
    if 40.54 < lat < 40.81 and -73.96 < lon < -73.70:
        return "Queens"

    # Bronx
    if 40.79 < lat < 40.92 and -73.93 < lon < -73.76:
        return "Bronx"

    # Staten Island
    if 40.48 < lat < 40.65 and -74.25 < lon < -74.05:
        return "Staten Island"

    return "Unknown"
