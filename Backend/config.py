from pathlib import Path

# Resolve everything relative to this file so scripts work from any
# working directory (previously "../Data/..." only worked from Backend/).
BACKEND_DIR = Path(__file__).resolve().parent
DATA = BACKEND_DIR.parent / "Data" / "merged_airbnb.csv"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"
LINEARMODEL_PATH = ARTIFACT_DIR / "lin_reg.joblib"
RANDFOREST_PATH = ARTIFACT_DIR / "rand_forest.joblib"
NEURAL_PATH = ARTIFACT_DIR / "neural_net.keras"
KMEANS_PATH = ARTIFACT_DIR / "kmeans_geo.joblib"
CONFORMAL_PATH = ARTIFACT_DIR / "conformal.json"
RANDOM_STATE = 40
N_CLUSTERS = 20
CONFORMAL_ALPHA = 0.1  # 90% prediction interval

# --- Feature schema shared by training (train.py) and serving (features.py) ---
NUMERIC_FEATURES = [
    "latitude", "longitude", "minimum_nights",
    "number_of_reviews", "availability_365",
    "calculated_host_listings_count", "sentiment_compound",
    "geo_cluster",
]
CATEGORICAL_FEATURES = ["room_type", "neighbourhood_group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The API takes a borough, not coordinates, so each borough is mapped to a
# representative point. geo_cluster at serve time therefore depends only on
# the borough (see README, Limitations).
BOROUGH_COORDS = {
    "Manhattan":     (40.7831, -73.9712),
    "Brooklyn":      (40.6782, -73.9442),
    "Queens":        (40.7282, -73.7949),
    "Bronx":         (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}

# A listing that does not exist yet has no review history and one listing.
NEW_LISTING_NUMBER_OF_REVIEWS = 0
NEW_LISTING_HOST_LISTINGS_COUNT = 1
