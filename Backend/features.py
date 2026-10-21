"""Feature construction shared by training and serving.

Anything that turns raw listing fields into model inputs lives here so the
two code paths cannot drift apart again (the live path used to hardcode
sentiment_compound = 0.0 while training computed real VADER scores).
"""
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import (
    BOROUGH_COORDS,
    FEATURES,
    NEW_LISTING_HOST_LISTINGS_COUNT,
    NEW_LISTING_NUMBER_OF_REVIEWS,
)

_analyzer = None


def _sia():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def combine_text(name, description):
    """The text we score for sentiment: listing name + description."""
    return f"{name or ''} {description or ''}"


def text_sentiment(text):
    """VADER compound score in [-1, 1]. Empty text scores 0.0."""
    return float(_sia().polarity_scores(text or "")["compound"])


def build_live_features(borough, room_type, minimum_nights,
                        availability_365, description, km):
    """One-row feature frame for a single prediction request.

    Columns and order match config.FEATURES, which train.py also uses.
    """
    lat, lon = BOROUGH_COORDS[borough]
    row = {
        "latitude": lat,
        "longitude": lon,
        "minimum_nights": minimum_nights,
        "number_of_reviews": NEW_LISTING_NUMBER_OF_REVIEWS,
        "availability_365": availability_365,
        "calculated_host_listings_count": NEW_LISTING_HOST_LISTINGS_COUNT,
        "sentiment_compound": text_sentiment(combine_text("", description)),
        "geo_cluster": int(km.predict(pd.DataFrame([[lat, lon]], columns=["latitude", "longitude"]))[0]),
        "room_type": room_type,
        "neighbourhood_group": borough,
    }
    return pd.DataFrame([row])[FEATURES]
