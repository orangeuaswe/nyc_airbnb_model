import pandas as panda
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import load
from config import KMEANS_PATH

sia = SentimentIntensityAnalyzer()
kmeans = load(KMEANS_PATH)

def build_live_features(req):
    sentiment = sia.polarity_scores(req.description or "")["compound"]
    geo = int(kmeans.predict([[req.latitude,req.longitude]])[0])
    df = panda.DataFrame([{
        "latitude": req.latitude,
        "longitude": req.longitude,
        "minimum_nights": req.minimum_nights,
        "number_of_reviews": 0,
        "availability_365": 180,
        "calculated_host_listings_count": 1,
        "sentiment_compound": sentiment,
        "geo_cluster": geo,
        "room_type": req.room_type,
        "neighbourhood_group": req.borough
    }])
    return df
