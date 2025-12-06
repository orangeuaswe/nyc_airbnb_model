import pandas as panda
import numpy as num
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import DATA, RANDOM_STATE, N_CLUSTERS

sia = SentimentIntensityAnalyzer()

def loadRaw(path=DATA):
    frame = panda.read_csv(path)
    return frame

def basicClean(frame):
    frame = frame.copy()
    frame = frame.dropna(subset=["price","latitude","longitude","room_type"])
    frame = frame[(frame["price"]>=10)&(frame["price"]<=2000)]
    frame = frame.drop_duplicates()
    return frame

def fixMissingAndDefaults(frame):
    frame = frame.copy()
    numeric_defaults = {
        "minimum_nights": 1,
        "number_of_reviews": 0,
        "availability_365": 0,
        "calculated_host_listings_count": 1
    }
    for c,v in numeric_defaults.items():
        if c not in frame.columns:
            frame[c] = v
        else:
            frame[c] = frame[c].fillna(v)
    cat_defaults = {
        "neighbourhood_group": "Manhattan",
        "room_type": "Entire home/apt"
    }
    for c,v in cat_defaults.items():
        if c not in frame.columns:
            frame[c] = v
        else:
            frame[c] = frame[c].fillna(v)
    if "name" not in frame.columns:
        frame["name"] = ""
    else:
        frame["name"] = frame["name"].fillna("")
    if "description" not in frame.columns:
        frame["description"] = ""
    else:
        frame["description"] = frame["description"].fillna("")
    return frame

def filterGeographicOutliers(frame):
    frame = frame.copy()
    frame = frame[
        (frame["latitude"].between(40.4,40.95)) &
        (frame["longitude"].between(-74.25,-73.7))
    ]
    return frame

def addSentiment(frame):
    frame = frame.copy()
    text = (frame["name"].fillna("") + " " + frame["description"].fillna("")).astype(str)
    frame["sentiment_compound"] = text.apply(lambda t: sia.polarity_scores(t)["compound"])
    return frame

def addGeoClusters(frame):
    frame = frame.copy()
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    frame["geo_cluster"] = km.fit_predict(frame[["latitude","longitude"]])
    return frame, km

def cleanFull(path=DATA):
    df = loadRaw(path)
    df = basicClean(df)
    df = filterGeographicOutliers(df)
    df = fixMissingAndDefaults(df)
    df = addSentiment(df)
    df, km = addGeoClusters(df)
    df = df.reset_index(drop=True)
    return df, km
