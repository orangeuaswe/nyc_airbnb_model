import joblib
import tensorflow as tf
from config import (
    PREPROCESSOR_PATH,
    RANDFOREST_PATH,
    LINEARMODEL_PATH,
    NEURAL_PATH,
    KMEANS_PATH
)

def load_artifacts():
    pre = joblib.load(PREPROCESSOR_PATH)
    rf  = joblib.load(RANDFOREST_PATH)
    lr  = joblib.load(LINEARMODEL_PATH)
    km  = joblib.load(KMEANS_PATH)
    nn  = tf.keras.models.load_model(NEURAL_PATH)

    return pre, rf, lr, km, nn
