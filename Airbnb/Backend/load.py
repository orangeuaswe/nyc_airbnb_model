from joblib import load
import tensorflow as tf
from config import PREPROCESSOR_PATH, RANDFOREST_PATH, LINEARMODEL_PATH, NEURAL_PATH

def load_artifacts():
    pre = load(PREPROCESSOR_PATH)
    rf  = load(RANDFOREST_PATH)
    lr  = load(LINEARMODEL_PATH)
    nn  = tf.keras.models.load_model(NEURAL_PATH)
    return pre, rf, lr, nn
