import os
import pandas as panda
import numpy as num
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers, models
from config import *
from clean import cleanFull

os.makedirs(ARTIFACT_DIR, exist_ok=True)

df, km = cleanFull(DATA)
dump(km, KMEANS_PATH)

numeric = [
    "latitude","longitude","minimum_nights",
    "number_of_reviews","availability_365",
    "calculated_host_listings_count","sentiment_compound",
    "geo_cluster"
]

categorical = ["room_type","neighbourhood_group"]

X = df[numeric + categorical]
y = df["price"]

pre = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

X_train_p = pre.fit_transform(X_train)
X_test_p = pre.transform(X_test)

dump(pre, PREPROCESSOR_PATH)

lr = LinearRegression().fit(X_train_p,y_train)
dump(lr, LINEARMODEL_PATH)

rf = RandomForestRegressor(
    n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1
).fit(X_train_p,y_train)
dump(rf, RANDFOREST_PATH)

dim = X_train_p.shape[1]
nn = models.Sequential([
    layers.Input(shape=(dim,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
nn.compile(optimizer="adam", loss="mse", metrics=["mae"])

dense_train = X_train_p.toarray() if hasattr(X_train_p,"toarray") else X_train_p
nn.fit(dense_train, y_train.values, epochs=20, batch_size=256, verbose=1)

nn.save(NEURAL_PATH)

dense_test = X_test_p.toarray() if hasattr(X_test_p,"toarray") else X_test_p
test_pred = nn.predict(dense_test).ravel()

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, test_pred)
rmse = root_mean_squared_error(y_test, test_pred) ** 0.5

print("NN Final MAE:", mae)
print("NN Final RMSE:", rmse)
