# NYC Airbnb Price Predictor

An end-to-end machine learning application that predicts nightly prices for NYC
Airbnb listings. It covers the full pipeline: merging multiple raw datasets,
cleaning and feature engineering, training and comparing several models, and
serving live predictions through a REST API with a web frontend.

## Overview

Given a listing's borough, room type, availability, and a short description, the
app predicts a nightly price. Two things make the model more than a
standard regression:

- **Text sentiment** — listing names and descriptions are scored with VADER
  sentiment analysis, so the tone of a listing becomes a feature.
- **Geographic clustering** — listing coordinates are grouped into 20 KMeans
  clusters, giving the model a learned sense of "neighborhood" beyond the five
  boroughs.

Three models are trained and compared (linear regression, random forest, and a
Keras neural network); the random forest is used as the primary predictor at
serve time, with the other two returned alongside for comparison.

## Architecture

```
Data (2019 CSV + 2024 CSV + Excel)
        │
        ▼
   merge.py            → normalize columns, merge, dedupe → merged_airbnb.csv
        │
        ▼
   clean.py            → filter price/geo outliers, fill defaults,
                          add VADER sentiment, add KMeans geo clusters
        │
        ▼
   train.py            → preprocess (scale + one-hot), train LR / RF / NN,
                          save artifacts with joblib / keras
        │
        ▼
   server.py (FastAPI) → load artifacts once, expose /predict and /health
        │
        ▼
   Frontend (HTML/JS)  → form input → calls API → shows predicted price
```

## Tech stack

- **ML / data:** scikit-learn (LinearRegression, RandomForest, KMeans,
  ColumnTransformer), TensorFlow/Keras, pandas, NumPy
- **NLP:** NLTK VADER sentiment
- **API:** FastAPI, Uvicorn, Pydantic
- **Visualization:** Matplotlib, Seaborn, GeoPandas, contextily
- **Frontend:** HTML, CSS, vanilla JavaScript

## Project structure

```
Airbnb/
├── Data/
│   ├── merge.py                 merge raw datasets into merged_airbnb.csv
│   ├── AB_NYC_2019.csv
│   ├── new_york_listings_2024.csv
│   └── AirBnbNYC_Data.xlsx
├── Backend/
│   ├── config.py                paths, random seed, cluster count
│   ├── clean.py                 cleaning + feature engineering pipeline
│   ├── train.py                 trains and saves all three models
│   ├── features.py / predict.py build live features for a single request
│   ├── server.py                FastAPI app (/predict, /health)
│   └── visual.py                geospatial density heatmaps
└── Frontend/
    ├── index.html
    ├── app.js                   calls the API, renders the result
    └── style.css
```

## Setup

Install dependencies:

```bash
pip install -r Airbnb/reqs.txt
python -m nltk.downloader vader_lexicon
```

## Usage

**1. Merge the datasets**

```bash
cd Airbnb/Data
python merge.py            # produces merged_airbnb.csv
```

**2. Train the models**

```bash
cd ../Backend
python train.py            # cleans data, trains LR / RF / NN, saves artifacts,
                           # prints final MAE and RMSE
```

**3. Run the API**

```bash
python server.py           # serves on http://127.0.0.1:5000
```

**4. Open the frontend**

Open `Airbnb/Frontend/index.html` in a browser (update `API_BASE_URL` in
`app.js` if your API runs on a different host/port).

**Visualize listing density (optional)**

```bash
python visual.py           # hexbin density heatmap of listings across NYC
```

## API

`POST /predict`

```json
{
  "borough": "Manhattan",
  "room_type": "Entire home/apt",
  "minimum_nights": 2,
  "availability_365": 180,
  "guests": 2,
  "description": "Bright, cozy apartment near the park"
}
```

Response:

```json
{
  "random_forest": 185.0,
  "linear_regression": 172.4,
  "neural_network": 190.1,
  "final_price": 185.0,
  "confidence_low": 166.5,
  "confidence_high": 203.5
}
```

`GET /health` returns `{ "status": "ok" }`.

## Notes and possible next steps

- The API currently returns the random forest prediction as the primary price;
  the linear and neural-network outputs are included for comparison.
- A natural extension is to reconcile the training and live feature paths so live
  requests compute sentiment the same way training does, and to expose model
  choice as a parameter.

## License

MIT
