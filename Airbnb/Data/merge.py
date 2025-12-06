import pandas as pd

def load_file(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file: {path}")
    return df

def normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def extract_year_from_filename(name):
    for token in name.split("_"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None

def merge_airbnb(files, output="merged_airbnb.csv"):
    frames = []
    for file in files:
        df = load_file(file)
        df = normalize_cols(df)
        year = extract_year_from_filename(file)
        df["source_file"] = file
        df["year"] = year
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates()
    merged.to_csv(output, index=False)
    return merged

if __name__ == "__main__":
    merge_airbnb([
        "AB_NYC_2019.csv",
        "new_york_listings_2024.csv",
        "AirBnbNYC_Data.xlsx"
    ])
