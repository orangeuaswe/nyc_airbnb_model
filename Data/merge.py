import pandas as pd
import os

REQUIRED_COLUMNS = {"price", "latitude", "longitude", "room_type"}


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
    # Strip the extension first: "AB_NYC_2019.csv".split("_") ends in
    # "2019.csv", which is not all digits, so the year was never found.
    stem = os.path.splitext(os.path.basename(name))[0]
    for token in stem.split("_"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def merge_airbnb(files, output="merged_airbnb.csv"):
    frames = []
    for file in files:
        df = normalize_cols(load_file(file))
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"Skipping {file}: missing required columns {sorted(missing)}")
            continue
        df["source_file"] = os.path.basename(file)
        df["year"] = extract_year_from_filename(file)
        print(f"Using {file}: {len(df)} rows")
        frames.append(df)

    if not frames:
        raise ValueError("No usable input files.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates()
    merged.to_csv(output, index=False)
    print(f"Wrote {output}: {len(merged)} rows")
    return merged


if __name__ == "__main__":
    merge_airbnb([
        "AB_NYC_2019.csv",
        "new_york_listings_2024.csv",
        "AirBnbNYC_Data.xlsx",
    ])

