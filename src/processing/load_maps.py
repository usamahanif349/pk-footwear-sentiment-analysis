import glob
import os
import pandas as pd

RAW_FILES = glob.glob("data/raw/*.xlsx") + glob.glob("data/raw/*.csv")
OUT_PATH = "data/processed/maps_reviews.csv"


def detect_brand(text):
    t = str(text).lower()
    if "ndure" in t:
        return "Ndure"
    if "borjan" in t or "borgan" in t:
        return "Borjan"
    if "servis" in t or "service shoes" in t:
        return "Servis"
    return None


frames = []
for path in RAW_FILES:
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    if "review_text" not in df.columns:
        continue  # skips the YouTube file
    print(f"{path}: {len(df)} rows")
    frames.append(df)

if not frames:
    raise SystemExit("No Outscraper files found in data/raw")

df = pd.concat(frames, ignore_index=True)
df["brand"] = df["name"].apply(detect_brand)
df["brand"] = df["brand"].fillna(df["query"].apply(detect_brand))

out = pd.DataFrame({
    "brand": df["brand"],
    "source": "google_maps",
    "review_id": df["review_id"],
    "text": df["review_text"],
    "rating": df["review_rating"],
    "date": df["review_datetime_utc"],
    "product_or_outlet": df["name"],
    "video_id": None,
    "likes": df["review_likes"],
})

total = len(out)
out = out.dropna(subset=["brand"]).drop_duplicates(subset=["brand", "review_id"])
with_text = out["text"].notna().sum()

os.makedirs("data/processed", exist_ok=True)
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\nRows in: {total} | kept: {len(out)} | with written text: {with_text}")
print("\nReviews per brand:")
print(out["brand"].value_counts())
print("\nStar rating counts:")
print(out["rating"].value_counts().sort_index())