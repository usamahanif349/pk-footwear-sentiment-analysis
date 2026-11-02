import os
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise SystemExit("YOUTUBE_API_KEY missing. Check your .env file.")

youtube = build("youtube", "v3", developerKey=API_KEY)

BRANDS = {
    "Borjan": ["Borjan shoes", "Borjan shoes review", "Borjan shoes ad"],
    "Servis": ["Servis shoes", "Servis shoes review", "Servis footwear Pakistan"],
    "Ndure": ["Ndure shoes", "Ndure shoes review", "Ndure by Servis"],
}
VIDEOS_PER_QUERY = 10
MAX_COMMENTS_PER_VIDEO = 500
OUT_PATH = "data/raw/youtube_comments.csv"


def find_videos(brand, query):
    res = youtube.search().list(
        q=query, part="snippet", type="video",
        maxResults=VIDEOS_PER_QUERY, regionCode="PK",
    ).execute()
    videos = []
    for item in res.get("items", []):
        s = item["snippet"]
        text = f"{s['title']} {s['description']} {s['channelTitle']}".lower()
        if brand.lower() in text:
            videos.append((item["id"]["videoId"], s["title"]))
    return videos


def get_comments(brand, video_id, title):
    rows, token = [], None
    while len(rows) < MAX_COMMENTS_PER_VIDEO:
        try:
            res = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=token, textFormat="plainText",
            ).execute()
        except HttpError as e:
            print(f"  skipped {video_id}: {e.resp.status} (comments disabled?)")
            break
        for item in res["items"]:
            c = item["snippet"]["topLevelComment"]
            sn = c["snippet"]
            rows.append({
                "brand": brand,
                "source": "youtube",
                "review_id": c["id"],
                "text": sn["textDisplay"],
                "rating": None,
                "date": sn["publishedAt"],
                "product_or_outlet": title,
                "video_id": video_id,
                "likes": sn["likeCount"],
            })
        token = res.get("nextPageToken")
        if not token:
            break
    return rows


def main():
    all_rows, seen_videos = [], set()
    for brand, queries in BRANDS.items():
        for q in queries:
            for vid, title in find_videos(brand, q):
                if (brand, vid) in seen_videos:
                    continue
                seen_videos.add((brand, vid))
                comments = get_comments(brand, vid, title)
                print(f"{brand} | {title[:50]} | {len(comments)} comments")
                all_rows.extend(comments)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["brand", "review_id"])
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("\nTotal comments per brand:")
    print(df["brand"].value_counts())


if __name__ == "__main__":
    main()