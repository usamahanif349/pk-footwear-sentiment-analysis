import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from wordcloud import WordCloud, STOPWORDS

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

maps = pd.read_csv("data/processed/maps_reviews.csv")
yt = pd.read_csv("data/raw/youtube_comments.csv")
df = pd.concat([maps, yt], ignore_index=True)
df["text"] = df["text"].fillna("").astype(str)

ROMAN = {"hai", "hain", "nahi", "nahin", "bohat", "bahut", "acha", "achi", "achay",
         "bekar", "kya", "aur", "ka", "ki", "ke", "mein", "main", "se", "ko", "bhai",
         "yar", "yaar", "kharab", "behtareen", "mehnga", "mehngi", "sasta", "qeemat",
         "jooti", "dukan", "bohot", "zabardast", "kamal", "wala", "wali", "lekin", "magar"}


def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def lang(t):
    if re.search(r"[\u0600-\u06FF]", t):
        return "urdu_script"
    hits = sum(1 for w in t.split() if w in ROMAN)
    return "roman_urdu" if hits >= 2 else "english"


def to_label(r):
    if pd.isna(r):
        return None
    r = int(r)
    return "negative" if r <= 2 else ("neutral" if r == 3 else "positive")


df["clean_text"] = df["text"].apply(clean)
df = df[df["clean_text"].str.len() > 2].copy()
df["language"] = df["clean_text"].apply(lang)
df["label"] = df["rating"].apply(to_label)

print("Language split:\n", df["language"].value_counts(), "\n")
labeled = df[df["label"].notna()]
print("Labeled training rows:", len(labeled))
print(labeled["label"].value_counts(), "\n")

X_tr, X_te, y_tr, y_te = train_test_split(
    labeled["clean_text"], labeled["label"], test_size=0.25,
    random_state=42, stratify=labeled["label"])

# char n-grams cope with Roman Urdu spelling variations (acha/achaa/achha)
def make_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           min_df=2, sublinear_tf=True)

models = {
    "Logistic Regression": Pipeline([("v", make_vec()),
        ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Naive Bayes": Pipeline([("v", make_vec()), ("c", MultinomialNB())]),
}

rows, best = [], None
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    rows.append({"model": name, "accuracy": round(acc, 3), "macro_f1": round(f1, 3)})
    if best is None or f1 > best[1]:
        best = (name, f1, pipe, pred)

res = pd.DataFrame(rows)
print(res.to_string(index=False))
best_name, _, best_pipe, best_pred = best
report = classification_report(y_te, best_pred)
print(f"\nBest model: {best_name}\n{report}")

labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_te, best_pred, labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
plt.title(f"Confusion matrix: {best_name}")
plt.savefig("reports/figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

with open("reports/metrics.txt", "w", encoding="utf-8") as f:
    f.write(res.to_string(index=False) + f"\n\nBest: {best_name}\n{report}")

# refit on all labeled data, predict everything
best_pipe.fit(labeled["clean_text"], labeled["label"])
df["predicted"] = best_pipe.predict(df["clean_text"])
df["sentiment"] = df["label"].fillna(df["predicted"])

pivot = (df.groupby(["brand", "source"])["sentiment"]
           .value_counts(normalize=True).unstack(fill_value=0) * 100)
pivot = pivot.reindex(columns=labels, fill_value=0)
pivot.plot(kind="bar", stacked=True, figsize=(9, 5),
           color=["#d9534f", "#aaaaaa", "#5cb85c"])
plt.ylabel("% of reviews")
plt.title("Sentiment by brand and source")
plt.xticks(rotation=30, ha="right")
plt.savefig("reports/figures/sentiment_by_brand.png", dpi=150, bbox_inches="tight")
plt.close()

STOP = set(STOPWORDS) | {"hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "main",
        "aur", "bhi", "to", "ho", "hi", "na", "ye", "yeh", "wo", "woh", "nahi", "kya",
        "shoes", "shoe", "borjan", "servis", "ndure", "service"}
for s in ["positive", "negative"]:
    txt = " ".join(df[(df["sentiment"] == s) & (df["language"] != "urdu_script")]["clean_text"])
    if txt.strip():
        WordCloud(width=900, height=450, background_color="white",
                  stopwords=STOP).generate(txt).to_file(f"reports/figures/wordcloud_{s}.png")

df.to_csv("data/processed/results.csv", index=False, encoding="utf-8-sig")
print("\nSentiment % by brand and source:\n", pivot.round(1))
print("\nDone. Check reports/figures/")