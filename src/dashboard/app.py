import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS

st.set_page_config(page_title="PK Footwear Sentiment", layout="wide")

DATA = "data/processed/results_bert.csv"
BENCH = "reports/benchmark.csv"
ORDER = ["negative", "neutral", "positive"]
COLORS = {"negative": "#d9534f", "neutral": "#aaaaaa", "positive": "#5cb85c"}
STOP = set(STOPWORDS) | {
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "main", "aur", "bhi", "to",
    "ho", "hi", "na", "ye", "yeh", "wo", "woh", "nahi", "kya", "shoes", "shoe",
    "borjan", "servis", "ndure", "service", "br", "just", "like", "one"}


@st.cache_data
def load():
    df = pd.read_csv(DATA)
    df["text"] = df["text"].fillna("")
    df["clean_text"] = df["clean_text"].fillna("")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")
    # Maps = star-based label, YouTube = model prediction
    df["final"] = df["label"].fillna(df["bert_pred"])
    return df


df = load()

st.title("Pakistani Footwear Brands: Sentiment Dashboard")
st.caption("Borjan vs Servis vs Ndure | YouTube comments + Google Maps reviews | "
           "Maps sentiment comes from star ratings, YouTube sentiment from a "
           "pretrained multilingual model (XLM-R)")

st.sidebar.header("Filters")
all_b, all_s, all_l = (sorted(df[c].unique()) for c in ["brand", "source", "language"])
brands = st.sidebar.multiselect("Brand", all_b, default=all_b)
sources = st.sidebar.multiselect("Source", all_s, default=all_s)
langs = st.sidebar.multiselect("Language", all_l, default=all_l)

f = df[df["brand"].isin(brands) & df["source"].isin(sources) & df["language"].isin(langs)]
if f.empty:
    st.warning("No data for these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews / comments", f"{len(f):,}")
c2.metric("Positive", f"{(f['final'] == 'positive').mean() * 100:.0f}%")
c3.metric("Negative", f"{(f['final'] == 'negative').mean() * 100:.0f}%")
stars = f["rating"].mean()
c4.metric("Avg Maps stars", f"{stars:.2f}" if pd.notna(stars) else "n/a")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Brand comparison", "Outlets & trends", "Words & language", "Browse reviews", "Models & scale"])

with tab1:
    g = (f.groupby(["brand", "source"])["final"].value_counts(normalize=True)
          .mul(100).rename("pct").reset_index())
    g["group"] = g["brand"] + " | " + g["source"]
    fig = px.bar(g, x="group", y="pct", color="final", category_orders={"final": ORDER},
                 color_discrete_map=COLORS, labels={"pct": "% of reviews", "group": ""})
    st.plotly_chart(fig, use_container_width=True)
    n = f.groupby(["brand", "source"]).size().reset_index(name="rows")
    st.dataframe(n, hide_index=True)
    st.caption("Compare brands within the same source only. Maps reviews are about a store visit, "
               "YouTube comments are about ads and videos.")
    m = f[f["rating"].notna()]
    if not m.empty:
        s = m.groupby(["brand", "rating"]).size().reset_index(name="n")
        st.plotly_chart(px.bar(s, x="rating", y="n", color="brand", barmode="group",
                               title="Google Maps star ratings"), use_container_width=True)

with tab2:
    mp = f[f["source"] == "google_maps"]
    if mp.empty:
        st.info("No Google Maps data in the current filters.")
    else:
        o = (mp.groupby(["brand", "product_or_outlet"])
               .agg(reviews=("review_id", "count"), avg_stars=("rating", "mean"))
               .round(2).reset_index().sort_values("reviews", ascending=False))
        st.subheader("Outlet performance (Google Maps)")
        st.dataframe(o, hide_index=True)
    t = f.dropna(subset=["date"]).copy()
    if not t.empty:
        t["month"] = t["date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
        tm = t.groupby(["month", "final"]).size().reset_index(name="n")
        st.subheader("Sentiment over time")
        st.plotly_chart(px.line(tm, x="month", y="n", color="final", markers=True,
                                color_discrete_map=COLORS), use_container_width=True)

with tab3:
    left, right = st.columns(2)
    with left:
        sel = st.selectbox("Most common words in", ORDER, index=0)
        words = " ".join(f[f["final"] == sel]["clean_text"]).split()
        words = [w for w in words if len(w) > 2 and w not in STOP]
        if words:
            top = pd.Series(words).value_counts().head(15).reset_index()
            top.columns = ["word", "count"]
            st.plotly_chart(px.bar(top.sort_values("count"), x="count", y="word",
                                   orientation="h"), use_container_width=True)
        else:
            st.info("Not enough text for this selection.")
    with right:
        lc = f["language"].value_counts().reset_index()
        lc.columns = ["language", "count"]
        st.plotly_chart(px.pie(lc, names="language", values="count", title="Language mix"),
                        use_container_width=True)
        st.caption("Roman Urdu is detected with a simple word list, so some is counted as English.")

with tab4:
    q = st.text_input("Search text")
    sent = st.multiselect("Sentiment", ORDER, default=ORDER)
    v = f[f["final"].isin(sent)]
    if q:
        v = v[v["text"].str.contains(q, case=False, na=False)]
    st.dataframe(v[["brand", "source", "product_or_outlet", "rating", "final", "text"]],
                 hide_index=True)

with tab5:
    st.subheader("Model comparison")
    st.table(pd.DataFrame({
        "Model": ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)", "XLM-R pretrained"],
        "Accuracy": [0.51, 0.43, 0.55],
        "Macro F1": [0.23, 0.40, 0.49],
        "Evaluated on": ["51 held-out rows", "51 held-out rows", "203 labeled rows"]}))
    try:
        b = pd.read_csv(BENCH)
        st.subheader("Spark vs pandas (replicated data)")
        st.plotly_chart(px.line(b, x="rows", y=["pandas_sec", "spark_sec"], markers=True,
                                log_x=True, labels={"value": "seconds"}),
                        use_container_width=True)
    except FileNotFoundError:
        st.info("benchmark.csv not found in reports/")

with st.expander("Limitations (read before quoting any number)"):
    st.markdown(
        "- Only 203 labeled rows, so scores are noisy.\n"
        "- Servis has YouTube data only, and those labels are unverified model predictions.\n"
        "- Maps data covers a few outlets per brand, not the whole brand.\n"
        "- Neutral (3-star) is the hardest class for every model.")