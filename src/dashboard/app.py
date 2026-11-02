import html
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS

st.set_page_config(page_title="PK Footwear Sentiment", page_icon="👟", layout="wide")

DATA = "data/processed/results_bert.csv"
BENCH = "reports/benchmark.csv"
ORDER = ["negative", "neutral", "positive"]
NEG, NEU, POS = "#EF4444", "#9CA3AF", "#22C55E"
COLORS = {"negative": NEG, "neutral": NEU, "positive": POS}
BRAND_COLORS = {"Borjan": "#6C5CE7", "Servis": "#00B8D9", "Ndure": "#FF7675"}

STOP = set(STOPWORDS) | {
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "main", "aur", "bhi", "to",
    "ho", "hi", "na", "ye", "yeh", "wo", "woh", "nahi", "kya", "shoes", "shoe",
    "borjan", "servis", "ndure", "service", "br", "just", "like", "one"}

ASPECTS = {
    "price": ["price", "prices", "expensive", "costly", "cheap", "mehnga", "mehngi",
              "mehngay", "sasta", "sasti", "qeemat", "value", "paisa", "paise", "discount", "sale"],
    "quality": ["quality", "durable", "durability", "material", "leather", "sole", "stitching",
                "tut", "toot", "tootgaya", "original", "fake"],
    "comfort": ["comfort", "comfortable", "uncomfortable", "soft", "cushion", "pain",
                "aaram", "aram", "walking"],
    "size": ["size", "sizes", "sizing", "fitting", "tight", "loose", "chota", "chhota",
             "bara", "bada"],
    "design": ["design", "designs", "style", "stylish", "look", "looks", "color", "colour",
               "collection", "variety"],
    "staff": ["staff", "salesman", "salesmen", "rude", "helpful", "polite", "behaviour",
              "behavior", "manager", "counter", "cashier"],
    "delivery": ["delivery", "delivered", "courier", "parcel", "shipping", "order", "online"],
}

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#4f46e5 0%,#7c3aed 50%,#db2777 100%);
       padding: 26px 32px; border-radius: 18px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: 6px 0 0 0; color: #fff; opacity: .92;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e5e7eb; padding: 14px 16px;
       border-radius: 14px; box-shadow: 0 1px 4px rgba(0,0,0,.06);}
div[data-testid="stMetric"] * {color: #111827 !important;}
.card {background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px;
       box-shadow: 0 1px 6px rgba(0,0,0,.06);}
.card, .card * {color: #111827;}
.card h3 {margin: 0 0 4px 0;}
.quote {border-left: 4px solid #7c3aed; background: #f8f7ff; padding: 10px 14px;
        border-radius: 8px; margin-bottom: 10px; font-size: .92rem; color: #111827;}
.quote small {opacity: .65;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    df = pd.read_csv(DATA)
    df["text"] = df["text"].fillna("")
    df["clean_text"] = df["clean_text"].fillna("")
    df["likes"] = df["likes"].fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")
    df["final"] = df["label"].fillna(df["bert_pred"])  # Maps = stars, YouTube = model
    for a, words in ASPECTS.items():
        pat = re.compile(r"\b(?:" + "|".join(words) + r")\b")
        df[f"asp_{a}"] = df["clean_text"].apply(lambda t, p=pat: bool(p.search(t)))
    return df


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      legend_title_text="")
    return fig


def scorecard(b, d):
    n = len(d)
    pos = (d["final"] == "positive").mean() * 100
    neg = (d["final"] == "negative").mean() * 100
    neu = max(0, 100 - pos - neg)
    stars = d["rating"].mean()
    stars_txt = f"{stars:.2f} / 5" if pd.notna(stars) else "no star data"
    srcs = ", ".join(sorted(d["source"].unique()))
    color = BRAND_COLORS.get(b, "#6C5CE7")
    return f"""<div class="card" style="border-top:6px solid {color}">
    <h3>{b}</h3><div style="opacity:.65">{n:,} reviews and comments</div>
    <div style="display:flex;height:10px;border-radius:6px;overflow:hidden;margin:12px 0">
      <div style="width:{pos}%;background:{POS}"></div>
      <div style="width:{neu}%;background:{NEU}"></div>
      <div style="width:{neg}%;background:{NEG}"></div></div>
    <b>{pos:.0f}%</b> positive &nbsp;|&nbsp; <b>{neg:.0f}%</b> negative<br>
    Avg Maps rating: <b>{stars_txt}</b><br>
    <small style="opacity:.6">Sources: {srcs}</small></div>"""


def quote_html(r):
    return (f'<div class="quote">{html.escape(r.text[:280])}<br>'
            f'<small>{r.brand} | {r.source} | {r.likes:.0f} likes</small></div>')


df = load()

st.markdown("""<div class="hero"><h1>👟 Pakistani Footwear: Brand Sentiment Dashboard</h1>
<p>Borjan vs Servis vs Ndure | YouTube comments + Google Maps reviews | English, Roman Urdu and Urdu</p>
</div>""", unsafe_allow_html=True)

st.sidebar.header("Filters")
all_b, all_s, all_l = (sorted(df[c].unique()) for c in ["brand", "source", "language"])
brands = st.sidebar.multiselect("Brand", all_b, default=all_b)
sources = st.sidebar.multiselect("Source", all_s, default=all_s)
langs = st.sidebar.multiselect("Language", all_l, default=all_l)
st.sidebar.caption("Sentiment: Google Maps = from star ratings (1-2 negative, 3 neutral, 4-5 positive). "
                   "YouTube = predicted by a pretrained multilingual model (XLM-R).")

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
st.write("")

tabs = st.tabs(["Overview", "Aspects", "Outlets & trends", "Words & voices", "Browse", "Models & scale"])

with tabs[0]:
    present = sorted(f["brand"].unique())
    for col, b in zip(st.columns(len(present)), present):
        col.markdown(scorecard(b, f[f["brand"] == b]), unsafe_allow_html=True)
    st.write("")
    g = (f.groupby(["brand", "source"])["final"].value_counts(normalize=True)
          .mul(100).rename("pct").reset_index())
    g["group"] = g["brand"] + " | " + g["source"]
    fig = px.bar(g, x="group", y="pct", color="final", category_orders={"final": ORDER},
                 color_discrete_map=COLORS, labels={"pct": "% of reviews", "group": ""},
                 title="Sentiment by brand and source")
    st.plotly_chart(style(fig), use_container_width=True)
    st.caption("Compare brands within the same source only: Maps reviews are about a store visit, "
               "YouTube comments are about ads and videos.")
    m = f[f["rating"].notna()]
    if not m.empty:
        s = m.groupby(["brand", "rating"]).size().reset_index(name="n")
        fig = px.bar(s, x="rating", y="n", color="brand", barmode="group",
                     color_discrete_map=BRAND_COLORS, title="Google Maps star ratings")
        st.plotly_chart(style(fig, 320), use_container_width=True)

with tabs[1]:
    st.subheader("What are people talking about?")
    rows = []
    for a in ASPECTS:
        m = f[f[f"asp_{a}"]]
        if len(m) == 0:
            continue
        rows.append({"aspect": a.title(), "mentions": len(m),
                     "positive": (m["final"] == "positive").mean() * 100,
                     "negative": (m["final"] == "negative").mean() * 100})
    if rows:
        asp = pd.DataFrame(rows)
        asp["net"] = asp["positive"] - asp["negative"]
        asp["label"] = asp["mentions"].apply(lambda n: f"n={n}")
        fig = px.bar(asp.sort_values("net"), x="net", y="aspect", orientation="h",
                     color="net", text="label", range_color=[-100, 100],
                     color_continuous_scale=[NEG, "#E5E7EB", POS],
                     labels={"net": "Net sentiment (% positive minus % negative)", "aspect": ""},
                     title="Net sentiment by topic")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig), use_container_width=True)
        hm = pd.DataFrame({a.title(): f[f[f"asp_{a}"]].groupby("brand").size() for a in ASPECTS}).fillna(0).T
        if hm.values.sum() > 0:
            fig = px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale="Purples",
                            title="How often each topic is mentioned, by brand")
            st.plotly_chart(style(fig, 340), use_container_width=True)
    else:
        st.info("No topic keywords found in the current selection.")
    st.caption("Topics are found with keyword lists (English and Roman Urdu). One review can mention "
               "several topics, and counts are small, so treat this as a signal, not a verdict.")

with tabs[2]:
    mp = f[f["source"] == "google_maps"]
    if mp.empty:
        st.info("No Google Maps data in the current filters.")
    else:
        o = (mp.groupby(["brand", "product_or_outlet"])
               .agg(reviews=("review_id", "count"), avg_stars=("rating", "mean"))
               .round(2).reset_index().sort_values("avg_stars", ascending=False))
        st.subheader("Outlet leaderboard (Google Maps)")
        st.dataframe(o, hide_index=True, column_config={
            "avg_stars": st.column_config.ProgressColumn("Avg stars", min_value=0, max_value=5, format="%.2f"),
            "product_or_outlet": "Outlet", "brand": "Brand", "reviews": "Reviews"})
    t = f.dropna(subset=["date"]).copy()
    if not t.empty:
        t["month"] = t["date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
        tm = t.groupby(["month", "final"]).size().reset_index(name="n")
        fig = px.line(tm, x="month", y="n", color="final", markers=True,
                      color_discrete_map=COLORS, title="Sentiment over time")
        st.plotly_chart(style(fig), use_container_width=True)

with tabs[3]:
    left, right = st.columns(2)
    with left:
        sel = st.selectbox("Most common words in", ORDER, index=0)
        words = " ".join(f[f["final"] == sel]["clean_text"]).split()
        words = [w for w in words if len(w) > 2 and w not in STOP]
        if words:
            top = pd.Series(words).value_counts().head(15).reset_index()
            top.columns = ["word", "count"]
            fig = px.bar(top.sort_values("count"), x="count", y="word", orientation="h",
                         color_discrete_sequence=[COLORS[sel]])
            st.plotly_chart(style(fig, 420), use_container_width=True)
        else:
            st.info("Not enough text for this selection.")
    with right:
        lc = f["language"].value_counts().reset_index()
        lc.columns = ["language", "count"]
        fig = px.pie(lc, names="language", values="count", hole=0.55, title="Language mix")
        st.plotly_chart(style(fig, 420), use_container_width=True)
        st.caption("Roman Urdu is detected with a simple word list, so some is counted as English.")
    st.subheader("Voice of the customer")
    qa, qb = st.columns(2)
    good = f[(f["final"] == "positive") & f["text"].str.len().between(40, 300)].sort_values("likes", ascending=False).head(3)
    bad = f[(f["final"] == "negative") & f["text"].str.len().between(40, 300)].sort_values("likes", ascending=False).head(3)
    qa.markdown("**Most-liked praise**")
    for r in good.itertuples():
        qa.markdown(quote_html(r), unsafe_allow_html=True)
    qb.markdown("**Most-liked complaints**")
    for r in bad.itertuples():
        qb.markdown(quote_html(r), unsafe_allow_html=True)

with tabs[4]:
    q = st.text_input("Search text")
    sent = st.multiselect("Sentiment", ORDER, default=ORDER)
    v = f[f["final"].isin(sent)]
    if q:
        v = v[v["text"].str.contains(q, case=False, na=False)]
    st.caption(f"{len(v):,} rows")
    st.dataframe(v[["brand", "source", "product_or_outlet", "rating", "final", "text"]], hide_index=True)

with tabs[5]:
    st.subheader("Model comparison")
    st.table(pd.DataFrame({
        "Model": ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)", "XLM-R pretrained"],
        "Accuracy": [0.51, 0.43, 0.55],
        "Macro F1": [0.23, 0.40, 0.49],
        "Evaluated on": ["51 held-out rows", "51 held-out rows", "203 labeled rows"]}))
    try:
        b = pd.read_csv(BENCH)
        fig = px.line(b, x="rows", y=["pandas_sec", "spark_sec"], markers=True, log_x=True,
                      labels={"value": "seconds"}, title="Spark vs pandas (replicated data)")
        st.plotly_chart(style(fig), use_container_width=True)
    except FileNotFoundError:
        st.info("benchmark.csv not found in reports/")

with st.expander("Limitations (read before quoting any number)"):
    st.markdown(
        "- Only 203 labeled rows, so scores are noisy.\n"
        "- Servis has YouTube data only, and those labels are unverified model predictions.\n"
        "- Maps data covers a few outlets per brand, not the whole brand.\n"
        "- Neutral (3-star) is the hardest class for every model.\n"
        "- Topic detection uses keyword lists, so it will miss some mentions.")

st.caption("Big Data course project | Data: YouTube Data API v3, Google Maps (Outscraper export)")