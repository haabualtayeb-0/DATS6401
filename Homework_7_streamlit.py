import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(page_title="Tweet Text Explorer", page_icon="💬", layout="wide")

DATA = Path(__file__).resolve().parent / "data.csv"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
OTHER, NOISE = "#a8a69f", "#dcdbd5"

pio.templates["hw7"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
              size=13, color=INK_2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BASELINE,
               tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12)),
    yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=INK_2)),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID,
                    font=dict(color=INK, size=12)),
    margin=dict(l=0, r=8, t=8, b=0), bargap=0.3,
))
pio.templates.default = "simple_white+hw7"

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1300px;}
  h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: .8rem;}
  .stTabs [data-baseweb="tab-list"] {gap: .5rem; border-bottom: 1px solid #e1e0d9;}
  .stTabs [data-baseweb="tab"] {padding: .6rem 1rem; font-weight: 600;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  .note-label {font-size: .75rem; font-weight: 700; letter-spacing: .06em;
               text-transform: uppercase; color: #898781; margin-bottom: .2rem;}
  .tweet {background: #f3f2ee; border-radius: 8px; padding: .7rem .9rem;
          font-size: .9rem; color: #0b0b0b; min-height: 5.5rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------- helpers -----
@st.cache_data
def load_data():
    return pd.read_csv(DATA).dropna(subset=["OriginalTweet"])


def load_tweets():
    return load_data()["OriginalTweet"].astype(str)


def clean(text):
    """Strip URLs and @mentions — the preprocessing choice that matters most here."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return text


@st.cache_data
def get_docs(n, strip_urls):
    tweets = load_tweets()
    docs = tweets.sample(n, random_state=0) if n < len(tweets) else tweets
    docs = docs.tolist()
    return tuple(clean(d) for d in docs) if strip_urls else tuple(docs)


@st.cache_data
def term_counts(docs, drop_stopwords):
    cv = CountVectorizer(stop_words="english" if drop_stopwords else None)
    X = cv.fit_transform(docs)
    return pd.Series(X.sum(axis=0).A1, index=cv.get_feature_names_out())


@st.cache_resource
def tfidf(docs, drop_stopwords):
    tv = TfidfVectorizer(stop_words="english" if drop_stopwords else None)
    return tv.fit_transform(docs), tv.get_feature_names_out()


@st.cache_data
def project(docs, drop_stopwords):
    T, _ = tfidf(docs, drop_stopwords)
    return TruncatedSVD(n_components=2, random_state=0).fit_transform(T)


@st.cache_resource(show_spinner="Fitting BERTopic (first run takes a minute)…")
def fit_bertopic(n_docs, min_topic_size):
    import torch
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    docs = load_tweets().sample(n_docs, random_state=42).tolist()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    model = BERTopic(embedding_model=embedder, min_topic_size=min_topic_size)
    topics, _ = model.fit_transform(docs)
    return model, docs, np.array(topics)


def bar(series, xlabel, color=BLUE, height=450):
    s = series.sort_values()
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h", marker_color=color,
        marker_cornerradius=4,
        hovertemplate="<b>%{y}</b><br>" + xlabel + ": %{x:,.3~f}<extra></extra>"))
    fig.update_layout(height=height, xaxis_title=xlabel)
    return fig


def chart(fig):
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displayModeBar": False})


def note(reveals, misleads=None):
    with st.container(border=True):
        cols = st.columns(2 if misleads else 1)
        with cols[0]:
            st.markdown('<div class="note-label">What this view reveals</div>',
                        unsafe_allow_html=True)
            st.markdown(reveals)
        if misleads:
            with cols[1]:
                st.markdown('<div class="note-label">How it could mislead</div>',
                            unsafe_allow_html=True)
                st.markdown(misleads)


def topic_label(model, t, n=3):
    return " · ".join(w for w, _ in model.get_topic(t)[:n])


data = load_data()
dates = pd.to_datetime(data["TweetAt"], format="%d-%m-%Y", errors="coerce")

st.title("Coronavirus Tweets as Text Data")

with st.sidebar:
    st.header("Corpus settings")
    n_docs = st.slider("Tweets to sample", 1000, 20000, 8000, 1000)
    st.markdown("**Preprocessing**")
    strip_urls = st.checkbox("Remove URLs and @mentions", value=True)
    drop_sw = st.checkbox("Remove English stop words", value=True)

docs = get_docs(n_docs, strip_urls)
counts = term_counts(docs, drop_sw)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tweets in corpus", f"{len(data):,}")
m2.metric("Tweets sampled", f"{len(docs):,}")
m3.metric("Vocabulary", f"{len(counts):,} terms")
m4.metric("Date range", f"{dates.min():%b %d} – {dates.max():%b %d, %Y}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Frequency", "Distinctive terms", "Projection & clusters", "Topics (BERTopic)"])

# Tab 1
with tab1:
    n_terms = st.slider("Terms to show", 10, 40, 20, 5)
    T, terms = tfidf(docs, drop_sw)
    keyness = pd.Series(T.sum(axis=0).A1, index=terms)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Raw counts")
        st.caption("How many times each term appears across the sample.")
        chart(bar(counts.nlargest(n_terms), "count", height=520))
    with right:
        st.subheader("Summed TF-IDF (keyness)")
        st.caption("Each term's TF-IDF weight, summed over all tweets.")
        chart(bar(keyness.nlargest(n_terms), "summed TF-IDF weight",
                  color=ORANGE, height=520))

    overlap = set(counts.nlargest(n_terms).index) & set(keyness.nlargest(n_terms).index)
    st.caption(f"{len(overlap)} of the top {n_terms} terms appear in both lists. "
               f"Toggle the sidebar options to see preprocessing change the ranking: "
               f"with URLs kept, **https** and **co** (from t.co links) top the counts; "
               f"with stop words kept, **the**, **to** and **and** do.")
    note("Which words dominate the conversation (coronavirus, covid, grocery, "
         "store, prices, food). Position and length make ranks easy to compare.",
         "The ranking is largely an artifact of preprocessing: without removing "
         "URLs, link fragments look like the most 'important' words. Counts also "
         "reward words repeated by many near-duplicate tweets and hashtags.")

# Tab 2
with tab2:
    st.subheader("The 5 most distinctive terms of a tweet")
    T, terms = tfidf(docs, drop_sw)
    picks = st.multiselect("Tweets (by index in the sample)", range(len(docs)),
                           default=[1, 7], max_selections=4)
    cols = st.columns(max(len(picks), 1), gap="large")
    for col, d in zip(cols, picks):
        with col:
            st.markdown(f"**Tweet {d}**")
            st.markdown(f'<div class="tweet">{docs[d][:280]}</div>',
                        unsafe_allow_html=True)
            row = pd.Series(T[d].toarray()[0], index=terms)
            top = row[row > 0].nlargest(5)
            if top.empty:
                st.info("No terms left after preprocessing.")
            else:
                chart(bar(top, "TF-IDF weight", color=ORANGE, height=240))
    note("What makes one tweet different from the rest: words frequent here "
         "but rare in the corpus.",
         "In short texts, a single typo or rare word gets a huge IDF and ranks "
         "as 'distinctive' even though it says little about the tweet's meaning.")

# Tab 3
with tab3:
    st.subheader("TF-IDF → TruncatedSVD (2-D), clustered three ways")
    Y = project(docs, drop_sw)

    c1, c2, c3 = st.columns(3, gap="large")
    k = c1.slider("KMeans k", 2, 10, 4)
    eps = c2.slider("DBSCAN eps", 0.005, 0.05, 0.02, 0.005, format="%.3f")
    mcs = c3.slider("HDBSCAN min_cluster_size", 5, 100, 15, 5)

    runs = [
        (f"KMeans · k={k}", KMeans(k, n_init=10, random_state=0).fit_predict(Y)),
        (f"DBSCAN · eps={eps}", DBSCAN(eps=eps, min_samples=5).fit_predict(Y)),
        (f"HDBSCAN · min size {mcs}", HDBSCAN(min_cluster_size=mcs).fit_predict(Y)),
    ]
    cols = st.columns(3, gap="large")
    for col, (name, labels) in zip(cols, runs):
        found = len(set(labels) - {-1})
        noise = int((labels == -1).sum())
        # Colour only the three largest clusters; more hues stop being
        # distinguishable on a scatter. The rest fold into "other clusters".
        sizes = pd.Series(labels[labels != -1]).value_counts()
        top3 = sizes.index[:3].tolist()
        group = np.where(labels == -1, "unassigned",
                         np.where(np.isin(labels, top3), "", "other clusters"))
        names = {c: f"cluster {c}" for c in top3}
        group = [names.get(l, g) if g == "" else g for l, g in zip(labels, group)]
        df = pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1], "group": group,
                           "cluster": labels, "tweet": [d[:90] for d in docs]})
        cmap = {names[c]: h for c, h in zip(top3, (BLUE, ORANGE, AQUA))}
        cmap.update({"other clusters": OTHER, "unassigned": NOISE})
        order = ["unassigned", "other clusters"] + [names[c] for c in top3][::-1]
        fig = go.Figure()
        for g in order:          # draw grey first so coloured points sit on top
            sub = df[df["group"] == g]
            if sub.empty:
                continue
            fig.add_scatter(x=sub["x"], y=sub["y"], mode="markers", name=g,
                            marker=dict(size=5, color=cmap[g], opacity=0.75),
                            customdata=sub[["cluster", "tweet"]],
                            hovertemplate="cluster %{customdata[0]}<br>"
                                          "%{customdata[1]}<extra></extra>")
        fig.update_layout(height=380, xaxis_title="SVD 1", yaxis_title="SVD 2",
                          legend=dict(orientation="h", y=-0.2, traceorder="reversed",
                                      font=dict(size=11)),
                          yaxis=dict(showgrid=True, gridcolor=GRID,
                                     tickfont=dict(color=MUTED)))
        with col:
            st.markdown(f"**{name}**")
            st.caption(f"{found} groups · {noise:,} unassigned")
            chart(fig)

    st.caption(f"TruncatedSVD of TF-IDF vectors for {len(docs):,} sampled tweets. "
               f"The three largest clusters in each panel are coloured; smaller "
               f"clusters are grey and unassigned points are light grey. Two "
               f"components keep only a small share of the variance, so most "
               f"tweets pile up near the origin.")
    note("Tweets that share vocabulary land near each other; the arms that stick "
         "out are groups using the same dominant words (e.g. grocery/store, "
         "prices/consumer).",
         "The three algorithms give three different 'group counts' for the same "
         "points. KMeans always returns exactly k, and DBSCAN depends heavily on eps. "
         "Clusters reflect parameters and a lossy 2-D projection, not true themes.")

# Tab 4
with tab4:
    st.subheader("BERTopic on a sample of tweets")
    c1, c2, c3 = st.columns([2, 2, 1], gap="large", vertical_alignment="bottom")
    bt_n = c1.select_slider("Tweets to model", [1000, 2000, 3000, 5000], 2000)
    bt_min = c2.slider("min_topic_size", 5, 50, 5, 5)
    run = c3.toggle("Run BERTopic", value=False,
                    help="Loads a sentence-transformer, so it is off by default.")

    if not run:
        st.info("Switch on **Run BERTopic** to fit the topic model.")
        st.stop()

    model, bt_docs, topics = fit_bertopic(bt_n, bt_min)
    info = model.get_topic_info()
    real = info[info["Topic"] != -1].set_index("Topic")
    n_noise = int((topics == -1).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Topics found", len(real))
    m2.metric("Largest topic", f"{real['Count'].max():,} tweets")
    m3.metric("Unassigned (topic −1)", f"{n_noise / len(bt_docs):.0%}")

    pick = st.selectbox(
        "Topic", real.index.tolist(),
        format_func=lambda t: f"Topic {t}: {topic_label(model, t, 4)} "
                              f"({real.loc[t, 'Count']} tweets)")
    left, right = st.columns([2, 3], gap="large")
    with left:
        words = pd.Series(dict(model.get_topic(pick)))
        chart(bar(words, "c-TF-IDF score", height=360))
    with right:
        st.markdown("**Example tweets**")
        examples = [d for d, t in zip(bt_docs, topics) if t == pick][:8]
        st.dataframe(pd.DataFrame({"tweet": examples}), hide_index=True,
                     width="stretch", height=330)

    view = st.segmented_control("Overview", ["Intertopic distance map",
                                             "Topic hierarchy", "Topic info table"],
                                default="Intertopic distance map")
    try:
        if view == "Intertopic distance map":
            chart(model.visualize_topics())
        elif view == "Topic hierarchy":
            chart(model.visualize_hierarchy())
        elif view == "Topic info table":
            st.dataframe(info, width="stretch")
    except Exception as exc:  
        st.warning(f"Could not draw this view: {exc}")

    biggest = real["Count"].nlargest(3)
    big_txt = "; ".join(f"**Topic {t}** ({topic_label(model, t)}, {c} tweets)"
                        for t, c in biggest.items())
    ids = real.index.to_numpy()
    E = model.topic_embeddings_[ids + model._outliers]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    i, j = np.unravel_index(S.argmax(), S.shape)
    a, b = ids[i], ids[j]
    note(f"Grouping {len(bt_docs):,} tweets by meaning gives **{len(real)} topics**. "
         f"The three largest are {big_txt}. Together they hold "
         f"{biggest.sum() / len(bt_docs):.0%} of the sample, while "
         f"{n_noise:,} tweets ({n_noise / len(bt_docs):.0%}) fit no topic. "
         f"The two most similar topics are **Topic {a}** ({topic_label(model, a)}) "
         f"and **Topic {b}** ({topic_label(model, b)}), cosine similarity "
         f"{S[i, j]:.2f}; the hierarchy view shows where they and others merge "
         f"into broader themes.")
