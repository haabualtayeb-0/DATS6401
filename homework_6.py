"""Homework 6 — Twitter ego-network explorer.

Run with:  streamlit run homework_6.py

Data: SNAP ego-Twitter (McAuley & Leskovec, 2012). A small subset lives in
data/; run download_data.py to rebuild it from the original archive.
"""

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from networkx.algorithms.community import greedy_modularity_communities, modularity
from pyvis.network import Network

HERE = Path(__file__).parent
# Prefer the small bundled subset; fall back to the full SNAP folder if present.
DATA_DIR = HERE / "data" if (HERE / "data").exists() else HERE / "twitter"

# Categorical palette, assigned in fixed order to communities ranked by size.
# Anything past the 8th community (or smaller than MIN_COMMUNITY) is "Other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#9a9993"
MIN_COMMUNITY = 3
OTHER = -1

SIZE_METRICS = {
    "In-Degree (Followers)": "in_degree",
    "Out-Degree (Following)": "out_degree",
    "Betweenness (Bridges)": "betweenness",
    "PageRank": "pagerank",
}
LAYOUTS = ["Spring", "Kamada-Kawai", "Circular by Community"]


# ---------------------------------------------------------------- data ----

@st.cache_data
def list_egos():
    """Ego ids available on disk, with node/edge counts for the selector."""
    egos = {}
    for f in sorted(DATA_DIR.glob("*.edges")):
        edges = pd.read_csv(f, sep=" ", header=None, dtype=str)
        n_nodes = pd.unique(edges.values.ravel()).size
        egos[f.stem] = f"Ego {f.stem} — {n_nodes} Accounts, {len(edges):,} Follows"
    return egos


@st.cache_data
def load_graph(ego):
    """Directed graph: an edge a -> b means account a follows account b."""
    return nx.read_edgelist(DATA_DIR / f"{ego}.edges", create_using=nx.DiGraph)


@st.cache_data
def compute_measures(ego):
    """Centralities and communities on the FULL graph, so filtering never
    changes a node's size or color."""
    G = load_graph(ego)
    U = G.to_undirected()

    comms = sorted(greedy_modularity_communities(U), key=len, reverse=True)
    q = modularity(U, comms)
    community = {}
    for cid, members in enumerate(comms):
        keep = cid < len(PALETTE) and len(members) >= MIN_COMMUNITY
        for n in members:
            community[n] = cid if keep else OTHER

    nodes = pd.DataFrame({
        "in_degree": dict(G.in_degree()),
        "out_degree": dict(G.out_degree()),
        "degree": dict(U.degree()),
        "betweenness": nx.betweenness_centrality(G),
        "pagerank": nx.pagerank(G),
        "community": community,
    })
    nodes.index.name = "account"
    return nodes, q, nx.reciprocity(G)


@st.cache_data
def community_terms(ego, top=3):
    """Hashtags/mentions that are most over-represented in each community.

    Account ids are anonymized, so these are the only clue to what a
    community is about.
    """
    nodes, _, _ = compute_measures(ego)
    names = []
    with open(DATA_DIR / f"{ego}.featnames", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            names.append(line.rstrip("\n").split(" ", 1)[-1])
    feat = pd.read_csv(DATA_DIR / f"{ego}.feat", sep=" ", header=None, dtype={0: str})
    feat = feat.set_index(0)
    feat = feat[feat.index.isin(nodes.index)]
    overall = feat.mean()

    terms = {}
    for cid, members in nodes.groupby("community").groups.items():
        sub = feat[feat.index.isin(members)]
        if cid == OTHER or sub.empty:
            continue
        # share inside the community minus share overall; need >= 2 users
        lift = (sub.mean() - overall)[sub.sum() >= 2].sort_values(ascending=False)
        terms[cid] = [names[i - 1] for i in lift.index[:top] if lift[i] > 0]
    return terms


@st.cache_data
def compute_layout(ego, layout, seed):
    """Positions come from the full graph so nodes stay put while filtering."""
    U = load_graph(ego).to_undirected()
    nodes, _, _ = compute_measures(ego)
    if layout.startswith("Spring"):
        pos = nx.spring_layout(U, seed=seed, k=2.5 / math.sqrt(len(U)), iterations=150)
    elif layout.startswith("Kamada"):
        pos = nx.kamada_kawai_layout(U)
    else:
        # walk the circle one community at a time, biggest accounts first
        order = nodes.assign(c=nodes["community"].replace(OTHER, 99)).sort_values(
            ["c", "degree"], ascending=[True, False]).index
        angles = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
        pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(order, angles)}
    xy = np.array(list(pos.values()))
    scale = 650 / np.abs(xy).max()
    return {n: (float(x) * scale, float(y) * scale) for n, (x, y) in pos.items()}


# ------------------------------------------------------------- drawing ----

def community_name(cid):
    return "Other" if cid == OTHER else f"C{int(cid) + 1}"


def community_color(cid):
    return OTHER_COLOR if cid == OTHER else PALETTE[int(cid)]


def build_network(H, nodes, pos, metric, label_ids):
    net = Network(height="640px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#0b0b0b", cdn_resources="remote")
    vmax = nodes[metric].max() or 1
    for n in H.nodes():
        row = nodes.loc[n]
        # area (not radius) proportional to the metric
        size = 5 + 26 * math.sqrt(row[metric] / vmax)
        net.add_node(
            n,
            label=n if n in label_ids else " ",
            title=(f"Account {n}\n"
                   f"Community: {community_name(row['community'])}\n"
                   f"Followers Here (In-Degree): {int(row['in_degree'])}\n"
                   f"Following Here (Out-Degree): {int(row['out_degree'])}\n"
                   f"Betweenness: {row['betweenness']:.3f}\n"
                   f"PageRank: {row['pagerank']:.4f}"),
            x=pos[n][0], y=pos[n][1], size=size, shape="dot",
            color={"background": community_color(row["community"]), "border": "#ffffff",
                   "highlight": {"background": community_color(row["community"]),
                                 "border": "#0b0b0b"}},
            borderWidth=1.5,
        )
    for a, b in H.edges():
        net.add_edge(a, b)
    net.set_options(json.dumps({
        "physics": {"enabled": False},
        "interaction": {"hover": True, "tooltipDelay": 80, "hideEdgesOnDrag": True},
        "nodes": {"font": {"size": 15, "strokeWidth": 4, "strokeColor": "#ffffff"}},
        "edges": {
            "smooth": False, "width": 0.6, "selectionWidth": 2,
            "color": {"color": "rgba(82,81,78,0.22)", "highlight": "#0b0b0b",
                      "hover": "#52514e"},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.35}},
        },
    }))
    return net


def legend_html(nodes, shown):
    chips = []
    for cid, total in nodes["community"].value_counts().sort_index().items():
        if cid == OTHER:
            continue
        chips.append((cid, total))
    if (nodes["community"] == OTHER).any():
        chips.append((OTHER, int((nodes["community"] == OTHER).sum())))
    parts = []
    for cid, total in chips:
        n_shown = int((shown["community"] == cid).sum())
        fade = "opacity:0.35;" if n_shown == 0 else ""
        parts.append(
            f"<span style='white-space:nowrap;margin-right:16px;{fade}'>"
            f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
            f"background:{community_color(cid)};margin-right:5px'></span>"
            f"{community_name(cid)} <span style='color:#52514e'>({n_shown}/{total})</span></span>")
    return "<div style='line-height:2;font-size:0.9rem'>" + "".join(parts) + "</div>"


# ----------------------------------------------------------------- app ----

st.set_page_config(page_title="Twitter Ego-Network Explorer", layout="wide")
st.title("Twitter Ego-Network Explorer")
st.caption("Who Follows Whom Among the Accounts One Twitter User Follows · SNAP Ego-Twitter")

egos = list_egos()
if not egos:
    st.error(f"No .edges files found in {DATA_DIR}. Run `python download_data.py` first.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    ego = st.selectbox("Ego Network", list(egos), format_func=egos.get)
    nodes, q, reciprocity = compute_measures(ego)
    terms = community_terms(ego)

    size_label = st.selectbox("Node Size Shows", list(SIZE_METRICS))
    metric = SIZE_METRICS[size_label]

    st.subheader("Filters")
    max_deg = int(nodes["degree"].max())
    min_deg = st.slider("Minimum Degree", 0, max_deg, 0)
    cids = [c for c in sorted(nodes["community"].unique()) if c != OTHER]
    if (nodes["community"] == OTHER).any():
        cids.append(OTHER)

    def describe(cid):
        size = int((nodes["community"] == cid).sum())
        return f"{community_name(cid)} ({size})"

    picked = st.multiselect("Communities", cids, default=cids, format_func=describe)
    mutual_only = st.checkbox("Mutual Follows Only")

G = load_graph(ego)
keep = nodes[(nodes["degree"] >= min_deg) & (nodes["community"].isin(picked))].index
H = G.subgraph(keep).copy()
if mutual_only:
    H.remove_edges_from([(a, b) for a, b in list(H.edges()) if not H.has_edge(b, a)])
shown = nodes.loc[list(H.nodes())]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accounts Shown", f"{H.number_of_nodes()} / {G.number_of_nodes()}")
m2.metric("Follows Shown", f"{H.number_of_edges():,} / {G.number_of_edges():,}")
m3.metric("Modularity Q", f"{q:.2f}")
m4.metric("Reciprocity", f"{reciprocity:.0%}")

tab_graph, tab_tables, tab_about = st.tabs(["Network", "Tables", "About & Limitations"])

with tab_graph:
    if H.number_of_nodes() == 0:
        st.warning("No accounts match these filters — lower the minimum degree "
                   "or select more communities.")
    else:
        lay_col, seed_col = st.columns([4, 1])
        layout = lay_col.radio("Layout", LAYOUTS, horizontal=True)
        seed = seed_col.number_input(
            "Spring Seed", 0, 999, 42,
            disabled=not layout.startswith("Spring"))
        st.markdown(legend_html(nodes, shown), unsafe_allow_html=True)
        label_ids = set(shown[metric].nlargest(5).index)
        pos = compute_layout(ego, layout, int(seed))
        net = build_network(H, nodes, pos, metric, label_ids)
        components.html(net.generate_html(), height=670)

with tab_tables:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Accounts Shown")
        table = shown.assign(community=shown["community"].map(community_name))
        table = table.sort_values(metric, ascending=False).rename(columns={
            "in_degree": "In-Degree", "out_degree": "Out-Degree", "degree": "Degree",
            "betweenness": "Betweenness", "pagerank": "PageRank",
            "community": "Community"}).rename_axis("Account")
        st.dataframe(table, width="stretch", height=420, column_config={
            "Betweenness": st.column_config.NumberColumn(format="%.3f"),
            "PageRank": st.column_config.NumberColumn(format="%.4f"),
        })
    with right:
        st.subheader("Communities")
        summary = pd.DataFrame([{
            "Community": community_name(c),
            "Accounts": int((nodes["community"] == c).sum()),
            "Shown": int((shown["community"] == c).sum()),
            "Distinctive Hashtags / Mentions": "  ".join(terms.get(c, [])) or "—",
        } for c in cids])
        st.dataframe(summary, width="stretch", hide_index=True)
        st.caption("Ids are anonymized; these are the tags each community "
                   "uses more than the rest of the network.")

with tab_about:
    n_comm = len([c for c in cids if c != OTHER])
    st.markdown(f"""
**What It Represents:** One Twitter user's follow list, from the SNAP ego-Twitter
dataset (2012). Each node is an account that user follows (ids anonymized); an
arrow a → b means *a follows b*. The user themself is left out, since they follow
everyone shown. This network has {G.number_of_nodes()} accounts and
{G.number_of_edges():,} follows, {reciprocity:.0%} of them returned.

**Why These Encodings:**
- **Color = Detected Community** (greedy modularity): Hue suits a categorical
  variable. It finds {n_comm} groups (Q = {q:.2f}): one person's follow list is
  several separate circles, not one crowd.
- **Size = Centrality:** In-degree shows who everyone in the
  circle follows; betweenness instead enlarges the accounts that bridge two
  communities.

**Limitations:** These are not facts, since changing the spring seed rearranges
the drawing while the data stays the same, so closeness on screen does not mean
two accounts are connected. In the denser networks the center also becomes a
hairball where single edges can't be traced.
""")

