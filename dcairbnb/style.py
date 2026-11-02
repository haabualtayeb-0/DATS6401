"""Palette, Plotly template and page building blocks, shared by every page.

The palette and category orders are the same as in midterm.ipynb (colorblind-validated, fixed order),
so an entity keeps its color across the notebook and the app.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- Palette -----------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # light = low, dark = high
BAND = "rgba(42,120,214,0.18)"   # SERIES[0] at low opacity, for uncertainty ranges
FADED = "#d9d8d1"                # de-emphasized marks next to a highlighted one

ROOM_ORDER = ["Entire home/apt", "Private room", "Hotel room", "Shared room"]
ROOM_COLORS = dict(zip(ROOM_ORDER, SERIES))
PORTFOLIO_BINS = [0, 1, 2, 5, 10, 50, float("inf")]
PORTFOLIO_LABELS = ["1", "2", "3–5", "6–10", "11–50", "51+"]
STAY_ORDER = ["1–2 nights", "3–6 nights", "7–29 nights", "30+ nights"]
TENURE_ORDER = ["0–1 yrs", "2–4 yrs", "5–9 yrs", "10+ yrs"]
ACT_ORDER = ["Reviewed in last 3 months", "Reviewed 3–12 months ago", "Last review over a year ago", "Never reviewed"]
LIC_ORDER = ["Hosted license", "Unhosted license", "Exempt", "Missing", "Other"]
LIC_COLORS = dict(zip(LIC_ORDER, [SERIES[0], SERIES[2], SERIES[3], SERIES[1], MUTED]))
SH_ORDER = ["Superhost", "Not superhost"]
SH_COLORS = {"Superhost": SERIES[0], "Not superhost": SERIES[1]}
DC_CENTER = dict(lat=38.905, lon=-77.025)

FONT = "Inter, Helvetica, Arial, sans-serif"

# Headlines live in the page (st.markdown), so the template has no chart titles and a transparent background
pio.templates["dcairbnb"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_2), automargin=True,
               title=dict(font=dict(color=INK_2, size=12))),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_2), automargin=True,
               title=dict(font=dict(color=INK_2, size=12))),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None, font=dict(color=INK_2, size=12),
                itemsizing="constant"),
    bargap=0.25, margin=dict(t=30, l=10, r=20, b=40),
    hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK, family=FONT)),
    coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=12, tickfont=dict(color=INK_2))),
))
pio.templates.default = "dcairbnb"

# Compact legend overlaid on a map's top-left corner (no "room_type" title, small text and swatches)
MAP_LEGEND = dict(title=None, orientation="v", x=0.01, xanchor="left", y=0.98, yanchor="top",
                  bgcolor="rgba(255,255,255,0.8)", bordercolor=GRID, borderwidth=1,
                  font=dict(size=11, color=INK_2), itemsizing="constant", tracegroupgap=0)

PLOTLY_CONFIG ={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem;}
h1, h2, h3 {letter-spacing: -0.01em;}
.kicker {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #256abf; margin-bottom: 0.2rem;}
.dek {font-size: 1.15rem; line-height: 1.55; color: #52514e; max-width: 46rem; margin-bottom: 0.6rem;}
.chart-head {font-family: "Source Serif 4", Georgia, serif; font-size: 1.32rem; font-weight: 650; line-height: 1.3; margin: 0.4rem 0 0.1rem;}
.chart-sub {color: #52514e; font-size: 0.95rem; margin-bottom: 0.2rem;}
.takeaway {border-left: 4px solid #2a78d6; background: #eef4fc; padding: 0.85rem 1.1rem; border-radius: 0 0.6rem 0.6rem 0;
           margin: 0.6rem 0 1.4rem; line-height: 1.55;}
.takeaway b:first-child {color: #184f95;}
.kicker-plain {text-transform: none; letter-spacing: 0.02em; font-size: 0.9rem;}
.hero-number {font-family: "Source Serif 4", Georgia, serif; font-size: 3.4rem; font-weight: 700; line-height: 1.05; color: #0b0b0b;}
.hero-range {font-size: 1.05rem; color: #52514e;}
.section-rule {border: none; border-top: 1px solid #e1e0d9; margin: 2.2rem 0 1.2rem;}
div[data-testid="stMetricValue"] {font-family: "Source Serif 4", Georgia, serif;}
/* Let long metric labels and notes wrap instead of being cut off with "…" */
div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] div,
div[data-testid="stMetricDelta"] div {white-space: normal !important; overflow: visible !important; text-overflow: clip !important;}
div[data-testid="stCaptionContainer"] p {color: #52514e;}
</style>
"""


def setup_page():
    """Global CSS. Called once per run from app.py."""
    st.html(CSS)


def chapter(kicker, title, dek):
    """Page header: small label, serif headline, one-paragraph summary."""
    st.html(f'<div class="kicker">{kicker}</div>')
    st.title(title, anchor=False)
    st.html(f'<div class="dek">{dek}</div>')


def section(title):
    st.html('<hr class="section-rule">')
    st.header(title_case(title), anchor=False)


def headline(text, sub=None):
    """A chart's headline (its takeaway) and optional subtitle."""
    st.html(f'<div class="chart-head">{text}</div>' + (f'<div class="chart-sub">{sub}</div>' if sub else ""))


def md_safe(text):
    """Escape dollar signs: Streamlit Markdown renders text between two $ as LaTeX math."""
    return text.replace("$", r"\$")


def show_chart(fig, alt, data=None, height=None, key=None):
    """Plot with a text description (alt text for screen readers) and an optional data table."""
    # Plotly Express titles legends with the raw column name ("room_type"); the entries explain themselves
    fig.update_layout(legend_title_text="")
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)
    st.caption(f"**Chart description:** {md_safe(alt)}")
    if data is not None:
        with st.expander("View the Data"):
            st.dataframe(data, hide_index=True, width="stretch")


def takeaway(text):
    st.html(f'<div class="takeaway"><b>Takeaway.</b> {text}</div>')


def next_page(page, label):
    st.html('<hr class="section-rule">')
    st.page_link(page, label=f"Next: {label}", icon=":material/arrow_forward:")


SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs.", "with"}


def title_case(text):
    """Headline-style capitalization that keeps acronyms ('DC') and apostrophes ("Host's") intact."""
    words = text.split()
    return " ".join(w if w.isupper() or (0 < i < len(words) - 1 and w.lower() in SMALL_WORDS) else w[:1].upper() + w[1:]
                    for i, w in enumerate(words))


def join_and(items):
    """'a, b and c' — for listing names inside a sentence."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def short_name(neighbourhood):
    """First name of a neighborhood list ('Capitol Hill, Lincoln Park' -> 'Capitol Hill')."""
    return neighbourhood.split(",")[0]
