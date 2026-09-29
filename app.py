import html
import sys
from collections import Counter
from datetime import date
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import models, prediction, feature_extraction as fe, preprocessing as pp
from utils import data_loader, plots

DATA_PATH = ROOT / "data" / "combined_train.csv"
EXAMPLE = "मुंबईमध्ये आज मुसळधार पावसामुळे अनेक भागांमध्ये वाहतूक विस्कळीत झाली."

st.set_page_config(
    page_title="Marathi News Desk",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------- News-room theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700;800&display=swap');

:root {
    --news-red: #c91f2b;
    --news-dark: #101214;
    --news-muted: #69707a;
    --news-line: #dfe2e5;
    --news-paper: #f7f7f5;
}

html, body, [class*="css"] {
    font-family: "Inter", "Noto Sans Devanagari", sans-serif;
}

.stApp {
    background: var(--news-paper);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* Masthead */
.news-masthead {
    background: #ffffff;
    border-top: 5px solid var(--news-red);
    border-bottom: 1px solid #111;
    padding: 1rem 1.35rem .85rem 1.35rem;
    margin-bottom: 1.15rem;
}

.news-kicker-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: .25rem 1rem;
    margin-bottom: .2rem;
}

.news-kicker {
    color: var(--news-red);
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.news-dateline {
    color: var(--news-muted);
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.news-brand {
    color: var(--news-dark);
    font-family: Georgia, "Noto Sans Devanagari", serif;
    font-size: 2.25rem;
    line-height: 1;
    font-weight: 800;
    margin: 0;
}

.news-subtitle {
    color: var(--news-muted);
    font-size: .84rem;
    margin-top: .45rem;
}

/* Double rule under the masthead: thick line, gap, thin line */
.news-rule {
    height: 5px;
    border-top: 3px solid var(--news-red);
    border-bottom: 1px solid var(--news-red);
    margin: .7rem 0 .2rem 0;
}

/* Section labels */
.section-kicker {
    color: var(--news-red);
    font-size: .73rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    border-left: 4px solid var(--news-red);
    padding-left: .65rem;
    margin: .4rem 0 .8rem 0;
}

/* Thin divider used between the input and the result */
.news-hr {
    border: 0;
    border-top: 1px solid var(--news-line);
    margin: 1.1rem 0 1rem 0;
}

/* News cards */
.news-card {
    background: #ffffff;
    border: 1px solid var(--news-line);
    border-radius: 3px;
    padding: 1.15rem 1.25rem;
    margin: .35rem 0 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,.035);
}

.news-card h3 {
    margin: 0 0 .35rem 0;
    font-family: Georgia, "Noto Sans Devanagari", serif;
    font-size: 1.35rem;
}

.news-label {
    color: var(--news-red);
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
}

/* Marathi text */
textarea, .mr {
    font-family: "Noto Sans Devanagari", "Nirmala UI", "Mangal", "Lohit Devanagari", sans-serif !important;
    font-size: 1.05rem;
}

.mr {
    background: #fff;
    color: #202428;
    padding: .75rem .9rem;
    border: 1px solid var(--news-line);
    border-left: 4px solid var(--news-red);
    margin: .25rem 0 .85rem 0;
}

/* Visible focus on the text input */
.stTextArea [data-baseweb="textarea"]:focus-within,
.stTextArea [data-baseweb="base-input"]:focus-within {
    border-color: var(--news-red) !important;
    box-shadow: 0 0 0 1px var(--news-red) !important;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: var(--news-red);
    border-color: var(--news-red);
    font-weight: 700;
    border-radius: 3px;
}

.stButton > button[kind="primary"]:hover {
    background: #a91823;
    border-color: #a91823;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #111;
}

.stTabs [data-baseweb="tab"] {
    background: #ffffff;
    border-radius: 0;
    padding: .7rem 1rem;
    font-weight: 600;
    color: #4d5359;
    transition: color .15s ease, background-color .15s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--news-red);
    background: #fbf3f3;
}

.stTabs [aria-selected="true"] {
    color: var(--news-red) !important;
    border-bottom: 3px solid var(--news-red) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid var(--news-line);
    border-top: 3px solid var(--news-red);
    padding: .75rem .9rem;
    border-radius: 2px;
}

/* Tables / expanders */
[data-testid="stDataFrame"], .stTable {
    border: 1px solid var(--news-line);
}

.streamlit-expanderHeader {
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111315;
}

section[data-testid="stSidebar"] * {
    color: #f2f2f2;
}

section[data-testid="stSidebar"] .sidebar-title {
    color: #ff6b74;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    border-bottom: 1px solid #2c3034;
    padding-bottom: .45rem;
    margin-bottom: .6rem;
}

section[data-testid="stSidebar"] .sidebar-note {
    color: #b9bdc2;
    font-size: .8rem;
    line-height: 1.6;
}

section[data-testid="stSidebar"] .sidebar-note b {
    color: #f2f2f2;
}

/* Headings use the same serif as the masthead */
h1, h2, h3 {
    font-family: Georgia, "Noto Sans Devanagari", serif !important;
}

h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.1rem !important; }

@media (max-width: 800px) {
    .news-brand { font-size: 1.7rem; }
}

/* ----------------------------------------------------------------
   Background graphics: subtle newsroom / newspaper atmosphere
   ---------------------------------------------------------------- */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .55;
    background-image:
        radial-gradient(circle at 12% 18%, rgba(201,31,43,.10) 0 2px, transparent 3px),
        radial-gradient(circle at 86% 72%, rgba(16,18,20,.08) 0 2px, transparent 3px),
        radial-gradient(circle at 72% 12%, rgba(201,31,43,.07) 0 1.5px, transparent 2.5px),
        linear-gradient(135deg, transparent 0 48%, rgba(16,18,20,.025) 49% 50%, transparent 51% 100%);
    background-size: 46px 46px, 62px 62px, 35px 35px, 180px 180px;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 420px;
    height: 420px;
    right: -150px;
    top: 90px;
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
    border: 1px solid rgba(201,31,43,.10);
    box-shadow:
        0 0 0 35px rgba(201,31,43,.025),
        0 0 0 70px rgba(201,31,43,.018),
        0 0 0 105px rgba(201,31,43,.012);
}

.block-container,
[data-testid="stSidebar"] {
    position: relative;
    z-index: 1;
}

/* Ticker: text enters from the right edge and scrolls left, then repeats */
.news-ticker {
    display: flex;
    align-items: center;
    overflow: hidden;
    height: 34px;
    background: #111315;
    color: white;
    margin: -0.2rem 0 1rem 0;
    border-radius: 2px;
    position: relative;
}

.news-ticker-label {
    flex: 0 0 auto;
    position: relative;
    z-index: 3;
    background: var(--news-red);
    height: 100%;
    display: flex;
    align-items: center;
    padding: 0 13px;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .1em;
    white-space: nowrap;
}

.news-ticker-text {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    font-size: .78rem;
    color: #e8e8e8;
    white-space: nowrap;
}

/* padding-left: 100% pushes the first character just past the right edge */
.news-ticker-track {
    display: inline-block;
    white-space: nowrap;
    padding-left: 100%;
    animation: tickerMove 36s linear infinite;
    will-change: transform;
}

.news-ticker-track span {
    display: inline-block;
    padding-right: 3.5rem;
}

.news-ticker:hover .news-ticker-track {
    animation-play-state: paused;
}

@keyframes tickerMove {
    from { transform: translateX(0); }
    to   { transform: translateX(-100%); }
}

@media (prefers-reduced-motion: reduce) {
    .news-ticker-track {
        animation: none;
        padding-left: 16px;
    }
}

/* Floating category chips */
.category-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: .3rem 0 1.1rem 0;
}

.category-chip {
    border: 1px solid #d9dce0;
    background: rgba(255,255,255,.88);
    padding: 5px 10px;
    border-radius: 999px;
    color: #555b62;
    font-size: .68rem;
    font-weight: 700;
}

.category-chip.hot {
    border-color: rgba(201,31,43,.35);
    color: var(--news-red);
    background: rgba(201,31,43,.055);
}

/* Hero classification card */
.hero-news {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(120deg, rgba(255,255,255,.98), rgba(250,247,244,.96));
    border: 1px solid #d8dadd;
    border-left: 6px solid var(--news-red);
    padding: 1.45rem 1.55rem;
    margin-bottom: 1.15rem;
    box-shadow: 0 8px 28px rgba(0,0,0,.055);
}

.hero-news::after {
    content: "NEWS";
    position: absolute;
    right: -18px;
    bottom: -34px;
    font-family: Georgia, serif;
    font-size: 7rem;
    font-weight: 900;
    color: rgba(201,31,43,.045);
    transform: rotate(-8deg);
}

.hero-news .eyebrow {
    color: var(--news-red);
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .18em;
}

.hero-news .headline {
    position: relative;
    z-index: 1;
    font-family: Georgia, "Noto Sans Devanagari", serif;
    font-size: 2rem;
    line-height: 1.15;
    font-weight: 800;
    margin: .3rem 0 .45rem;
    color: #151719;
}

.hero-news .summary {
    position: relative;
    z-index: 1;
    color: #626870;
    font-size: .9rem;
    max-width: 800px;
}

/* Small newspaper corner graphic */
.mini-press {
    position: absolute;
    right: 28px;
    top: 24px;
    width: 72px;
    height: 88px;
    border: 2px solid rgba(201,31,43,.18);
    transform: rotate(5deg);
    background: rgba(255,255,255,.6);
}

.mini-press::before {
    content: "📰";
    position: absolute;
    font-size: 27px;
    top: 8px;
    left: 20px;
}

.mini-press::after {
    content: "━━━━\\A ━━━\\A ━━━━━\\A ━━";
    white-space: pre;
    position: absolute;
    left: 12px;
    bottom: 8px;
    color: rgba(16,18,20,.25);
    font-size: 9px;
    line-height: 1.2;
}

@media (max-width: 800px) {
    .hero-news .headline { font-size: 1.5rem; }
    .mini-press { display: none; }
}

/* Classification result: "filed under" card and score list */
.result-card {
    background: #ffffff;
    border: 1px solid var(--news-line);
    border-top: 3px solid var(--news-red);
    padding: 1rem 1.15rem 1.05rem 1.15rem;
}

.result-card .news-label {
    display: block;
    margin-bottom: .25rem;
}

.result-card .result-category {
    font-family: Georgia, "Noto Sans Devanagari", serif;
    font-size: 1.9rem;
    line-height: 1.15;
    font-weight: 800;
    color: #151719;
    margin: 0 0 .5rem 0;
}

.result-card .result-meta {
    color: var(--news-muted);
    font-size: .78rem;
    border-top: 1px solid var(--news-line);
    padding-top: .5rem;
}

.score-list {
    background: #ffffff;
    border: 1px solid var(--news-line);
    padding: .35rem .95rem;
}

.score-row {
    display: grid;
    grid-template-columns: minmax(90px, 1.2fr) 3fr 62px;
    align-items: center;
    gap: .7rem;
    padding: .42rem 0;
    border-bottom: 1px dotted var(--news-line);
    font-size: .86rem;
    color: #2a2f34;
}

.score-row:last-child {
    border-bottom: 0;
}

.score-track {
    height: 6px;
    background: #eceeef;
}

.score-fill {
    height: 100%;
    background: #9aa0a6;
}

.score-row.top .score-fill {
    background: var(--news-red);
}

.score-row.top .score-name {
    font-weight: 700;
}

.score-val {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: #4d5359;
}

/* Side-by-side model comparison */
.result-card .result-note {
    font-size: .9rem;
    color: #2a2f34;
    margin-bottom: .5rem;
}

.result-card .result-note.split {
    color: #8a5a00;
}

.model-card {
    background: #ffffff;
    border: 1px solid var(--news-line);
    border-top: 3px solid #9aa0a6;
    padding: .85rem 1rem .6rem 1rem;
    height: 100%;
}

.model-card.best {
    border-top-color: var(--news-red);
}

.model-card .m-name {
    font-weight: 700;
    font-size: .9rem;
    color: #2a2f34;
}

.model-card .m-tag {
    display: inline-block;
    margin-left: .4rem;
    padding: 0 .4rem;
    font-size: .62rem;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--news-red);
    border: 1px solid var(--news-red);
    border-radius: 2px;
    vertical-align: 1px;
}

.model-card .m-pred {
    font-family: Georgia, "Noto Sans Devanagari", serif;
    font-size: 1.5rem;
    line-height: 1.2;
    font-weight: 800;
    color: #151719;
    margin: .3rem 0 .1rem 0;
}

.model-card .m-meta {
    color: var(--news-muted);
    font-size: .74rem;
    margin-bottom: .45rem;
}

.score-row.plain {
    grid-template-columns: 1fr auto;
}

/* Pipeline diagram at the bottom of the preprocessing tab, centred */
.st-key-pipeline_center .pipeline-heading {
    text-align: center;
    font-weight: 700;
    margin: 1.4rem 0 .4rem 0;
}

.st-key-pipeline_center [data-testid="stGraphVizChart"] {
    display: flex;
    justify-content: center;
}

</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return data_loader.load_dataset(DATA_PATH)


@st.cache_resource(show_spinner="Training models (first run only; cached afterwards)...")
def get_bundle(_df, text_col, label_col):
    return models.train_all(_df, text_col, label_col)


def mr(text):
    st.markdown(f'<div class="mr">{text if text else "&nbsp;"}</div>', unsafe_allow_html=True)


def pct(x):
    return f"{x * 100:.2f}%"


def pipeline_graph():
    st.graphviz_chart("""digraph G { rankdir=TB; node [shape=box, fontname="Helvetica", fontsize=11, style=filled, fillcolor="#f3f5f8", color="#1f4e79"];
    a [label="Marathi news text"]; b [label="Text normalization\\n(NFC, zero-width, digits, punctuation, whitespace)"];
    c [label="Tokenization\\n(Devanagari-aware regex)"]; d [label="Stopword removal\\n(data/marathi_stopwords.txt)"];
    e [label="TF-IDF vectorization\\n(word 1-2 grams, fitted on training split only)"];
    f [label="Classifier\\n(Multinomial NB | Logistic Regression | Linear SVM)"]; g [label="Predicted category", fillcolor="#dbe7f3"];
    a->b->c->d->e->f->g; }""")


def score_bars(items):
    """Horizontal bars for class probabilities (values between 0 and 1); the first item is the top class."""
    rows = []
    for i, (name, value) in enumerate(items):
        width = max(0.0, min(1.0, float(value))) * 100
        rows.append(
            f'<div class="score-row{" top" if i == 0 else ""}">'
            f'<span class="score-name">{html.escape(str(name))}</span>'
            f'<span class="score-track"><span class="score-fill" style="display:block;width:{width:.1f}%"></span></span>'
            f'<span class="score-val">{width:.2f}%</span>'
            f'</div>'
        )
    st.markdown('<div class="score-list">' + "".join(rows) + '</div>', unsafe_allow_html=True)


def model_card(name, res, is_best):
    """One model's prediction and scores, for the side-by-side comparison."""
    is_prob = res["score_type"] == "Probability"
    rows = []
    for i, (cat, value) in enumerate(res["top"]):
        cls = "score-row" + ("" if is_prob else " plain") + (" top" if i == 0 else "")
        if is_prob:
            width = max(0.0, min(1.0, float(value))) * 100
            rows.append(
                f'<div class="{cls}"><span class="score-name">{html.escape(str(cat))}</span>'
                f'<span class="score-track"><span class="score-fill" style="display:block;width:{width:.1f}%"></span></span>'
                f'<span class="score-val">{width:.2f}%</span></div>'
            )
        else:
            rows.append(
                f'<div class="{cls}"><span class="score-name">{html.escape(str(cat))}</span>'
                f'<span class="score-val">{float(value):+.3f}</span></div>'
            )
    tag = '<span class="m-tag">Selected model</span>' if is_best else ""
    kind = "Calibrated probability" if is_prob else "Decision score (not a probability)"
    return (
        f'<div class="model-card{" best" if is_best else ""}">'
        f'<div class="m-name">{html.escape(name)}{tag}</div>'
        f'<div class="m-pred">{html.escape(str(res["prediction"]))}</div>'
        f'<div class="m-meta">{kind}</div>'
        f'<div class="score-list" style="border:0;padding:0">{"".join(rows)}</div>'
        f'</div>'
    )


TODAY = date.today().strftime("%A, %d %B %Y")

st.markdown(f"""
<div class="news-masthead">
    <div class="news-kicker-row">
        <span class="news-kicker">AI NEWS DESK · MARATHI EDITION</span>
        <span class="news-dateline">{TODAY}</span>
    </div>
    <div class="news-brand">मराठी News Intelligence</div>
    <div class="news-subtitle">
        Automated Marathi news classification powered by NLP, TF-IDF and supervised machine learning.
    </div>
    <div class="news-rule"></div>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="news-ticker">
    <div class="news-ticker-label">BREAKING</div>
    <div class="news-ticker-text">
        <div class="news-ticker-track">
            <span>मराठी News Intelligence &nbsp; • &nbsp; NLP &nbsp; • &nbsp; TF-IDF &nbsp; • &nbsp; Machine Learning &nbsp; • &nbsp; Automated News Classification</span>
            <span>मराठी News Intelligence &nbsp; • &nbsp; NLP &nbsp; • &nbsp; TF-IDF &nbsp; • &nbsp; Machine Learning &nbsp; • &nbsp; Automated News Classification</span>
        </div>
    </div>
</div>
<div class="category-strip">
    <span class="category-chip hot">● LIVE ANALYSIS</span>
    <span class="category-chip">MARATHI</span>
    <span class="category-chip">NLP</span>
    <span class="category-chip">TF-IDF</span>
    <span class="category-chip">CLASSIFICATION</span>
    <span class="category-chip">AI NEWS DESK</span>
</div>
""", unsafe_allow_html=True)

try:
    data = get_data()
except data_loader.DatasetError as e:
    st.error(str(e)); st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading the dataset: {e}"); st.stop()

df, TXT, LBL, S = data["df"], data["text_col"], data["label_col"], data["stats"]
try:
    B = get_bundle(df, TXT, LBL)
except Exception as e:
    st.error(f"Model training failed: {e}"); st.stop()

R, best, classes = B["results"], B["best"], B["classes"]

with st.sidebar:
    st.markdown('<div class="sidebar-title">Desk status</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-note">'
        f'Selected model: <b>{html.escape(best)}</b><br>'
        f'Macro F1: <b>{pct(R[best]["f1"])}</b><br>'
        f'Categories: <b>{len(classes)}</b><br>'
        f'Training articles: <b>{B["n_train"]:,}</b><br>'
        f'Test articles: <b>{B["n_test"]:,}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

t_cls, t_pre, t_eval, t_info, t_data = st.tabs(["News Desk", "NLP Preprocessing", "Model Evaluation", "Project Details", "Dataset Overview"])

# ---------------------------------------------------------------- Dataset
with t_data:
    st.markdown('<div class="section-kicker">NEWSROOM · DATASET ARCHIVE</div>', unsafe_allow_html=True)
    st.markdown("""<div class="news-card"><div class="news-label">DATA DESK</div><h3>Dataset Overview</h3><div>Dataset statistics, category distribution, quality checks and sample records.</div></div>""", unsafe_allow_html=True)
    for n in data["notes"]:
        st.warning(n)
    c = st.columns(6)
    c[0].metric("Articles", f"{S['rows']:,}"); c[1].metric("Categories", S["n_classes"])
    c[2].metric("Avg. words", f"{S['words']['mean']:.1f}"); c[3].metric("Min / max words", f"{int(S['words']['min'])} / {int(S['words']['max'])}")
    c[4].metric("Avg. characters", f"{S['chars']['mean']:.1f}"); c[5].metric("Min / max chars", f"{int(S['chars']['min'])} / {int(S['chars']['max'])}")
    l, r = st.columns([3, 2])
    with l:
        st.pyplot(plots.bar_counts(S["counts"], "Articles per category"))
    with r:
        st.markdown("**Structure and quality checks**")
        st.table(pd.DataFrame({"Item": ["Text column", "Label column", "Missing text", "Missing label", "Duplicate headlines", "Largest / smallest class"],
                               "Value": [TXT, LBL, str(S["missing_text"]), str(S["missing_label"]), str(S["duplicates_text"]),
                                         f"{S['counts'].max():,} / {S['counts'].min():,} ({S['counts'].max() / S['counts'].min():.1f}x)"]}).set_index("Item"))
        st.markdown("**Category counts**")
        st.dataframe(S["counts"].rename("Articles").to_frame().assign(Share=lambda d: (d["Articles"] / d["Articles"].sum() * 100).round(2).astype(str) + "%"), width="stretch")
    st.markdown("**Sample of original data** (10 random rows, fixed seed)")
    st.dataframe(df.sample(10, random_state=1), width="stretch", hide_index=True)

# ---------------------------------------------------------------- Classification
with t_cls:
    st.markdown('<div class="section-kicker">NEWS DESK · CLASSIFICATION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-news">
        <div class="mini-press"></div>
        <div class="eyebrow">TOP STORY · AI CLASSIFICATION</div>
        <div class="headline">What is this Marathi story about?</div>
        <div class="summary">
            Paste a headline or short article and let the newsroom classifier identify its category
            using Marathi NLP, TF-IDF features and supervised machine learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    txt = st.text_area(
        "Marathi news",
        value=EXAMPLE,
        height=155,
        placeholder="उदा. मराठी बातमीचे शीर्षक किंवा लेख येथे लिहा...",
        key="news_input"
    )
    st.caption("The result updates automatically. After editing the text, click outside the box or press Ctrl+Enter to apply it.")

    st.markdown('<hr class="news-hr">', unsafe_allow_html=True)

    # All three models classify the text on every change; no model needs to be selected by hand.
    if not txt.strip():
        st.info("Enter Marathi news text above to see its predicted category.")
    else:
        order = [best] + sorted([m for m in R if m != best], key=lambda m: -R[m]["f1"])
        results, first_err = {}, None
        for m in order:
            res, err = prediction.predict(B, txt, m)
            if err:
                first_err = err
                break
            results[m] = res
        if first_err:
            st.error(first_err)
        else:
            preds = {m: results[m]["prediction"] for m in order}
            (top_cat, votes), = Counter(preds.values()).most_common(1)
            n = len(order)
            if votes == n:
                verdict, note, split = top_cat, f"All {n} models agree.", False
            elif votes >= 2:
                others = "; ".join(f"{html.escape(m)} predicts {html.escape(str(p))}" for m, p in preds.items() if p != top_cat)
                verdict, note, split = top_cat, f"{votes} of {n} models agree. {others}.", True
            else:
                verdict, split = preds[best], True
                note = f"The models disagree. Showing the prediction of {html.escape(best)}, the selected model."
            st.markdown(
                f'<div class="result-card">'
                f'<span class="news-label">Filed under</span>'
                f'<div class="result-category">{html.escape(str(verdict))}</div>'
                f'<div class="result-note{" split" if split else ""}">{note}</div>'
                f'<div class="result-meta">The category is the majority vote of the three models. '
                f'If all three differ, the selected model (highest Macro F1) decides.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
            for col, m in zip(st.columns(n), order):
                with col:
                    st.markdown(model_card(m, results[m], m == best), unsafe_allow_html=True)
            if any(results[m]["short_input"] for m in order):
                st.warning("Very short input (fewer than 3 informative tokens); the predictions may be unreliable.")
            for w in {results[m]["warning"] for m in order if results[m]["warning"]}:
                st.warning(w)
            t = results[best]["trace"]
            st.markdown("**Processed text passed to TF-IDF**"); mr(t["final_text"])
            st.caption(f"{len(t['tokens'])} tokens, {len(t['removed_stopwords'])} stopwords removed, {results[best]['known_features']} distinct TF-IDF features found in the training vocabulary. Full trace: NLP Preprocessing tab.")

# ---------------------------------------------------------------- Preprocessing
with t_pre:
    st.markdown('<div class="section-kicker">NEWSROOM · NLP PIPELINE</div>', unsafe_allow_html=True)
    src = st.radio("Input", ["Text from the News Classification tab", "Random article from the test split"], horizontal=True)
    if src.startswith("Random"):
        i = st.number_input("Test-set index", 0, B["n_test"] - 1, 0)
        text = B["X_test"].iloc[int(i)]; st.caption(f"Actual label: {B['y_test'].iloc[int(i)]}")
    else:
        text = st.session_state.get("news_input", EXAMPLE)
    if not str(text).strip():
        st.info("Enter text in the News Classification tab first.")
    else:
        t = pp.process(text)
        st.markdown("**Step 1 - Original text**"); mr(t["original"])
        st.markdown("**Step 2 - Normalization**"); mr(t["normalized"])
        st.write("Operations that changed this text:" if t["applied_steps"] else "No normalization operation changed this text.")
        for s in t["applied_steps"]:
            st.write("- " + s)
        st.markdown("**Step 3 - Tokenization**"); mr(" | ".join(t["tokens"])); st.caption(f"{len(t['tokens'])} tokens")
        st.markdown("**Step 4 - Stopword removal**")
        c1, c2, c3 = st.columns(3)
        c1.write("Before"); c1.markdown(f'<div class="mr">{" | ".join(t["tokens"])}</div>', unsafe_allow_html=True)
        c2.write("Removed stopwords"); c2.markdown(f'<div class="mr">{" | ".join(t["removed_stopwords"]) or "none"}</div>', unsafe_allow_html=True)
        c3.write("After"); c3.markdown(f'<div class="mr">{" | ".join(t["tokens_after"])}</div>', unsafe_allow_html=True)
        st.caption(f"Stopword list: data/marathi_stopwords.txt ({len(pp.load_stopwords())} entries)")
        st.markdown("**Step 5 - Final processed text (input to TF-IDF)**"); mr(t["final_text"])
        vec = B["pipelines"][best].named_steps["tfidf"]
        analyzer_terms = [w for w in vec.build_analyzer()(text) if " " not in w]
        st.caption("Consistency check against the trained vectorizer: " + ("identical" if analyzer_terms == t["tokens_after"] else "MISMATCH"))
        feats, shape, nnz = fe.sample_features(vec, text)
        st.markdown("**TF-IDF representation of this text**")
        st.caption(f"Sparse vector of dimension {shape[1]:,} with {nnz} non-zero entries (word unigrams and bigrams). Highest weights:")
        st.dataframe(pd.DataFrame(feats, columns=["Term", "TF-IDF weight"]).round(4), width="stretch", hide_index=True)

    # Pipeline diagram: bottom of the tab, centred
    with st.container(key="pipeline_center"):
        st.markdown('<div class="pipeline-heading">Pipeline</div>', unsafe_allow_html=True)
        _left, _mid, _right = st.columns([1, 1, 1])
        with _mid:
            pipeline_graph()

# ---------------------------------------------------------------- Evaluation
with t_eval:
    st.markdown('<div class="section-kicker">NEWSROOM · MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.subheader("Multi-Model Performance Comparison")
    st.caption(f"All three models: same stratified {int((1 - B['test_size']) * 100)}/{int(B['test_size'] * 100)} split (random_state={B['seed']}), same TF-IDF, evaluated on the same {B['n_test']:,} held-out articles. Precision, recall and F1 are macro-averaged (unweighted mean over classes) because classes are imbalanced.")
    tb = B["table"].copy()
    for m in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
        tb[m] = tb[m].map(pct)
    st.table(tb.set_index("Model Name"))
    bm = R[best]
    st.markdown(f"**Selected model: {best}**")
    st.write(f"Selected because it achieved the highest Macro F1 score ({pct(bm['f1'])}) on the common held-out test set. This holds for this dataset and this split only; it is not a general ranking of the algorithms.")
    cs = st.columns(4)
    for col, (lab, k) in zip(cs, [("Accuracy", "accuracy"), ("Macro Precision", "precision"), ("Macro Recall", "recall"), ("Macro F1", "f1")]):
        col.metric(lab, pct(bm[k]))
    ranked = sorted(R, key=lambda m: -R[m]["f1"])
    if len(ranked) > 1:
        gap = (R[ranked[0]]["f1"] - R[ranked[1]]["f1"]) * 100
        st.write(f"Macro F1 margin over {ranked[1]}: {gap:.2f} percentage points" + (" (small; a single split cannot establish a significant difference)." if gap < 1 else "."))
    GENERAL = {"Linear SVM": "Linear SVM maximizes the margin between classes and copes well with high-dimensional sparse TF-IDF vectors.",
               "Logistic Regression": "Logistic Regression learns a linear weight per term and suits sparse TF-IDF vectors; class weighting helps with imbalance.",
               "Multinomial Naive Bayes": "Multinomial Naive Bayes models term frequencies per class and is fast and strong on small or sparse text data."}
    st.markdown("**Why this model?**")
    st.write(f"Measured: {best} scored highest Macro F1 on the held-out set. General property (not measured here): {GENERAL[best]}")
    st.pyplot(plots.metric_bars(B["table"]))
    st.subheader("Confusion matrices")
    c1, c2 = st.columns([1, 3])
    with c1:
        cm_model = st.selectbox("Model", list(R), index=list(R).index(best))
        norm = st.radio("Values", ["Counts", "Row-normalised (recall)"])
        st.caption("Rows = actual category, columns = predicted category.")
    with c2:
        st.pyplot(plots.confusion_heatmap(R[cm_model]["confusion"], classes, norm.startswith("Row"), cm_model))
    st.subheader("Class-wise scores")
    c1, c2 = st.columns(2)
    with c1:
        pm = st.selectbox("Model ", list(R), index=list(R).index(best), key="pcm")
        st.pyplot(plots.class_f1(R[pm]["per_class"]))
    with c2:
        st.dataframe(R[pm]["per_class"].round(4), width="stretch", hide_index=True)
    st.subheader("Train / test distribution")
    st.pyplot(plots.split_bars(B["train_counts"], B["test_counts"]))
    st.subheader("TF-IDF features")
    ti = B["tfidf_info"]
    c1, c2 = st.columns([3, 2])
    c1.dataframe(pd.DataFrame(ti["top_terms"], columns=["Term", "Mean TF-IDF weight (train)"]), width="stretch", hide_index=True, column_config={"Mean TF-IDF weight (train)": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=float(ti["top_terms"][0][1]))})
    c2.write(f"Vocabulary size: {ti['vocab_size']:,}"); c2.write(f"Training matrix: {ti['train_shape'][0]:,} x {ti['train_shape'][1]:,}")
    c2.write(f"Test matrix: {ti['test_shape'][0]:,} x {ti['test_shape'][1]:,}"); c2.write(f"Avg. non-zero features per document: {ti['nnz_per_doc']:.1f} ({ti['density_pct']:.3f}% density)")
    c2.caption("TF-IDF turns each text into a sparse numeric vector: term frequency (sublinear) times inverse document frequency, L2-normalised.")
    st.subheader("Model details")
    with st.expander("Multinomial Naive Bayes"):
        st.write("Applies Bayes' theorem assuming terms are conditionally independent given the class; it estimates P(term | class) from (TF-IDF-weighted) term counts with Laplace smoothing and predicts the class with the highest posterior. Common for text because training is fast and it works with sparse counts. Here it uses default alpha=1.0 and cannot use class weights, so it is more affected by the class imbalance.")
    with st.expander("Logistic Regression"):
        st.write("Learns a weight vector per class over the TF-IDF features and converts scores into class probabilities (softmax for multiclass). Trained with L2 regularization (C=1.0) and class_weight='balanced'. Weights indicate which terms push a document toward a class.")
    with st.expander("Linear SVM"):
        st.write("Finds the hyperplane that separates classes with the maximum margin (one-vs-rest for multiclass, scikit-learn LinearSVC, C=1.0, class_weight='balanced'). Works well in high-dimensional sparse spaces such as TF-IDF. It outputs decision scores (signed distance from each hyperplane), not probabilities; the app labels them accordingly.")

# ---------------------------------------------------------------- Details
with t_info:
    st.markdown('<div class="section-kicker">NEWSROOM · PROJECT FILE</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset**")
        st.table(pd.DataFrame({"Item": ["File", "Records", "Categories", "Training samples", "Testing samples"],
                               "Value": [S["file"], f"{S['rows']:,}", str(S["n_classes"]), f"{B['n_train']:,}", f"{B['n_test']:,}"]}).set_index("Item"))
        st.write("Category names: " + ", ".join(classes))
        st.markdown("**NLP techniques**")
        st.write("- Text normalization (NFC, zero-width removal, digit mapping, punctuation, whitespace)\n- Regex tokenization for Devanagari\n- Stopword removal (custom list in data/marathi_stopwords.txt)\n- TF-IDF, word unigrams + bigrams, min_df=2, sublinear TF\n- Not used: stemming/lemmatization")
    with c2:
        st.markdown("**Machine learning models**"); st.write("- Multinomial Naive Bayes\n- Logistic Regression\n- Linear SVM")
        st.markdown("**Evaluation metrics**"); st.write("- Accuracy\n- Macro Precision\n- Macro Recall\n- Macro F1 (selection criterion)")
        st.markdown("**Final model**"); st.write(f"{best} (Macro F1 {pct(R[best]['f1'])} on the held-out test set)")
    st.markdown("**System architecture**"); pipeline_graph()