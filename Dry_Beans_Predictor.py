"""🫘 Dry Bean Classifier — Streamlit app.

Enter a bean's physical measurements and get the predicted variety
(Barbunya, Bombay, Cali, Dermason, Horoz, Seker or Sira) from the tuned SVM
trained in the Dry_Beans_Classifications notebook.

Run:  streamlit run Dry_Beans_Predictor.py
"""
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Bean Lens · Dry Bean Classifier", page_icon="🫘", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
DATA_PATH = BASE_DIR / "Dry_Beans_Dataset.xlsx"

# ---------------------------------------------------------------- theme ----
CLASS_COLORS = {
    "BARBUNYA": "#B5483A", "BOMBAY": "#6B4A8F", "CALI": "#D08A2E",
    "DERMASON": "#2F6B4F", "HOROZ": "#2C6E91", "SEKER": "#A3A33A", "SIRA": "#8A5A3B",
}
CLASS_BLURB = {
    "BARBUNYA": "Medium-large, plump beans with a rounded-oval outline.",
    "BOMBAY": "The giant of the group — by far the largest beans in the dataset.",
    "CALI": "Large, elongated kidney-type beans.",
    "DERMASON": "Small and smooth, and the most common variety in the data.",
    "HOROZ": "Slender and elongated — the most stretched-out outline.",
    "SEKER": "Small and nearly round — the roundest of all seven.",
    "SIRA": "Small-to-medium beans with a balanced, slightly oval shape.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, label, input { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.hero {
  position: relative; overflow: hidden; border-radius: 26px; padding: 2.4rem 2.6rem;
  background: radial-gradient(900px 340px at 105% -20%, #4C9A73 0%, rgba(76,154,115,0) 60%),
              linear-gradient(135deg, #1F4D38 0%, #2F6B4F 55%, #3C7F5F 100%);
  color: #FBF7EF; box-shadow: 0 18px 40px -22px rgba(31,77,56,.7); margin-bottom: 1.6rem;
}
.hero h1 { font-family: 'Fraunces', serif; font-weight: 800; font-size: 2.9rem; line-height: 1.05;
  margin: 0 0 .5rem 0; color: #FBF7EF; letter-spacing: -0.5px; }
.hero p { margin: 0; max-width: 620px; font-size: 1.05rem; color: #DDEBDD; line-height: 1.55; }
.hero .chips { margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.hero .chip { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
  padding: .25rem .8rem; border-radius: 999px; font-size: .8rem; font-weight: 500; }
.hero .beans { position: absolute; right: 2.2rem; top: 1.2rem; font-size: 4.4rem; opacity: .22;
  transform: rotate(-12deg); letter-spacing: .3rem; }

.panel-title { font-family: 'Fraunces', serif; font-size: 1.45rem; font-weight: 600; margin: 0 0 .15rem 0; color:#1F4D38; }
.panel-sub { color: #6E6A5E; font-size: .9rem; margin-bottom: .9rem; }

.result-card { border-radius: 24px; padding: 1.8rem 1.9rem; color: #fff; position: relative; overflow: hidden;
  box-shadow: 0 18px 36px -20px rgba(0,0,0,.45); margin-bottom: 1.1rem; }
.result-card::after { content: "🫘"; position: absolute; right: -22px; bottom: -48px; font-size: 7rem; opacity: .13; transform: rotate(18deg); }
.result-card .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; opacity: .85; font-weight: 600; }
.result-card .name { font-family: 'Fraunces', serif; font-weight: 800; font-size: 3.1rem; line-height: 1.05; margin: .25rem 0 .35rem; }
.result-card .blurb { font-size: .98rem; opacity: .95; max-width: 360px; }
.result-card .conf { margin-top: 1.1rem; display: inline-flex; align-items: baseline; gap: .55rem;
  background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3); padding: .4rem 1rem; border-radius: 999px; }
.result-card .conf b { font-size: 1.5rem; } .result-card .conf span { font-size: .82rem; opacity: .9; }

.mini { background: #fff; border: 1px solid #E7DEC9; border-radius: 16px; padding: .8rem 1rem; }
.mini .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: #8A8574; }
.mini .v { font-family: 'Fraunces', serif; font-size: 1.45rem; font-weight: 600; color: #1F4D38; }

.guide { background:#fff; border:1px solid #E7DEC9; border-left-width:7px; border-radius:16px; padding: .9rem 1.1rem; height: 100%; }
.guide h4 { font-family:'Fraunces',serif; margin:0 0 .2rem 0; font-size:1.15rem; }
.guide p { margin:0 0 .5rem 0; color:#5E5A4E; font-size:.88rem; }
.guide .st { font-size:.78rem; color:#8A8574; }

div[data-testid="stNumberInput"] input { border-radius: 12px; font-weight: 500; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 20px; }
button[kind="secondary"] { border-radius: 12px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- model ----
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """Load the saved model, training it from the dataset if it is missing."""
    if MODEL_PATH.is_file():
        return joblib.load(MODEL_PATH)
    if not DATA_PATH.is_file():
        st.error(
            "`model.joblib` and `Dry_Beans_Dataset.xlsx` are both missing. "
            "Place the dataset beside this app so the model can be trained."
        )
        st.stop()
    from train_model import train

    st.info("Training the classifier for first-time use. This may take a few minutes.")
    train(DATA_PATH, MODEL_PATH)
    return joblib.load(MODEL_PATH)


art = load_artifacts()
CLASSES, RANGES, PRESETS = art["classes"], art["ranges"], art["presets"]


def derive_features(area, perim, major, minor, convex, extent):
    """Compute the 16 model features from 6 raw measurements (formulas verified on the dataset)."""
    eq = np.sqrt(4 * area / np.pi)
    return {
        "Area": area, "Perimeter": perim, "MajorAxisLength": major, "MinorAxisLength": minor,
        "AspectRation": major / minor, "Eccentricity": np.sqrt(1 - (minor / major) ** 2),
        "ConvexArea": convex, "EquivDiameter": eq, "Extent": extent, "Solidity": area / convex,
        "roundness": 4 * np.pi * area / perim ** 2, "Compactness": eq / major,
        "ShapeFactor1": major / area, "ShapeFactor2": area / major ** 3,
        "ShapeFactor3": area / ((major / 2) ** 2 * np.pi),
        "ShapeFactor4": area / ((major / 2) * (minor / 2) * np.pi),
    }


def predict(feats):
    """Same preprocessing as the notebook: cap → log1p → standardise → SVM."""
    x = pd.DataFrame([feats])[art["features"]]
    x = x.clip(lower=art["lower"], upper=art["upper"], axis=1)
    x[art["skewed"]] = np.log1p(x[art["skewed"]])
    x = pd.DataFrame(art["scaler"].transform(x), columns=art["features"])
    return art["model"].predict_proba(x)[0]


# --------------------------------------------------------------- state -----
FIELDS = {  # key: (label, help, step, format)
    "Area": ("Area (px²)", "Number of pixels inside the bean's boundary.", 100.0, "%.0f"),
    "Perimeter": ("Perimeter (px)", "Length of the bean's border.", 5.0, "%.2f"),
    "MajorAxisLength": ("Major axis length (px)", "Longest distance across the bean.", 2.0, "%.2f"),
    "MinorAxisLength": ("Minor axis length (px)", "Longest width perpendicular to the major axis.", 2.0, "%.2f"),
    "ConvexArea": ("Convex area (px²)", "Pixels in the smallest convex shape enclosing the bean.", 100.0, "%.0f"),
    "Extent": ("Extent", "Bean area ÷ bounding-box area (0–1).", 0.01, "%.4f"),
}


def load_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = float(v)


if "Area" not in st.session_state:
    load_preset("DERMASON")

# ---------------------------------------------------------------- hero -----
st.markdown(
    """
<div class="hero">
  <div class="beans">🫘🫘</div>
  <h1>Bean Lens</h1>
  <p>Measure a dry bean, type in its dimensions, and the model tells you which of
  seven varieties it is — instantly, with a confidence score.</p>
  <div class="chips"><span class="chip">7 varieties</span><span class="chip">Tuned SVM</span>
  <span class="chip">13,611 beans trained</span><span class="chip">%.1f%% test accuracy</span></div>
</div>
""" % (art["metrics"]["accuracy"] * 100),
    unsafe_allow_html=True,
)

# ------------------------------------------------------------- sidebar -----
with st.sidebar:
    st.markdown("### 🧪 Try a sample")
    st.caption("Fill the form with the typical (median) measurements of a variety.")
    for c in CLASSES:
        st.button(c.title(), key=f"p_{c}", on_click=load_preset, args=(c,), width="stretch")
    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "Measurements come from computer-vision analysis of bean photos, so they are in **pixels**. "
        "Remaining shape features (aspect ratio, roundness, compactness…) are calculated automatically."
    )

# ---------------------------------------------------------------- main -----
left, right = st.columns([5, 6], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Bean measurements</div>'
                    '<div class="panel-sub">Adjust any value — the prediction updates live.</div>', unsafe_allow_html=True)
        vals = {}
        keys = list(FIELDS)
        for i in range(0, len(keys), 2):
            cols = st.columns(2)
            for col, k in zip(cols, keys[i:i + 2]):
                label, hlp, step, fmt = FIELDS[k]
                lo, hi = RANGES[k]
                with col:
                    vals[k] = st.number_input(label, min_value=0.0, step=step, format=fmt, help=hlp, key=k)

# ----------------------------------------------------------- validation ----
A, P, L, l, C, E = (vals[k] for k in FIELDS)
errors = []
if min(A, P, L, l, C) <= 0:
    errors.append("Area, perimeter, axis lengths and convex area must all be greater than zero.")
else:
    if l > L:
        errors.append("The **minor axis** can't be longer than the **major axis**.")
    if C < A:
        errors.append("**Convex area** can't be smaller than the bean's **area**.")
    if P < 2 * np.sqrt(np.pi * A):
        errors.append("**Perimeter** is too short for that area (a circle is the shortest possible outline).")
if not (0 < E <= 1):
    errors.append("**Extent** must be between 0 and 1.")

with right:
    if errors:
        st.markdown('<div class="panel-title">Check your inputs</div>', unsafe_allow_html=True)
        for e in errors:
            st.error(e, icon="⚠️")
    else:
        feats = derive_features(A, P, L, l, C, E)
        proba = predict(feats)
        order = np.argsort(proba)[::-1]
        top = CLASSES[order[0]]
        color = CLASS_COLORS[top]

        st.markdown(
            f"""<div class="result-card" style="background: linear-gradient(135deg, {color} 0%, {color}CC 100%);">
              <div class="eyebrow">Predicted variety</div>
              <div class="name">{top.title()}</div>
              <div class="blurb">{CLASS_BLURB[top]}</div>
              <div class="conf"><b>{proba[order[0]] * 100:.1f}%</b><span>confidence</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

        out_of_range = [FIELDS[k][0] for k in FIELDS if not RANGES[k][0] * 0.9 <= vals[k] <= RANGES[k][1] * 1.1]
        if out_of_range:
            st.warning("Outside the range the model was trained on: " + ", ".join(out_of_range)
                       + ". Treat this prediction with caution.", icon="📏")
        elif proba[order[0]] < 0.6:
            st.info("This bean sits between varieties — the model is not very sure. "
                    "Double-check the measurements.", icon="🤔")

        # probability chart
        names = [CLASSES[i].title() for i in order][::-1]
        pvals = [proba[i] * 100 for i in order][::-1]
        cols_ = [CLASS_COLORS[CLASSES[i]] if i == order[0] else "#D9D1BC" for i in order][::-1]
        fig = go.Figure(go.Bar(
            x=pvals, y=names, orientation="h", marker=dict(color=cols_, cornerradius=8),
            text=[f"{v:.1f}%" for v in pvals], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            height=320, margin=dict(l=0, r=40, t=34, b=0), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", size=13, color="#2B2A26"),
            title=dict(text="Class probabilities", font=dict(family="Fraunces", size=18, color="#1F4D38"), x=0),
            xaxis=dict(range=[0, 112], showgrid=False, visible=False), yaxis=dict(showgrid=False),
            bargap=0.28,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        m = st.columns(3)
        for col, (k, v) in zip(m, [("Aspect ratio", feats["AspectRation"]),
                                    ("Roundness", feats["roundness"]),
                                    ("Solidity", feats["Solidity"])]):
            col.markdown(f'<div class="mini"><div class="k">{k}</div><div class="v">{v:.3f}</div></div>',
                         unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs -----
st.write("")
t1, t2, t3 = st.tabs(["📐 Calculated features", "📖 Bean field guide", "🤖 About the model"])

with t1:
    if errors:
        st.info("Fix the inputs above to see the 16 features sent to the model.")
    else:
        st.caption("These 16 values are exactly what the model receives (before capping, log-transform and scaling).")
        df_f = pd.DataFrame({"Feature": list(feats), "Value": [round(float(v), 6) for v in feats.values()]})
        c1, c2 = st.columns(2)
        c1.dataframe(df_f.iloc[:8], hide_index=True, width="stretch")
        c2.dataframe(df_f.iloc[8:], hide_index=True, width="stretch")

with t2:
    cols = st.columns(4)
    for i, c in enumerate(CLASSES):
        s = art["class_stats"][c]
        with cols[i % 4]:
            st.markdown(
                f"""<div class="guide" style="border-left-color:{CLASS_COLORS[c]}">
                <h4 style="color:{CLASS_COLORS[c]}">{c.title()}</h4><p>{CLASS_BLURB[c]}</p>
                <div class="st">Median area <b>{s['Area']:,.0f}</b> px² · aspect <b>{s['AspectRation']:.2f}</b><br>
                roundness <b>{s['roundness']:.2f}</b> · {art['class_share'][c]}% of dataset</div></div>""",
                unsafe_allow_html=True,
            )
            st.write("")

with t3:
    mt = art["metrics"]
    a, b, c_ = st.columns(3)
    a.metric("Test accuracy", f"{mt['accuracy'] * 100:.2f}%")
    b.metric("Macro F1-score", f"{mt['f1_macro']:.3f}")
    c_.metric("Training / test beans", f"{mt['n_train']:,} / {mt['n_test']:,}")
    st.markdown(
        """
**Pipeline** (identical to the notebook): cap outliers at the 1st/99th percentile → `log1p` on skewed features →
`StandardScaler` → **SVM (RBF, C = 10, γ = 0.1)** chosen via grid-search over 7 candidate algorithms.
Confidence scores come from sigmoid calibration of the SVM output.

**Good to know:** the model was trained on measurements from a specific imaging setup, so use pixel values
on the same scale as the dataset. Dermason / Sira / Barbunya are the most easily confused classes.
"""
    )