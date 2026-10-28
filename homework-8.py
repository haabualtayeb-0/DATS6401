
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
CLASSES = ["cats", "dogs", "flowers"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SIZE = 128

st.set_page_config(page_title="Image Dataset Browser", layout="wide")


def to_array(img):
    """Same preprocessing as the notebook: RGB, 128x128."""
    return np.array(img.convert("RGB").resize((SIZE, SIZE)))


def to_features(arr):
    """What the models see: 32x32, scaled to 0-1."""
    return arr[::4, ::4] / 255


@st.cache_data
def load_data():
    classes = [c for c in CLASSES if (ROOT / c).is_dir()]
    images, labels, paths = [], [], []
    for cls in classes:
        for p in sorted((ROOT / cls).iterdir()):
            if p.suffix.lower() in EXTS:
                images.append(to_array(Image.open(p)))
                labels.append(cls)
                paths.append(str(p))
    return classes, np.stack(images), np.array(labels), paths


@st.cache_resource
def train_models():
    _, images, labels, _ = load_data()
    small = images[:, ::4, ::4] / 255
    Xtr, Xte, ytr, yte = train_test_split(small, labels, test_size=0.3,
                                          random_state=0, stratify=labels)
    Xtr_flat, Xte_flat = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    models = {
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        "Logistic regression": LogisticRegression(max_iter=2000),
    }
    scores = {}
    for name, m in models.items():
        m.fit(Xtr_flat, ytr)
        scores[name] = m.score(Xte_flat, yte)
    return models, scores


def occlude(model, img, cls, patch=9):
    """Slide a mean-colored patch over the image; record how far P(cls) falls."""
    k = list(model.classes_).index(cls)
    h, w = img.shape[:2]
    r, fill = patch // 2, img.mean(axis=(0, 1))
    base = model.predict_proba([img.ravel()])[0][k]
    batch = []
    for i in range(h):
        for j in range(w):
            t = img.copy()
            t[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = fill
            batch.append(t.ravel())
    return (base - model.predict_proba(np.array(batch))[:, k]).reshape(h, w)


classes, images, labels, paths = load_data()

st.sidebar.title("Image Dataset Browser")
page = st.sidebar.radio("Page", ["Browse Images", "Dataset Overview", "Saliency"])
st.sidebar.caption(f"{len(images)} images · {len(classes)} classes")



if page == "Browse Images":
    st.header("Browse Images")
    c1, c2 = st.columns([1, 3])
    cls = c1.selectbox("Class", classes, format_func=str.capitalize)
    idx = np.where(labels == cls)[0]
    per_page = 24
    n_pages = int(np.ceil(len(idx) / per_page))
    pg = c2.slider("Page", 1, n_pages, 1) if n_pages > 1 else 1
    shown = idx[(pg - 1) * per_page: pg * per_page]

    cols = st.columns(6)
    for n, i in enumerate(shown):
        cols[n % 6].image(images[i], caption=Path(paths[i]).name, width="stretch")

    st.subheader("Inspect One Image")
    pick = st.selectbox("Image", shown, format_func=lambda i: Path(paths[i]).name)
    img = images[pick]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
    axes[0].imshow(img); axes[0].set_title(f"Composite: {cls.capitalize()}")
    for k, (n, cm) in enumerate([("R", "Reds"), ("G", "Greens"), ("B", "Blues")]):
        axes[k + 1].imshow(img[..., k], cmap=cm, vmin=0, vmax=255)
        axes[k + 1].set_title(f"{n} (mean {img[..., k].mean():.0f})")
    for a in axes: a.axis("off")
    plt.tight_layout(); st.pyplot(fig)



elif page == "Dataset Overview":
    st.header("Dataset Overview")

    st.subheader("Class Balance")
    counts = pd.Series(labels).value_counts().sort_index()
    counts.index = counts.index.str.capitalize()
    fig, ax = plt.subplots(figsize=(7, 2.4))
    counts.plot.bar(ax=ax, color="#2E6E8E", rot=0)
    ax.set_ylabel("Images"); st.pyplot(fig)

    st.subheader("Mean Image per Class")
    st.caption("Blurry means the class is varied; a clear shape means similar framing.")
    fig, axes = plt.subplots(1, len(classes), figsize=(2.6 * len(classes), 2.6))
    for cls, ax in zip(classes, axes):
        ax.imshow(images[labels == cls].mean(axis=0) / 255)
        ax.set_title(cls.capitalize(), fontsize=9); ax.axis("off")
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Compare Two Classes' Intensity")
    c1, c2 = st.columns(2)
    a = c1.selectbox("Class A", classes, index=classes.index("cats"), format_func=str.capitalize)
    b = c2.selectbox("Class B", classes, index=classes.index("dogs"), format_func=str.capitalize)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for cls, color in zip([a, b], ["#2E6E8E", "#d9534f"]):
        vals = images[labels == cls].ravel()
        ax.hist(vals, bins=64, range=(0, 255), alpha=0.55, density=True,
                label=f"{cls.capitalize()} (mean {vals.mean():.1f})", color=color)
    ax.legend(); ax.set_xlabel("Pixel intensity (0–255, all RGB channels)")
    st.pyplot(fig)



else:
    st.header("Occlusion Saliency")
    st.write("A patch slides over the image; where covering it lowers the model's "
             "confidence, the map turns **red**. Blue means covering it *raised* confidence.")

    with st.spinner("Training models (first time only)…"):
        models, scores = train_models()

    c1, c2 = st.columns(2)
    name = c1.selectbox("Model", list(models),
                        format_func=lambda n: f"{n} ({100 * scores[n]:.1f}% test accuracy)")
    model = models[name]
    patch = c2.slider("Patch size (pixels on the 32×32 model input)", 3, 15, 9, step=2)

    source = st.radio("Image source", ["Upload an image", "Pick from dataset"], horizontal=True)
    arr = None
    if source == "Upload an image":
        up = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "bmp"])
        if up is not None:
            arr = to_array(Image.open(up))
    else:
        cls = st.selectbox("Class", classes, format_func=str.capitalize)
        idx = np.where(labels == cls)[0]
        pick = st.selectbox("Image", idx, format_func=lambda i: Path(paths[i]).name)
        arr = images[pick]

    if arr is not None:
        x = to_features(arr)
        proba = model.predict_proba([x.ravel()])[0]
        pred = model.classes_[np.argmax(proba)]
        st.write(f"Prediction: **{pred.capitalize()}** ({proba.max():.0%})")
        target = st.selectbox("Explain which class?", list(model.classes_),
                              index=list(model.classes_).index(pred), format_func=str.capitalize)

        with st.spinner("Computing saliency…"):
            sal = occlude(model, x, target, patch)
        lim = np.abs(sal).max() or 1e-9
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.55)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        ext = (-0.5, SIZE - 0.5, SIZE - 0.5, -0.5)   # Stretch the 32x32 map over the 128x128 image.
        axes[0].imshow(arr); axes[0].set_title("Image")
        hm = axes[1].imshow(sal, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[1].set_title(f"Drop in P({target.capitalize()})")
        fig.colorbar(hm, ax=axes[1], fraction=0.046)
        axes[2].imshow(arr)
        axes[2].imshow(sal, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=alpha,
                       extent=ext, interpolation="bilinear")
        axes[2].set_title("Overlay")
        for a in axes: a.set_xticks([]); a.set_yticks([])
        plt.tight_layout(); st.pyplot(fig)

        st.caption("If red sits on the background or edges rather than the object, the model "
                   "is using context (framing, image source) instead of the object itself.")

        probs = pd.Series(proba, index=[c.capitalize() for c in model.classes_]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 2.6))
        probs.plot.barh(ax=ax, color="#2E6E8E"); ax.set_xlabel("Probability")
        st.pyplot(fig)
