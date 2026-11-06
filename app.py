"""
Streamlit dashboard for YOLO11s training results
Classes: Clean Water, PET plastic, Polyester, Polyethylene, Polypropylene

Run:
    pip install streamlit ultralytics pandas plotly pyyaml pillow
    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from PIL import Image
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEFAULT_DIR = Path.home() / "Downloads" / "model2" / "train"
CLASS_NAMES = ["Clean Water", "PET plastic", "Polyester", "Polyethylene", "Polypropylene"]

st.set_page_config(page_title="YOLO Results Dashboard", page_icon="🧪", layout="wide")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🧪 YOLO11s Dashboard")
train_dir = Path(st.sidebar.text_input("Path to `train` folder", str(DEFAULT_DIR)))

if not train_dir.exists():
    st.error(f"Folder not found: `{train_dir}`. Update the path in the sidebar.")
    st.stop()

st.sidebar.success(f"Loaded: {train_dir.name}")
st.sidebar.caption("Model: YOLO11s · imgsz 768 · 100 epochs · 566 images (90/10 split)")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data
def load_args(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def show_image(name: str, caption: str | None = None):
    p = train_dir / name
    if p.exists():
        st.image(Image.open(p), caption=caption or name, use_container_width=True)
    else:
        st.warning(f"`{name}` not found")


def line_fig(df, cols, title):
    fig = go.Figure()
    for c in cols:
        if c in df:
            fig.add_trace(go.Scatter(x=df["epoch"], y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, xaxis_title="Epoch", height=350,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Microplastic Detection Dashboard")
st.caption("Interactive view of the YOLO11s training run: metrics, curves, confusion matrices, "
           "sample batches, validation predictions and live inference.")

results_path = train_dir / "results.csv"
df = load_results(str(results_path)) if results_path.exists() else None

tabs = st.tabs(["📊 Overview", "📈 Training Curves", "🎯 PR / F1 Curves",
                "🧩 Confusion Matrix", "🖼️ Dataset & Batches", "🔍 Validation Predictions",
                "⚙️ Config", "🚀 Try the Model"])

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    if df is None:
        st.warning("results.csv not found")
    else:
        last = df.iloc[-1]
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best = df.loc[best_idx]

        st.subheader("Final epoch metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{last['metrics/precision(B)']:.3f}")
        c2.metric("Recall", f"{last['metrics/recall(B)']:.3f}")
        c3.metric("mAP@50", f"{last['metrics/mAP50(B)']:.3f}")

        st.subheader("Best epoch (by mAP@50)")
        st.info(f"Epoch **{int(best['epoch'])}** — "
                f"P {best['metrics/precision(B)']:.3f} · R {best['metrics/recall(B)']:.3f} · "
                f"mAP50 {best['metrics/mAP50(B)']:.3f}")

        st.plotly_chart(
            line_fig(df, ["metrics/mAP50(B)",
                          "metrics/precision(B)", "metrics/recall(B)"], "Validation metrics over epochs"),
            use_container_width=True)

        st.subheader("Training summary image")
        show_image("results.png", "Ultralytics results.png")

# ----------------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------------
with tabs[1]:
    if df is None:
        st.warning("results.csv not found")
    else:
        pairs = [("box_loss", "Box loss"), ("cls_loss", "Class loss"), ("dfl_loss", "DFL loss")]
        fig = make_subplots(rows=1, cols=3, subplot_titles=[p[1] for p in pairs])
        for i, (k, _) in enumerate(pairs, start=1):
            for split, dash in (("train", "solid"), ("val", "dash")):
                col = f"{split}/{k}"
                if col in df:
                    fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], name=col,
                                             line=dict(dash=dash)), row=1, col=i)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        if "lr/pg0" in df:
            st.plotly_chart(line_fig(df, ["lr/pg0", "lr/pg1", "lr/pg2"], "Learning rate"),
                            use_container_width=True)

        with st.expander("Raw results.csv"):
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")

# ----------------------------------------------------------------------------
# PR / F1 curves
# ----------------------------------------------------------------------------
with tabs[2]:
    curves = [("BoxPR_curve.png", "Precision–Recall"), ("BoxF1_curve.png", "F1–Confidence"),
              ("BoxP_curve.png", "Precision–Confidence"), ("BoxR_curve.png", "Recall–Confidence")]
    cols = st.columns(2)
    for i, (f, t) in enumerate(curves):
        with cols[i % 2]:
            st.markdown(f"**{t}**")
            show_image(f, t)

# ----------------------------------------------------------------------------
# Confusion matrices
# ----------------------------------------------------------------------------
with tabs[3]:
    choice = st.radio("View", ["Normalized", "Raw counts", "Side by side"], horizontal=True)
    if choice == "Normalized":
        show_image("confusion_matrix_normalized.png")
    elif choice == "Raw counts":
        show_image("confusion_matrix.png")
    else:
        a, b = st.columns(2)
        with a:
            show_image("confusion_matrix.png", "Counts")
        with b:
            show_image("confusion_matrix_normalized.png", "Normalized")
    st.caption("Rows/columns are the 5 classes plus *background*. Strong diagonal = good detection.")

# ----------------------------------------------------------------------------
# Dataset & batches
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Label distribution")
    show_image("labels.jpg", "Class counts, box sizes and positions")

    st.subheader("Training batches (mosaic augmentation)")
    batch_files = sorted(train_dir.glob("train_batch*.jpg"),
                         key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    if batch_files:
        sel = st.select_slider("Batch", options=[p.name for p in batch_files])
        show_image(sel)

# ----------------------------------------------------------------------------
# Validation predictions
# ----------------------------------------------------------------------------
with tabs[5]:
    st.caption("Left: ground-truth labels. Right: model predictions.")
    n_batches = len(list(train_dir.glob("val_batch*_pred.jpg")))
    for i in range(n_batches):
        st.markdown(f"#### Validation batch {i}")
        a, b = st.columns(2)
        with a:
            show_image(f"val_batch{i}_labels.jpg", "Ground truth")
        with b:
            show_image(f"val_batch{i}_pred.jpg", "Predictions")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
with tabs[6]:
    args_path = train_dir / "args.yaml"
    if args_path.exists():
        args = load_args(str(args_path))
        key_params = {k: args.get(k) for k in
                      ["model", "data", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf",
                       "patience", "device", "seed"] if k in args}
        st.subheader("Key hyperparameters")
        st.table(pd.DataFrame(key_params.items(), columns=["Parameter", "Value"]).astype(str))
        with st.expander("Full args.yaml"):
            st.json(args)
    else:
        st.warning("args.yaml not found")

    st.subheader("Classes")
    st.write(", ".join(f"`{c}`" for c in CLASS_NAMES))

    weights = train_dir / "weights"
    if weights.exists():
        st.subheader("Weights")
        for w in sorted(weights.glob("*.pt")):
            st.write(f"📦 `{w.name}` — {w.stat().st_size / 1e6:.1f} MB")

# ----------------------------------------------------------------------------
# Live inference
# ----------------------------------------------------------------------------
with tabs[7]:
    best_pt = train_dir / "weights" / "best.pt"
    if not best_pt.exists():
        st.warning("weights/best.pt not found")
    else:
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        uploads = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True)
        if uploads:
            with st.spinner("Loading model..."):
                model = load_model(str(best_pt))
            for up in uploads:
                img = Image.open(up).convert("RGB")
                res = model.predict(img, conf=conf, imgsz=768, verbose=False)[0]
                annotated = Image.fromarray(res.plot()[..., ::-1])  # BGR -> RGB

                a, b = st.columns(2)
                a.image(img, caption=f"Original — {up.name}", use_container_width=True)
                b.image(annotated, caption="Prediction", use_container_width=True)

                if len(res.boxes):
                    det = pd.DataFrame({
                        "Class": [res.names[int(c)] for c in res.boxes.cls],
                        "Confidence": [round(float(c), 3) for c in res.boxes.conf],
                    })
                    st.dataframe(det, use_container_width=True, hide_index=True)
                else:
                    st.info("No detections at this threshold.")
                st.divider()