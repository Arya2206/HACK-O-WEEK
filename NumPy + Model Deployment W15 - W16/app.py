"""Interactive Streamlit demo for the from-scratch two-moons neural network."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_moons

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"

st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")
st.title("🧠 Neural Network Playground")
st.write(
    "Explore the two-moons classifier. Change the two input values, make a prediction, "
    "and see where the point falls on the model's decision map."
)
st.caption("Predictions are shown here and printed in the VS Code terminal whenever you change an input.")

@st.cache_resource
def load_model(path_string: str, modified_time: float):
    # Pickle/joblib files can execute code while loading. Only use trusted model files.
    return joblib.load(path_string)

if not MODEL_PATH.exists():
    st.error("I couldn't find model.joblib in this folder.")
    st.code("python train.py\npython -m streamlit run app.py", language="powershell")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info("Open the VS Code terminal in this project folder and run `python train.py`, then refresh this page.")
    st.stop()

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader("Try an input")
    x1 = st.slider("Input 1 (x₁)", min_value=-2.0, max_value=3.0, value=0.0, step=0.05)
    x2 = st.slider("Input 2 (x₂)", min_value=-1.5, max_value=2.0, value=0.5, step=0.05)
    point = np.array([[x1, x2]], dtype=float)
    probability = float(np.asarray(model.predict_proba(point)).reshape(-1)[0])
    predicted_class = int(np.asarray(model.predict(point)).reshape(-1)[0])
    # Also print each prediction to the VS Code terminal.
    print(
        f"[Web app prediction] input=({x1:.2f}, {x2:.2f}) | "
        f"class={predicted_class} | probability_class_1={probability:.4f}",
        flush=True,
    )
    st.metric("Predicted class", str(predicted_class))
    st.metric("Probability of class 1", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if predicted_class == 1:
        st.success("The model classifies this point as class 1.")
    else:
        st.info("The model classifies this point as class 0.")
    st.caption("The sliders provide the two numeric features expected by this model.")

with right:
    st.subheader("Decision map")
    X, y = make_moons(n_samples=600, noise=0.2, random_state=0)
    xx, yy = np.meshgrid(np.linspace(-2.0, 3.0, 180), np.linspace(-1.5, 2.0, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = np.asarray(model.predict_proba(grid)).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5.3))
    contour = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 21), alpha=0.55, cmap="coolwarm")
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=15, alpha=0.65, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=15, alpha=0.65, label="Class 1")
    ax.scatter([x1], [x2], marker="*", s=240, edgecolors="black", linewidths=1.2, label="Your input")
    ax.set_xlabel("Input 1 (x₁)")
    ax.set_ylabel("Input 2 (x₂)")
    ax.set_title("Model probability and classification boundary")
    ax.legend(loc="upper right")
    fig.colorbar(contour, ax=ax, label="Probability of class 1")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with st.expander("What am I looking at?"):
    st.markdown(
        "- **Dots:** example points from the two-moons dataset.\n"
        "- **Background colors:** the model's estimated probability of class 1.\n"
        "- **Black line:** the 50% decision boundary.\n"
        "- **Star:** the point selected with the sliders.\n\n"
        "This is a demo dataset with two numeric inputs, not a general-purpose predictor for arbitrary data."
    )

st.caption("Model source: model.joblib · Built with Streamlit, NumPy, scikit-learn and Matplotlib.")
