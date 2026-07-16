import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

# ===============================
# Load models
# ===============================
EFF_MODEL = "eff_model.keras"
CNN_MODEL = "structural_crack_cnn_model.h5"
META_MODEL = "meta_stack_logreg.pkl"

eff = tf.keras.models.load_model(EFF_MODEL, compile=False)
cnn = tf.keras.models.load_model(CNN_MODEL, compile=False)
meta = joblib.load(META_MODEL)

IMG_SIZE = (224, 224)


# ===============================
# Helper: load + resize image
# ===============================
def load_image(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return arr.astype("float32")


# ===============================
# Severity logic (simple 3-tier)
# ===============================
def severity_label(score):
    if score < 0.15:
        return "MILD"
    elif score < 0.45:
        return "MODERATE"
    else:
        return "SEVERE"


def maintenance(level):
    if level == "MILD":
        return "Monitor regularly, recheck in 1–3 months."
    elif level == "MODERATE":
        return "Schedule inspection within 1–2 weeks."
    else:  # SEVERE
        return "Immediate structural engineer inspection recommended."


# ===============================
# Combined prediction
# ===============================
def predict(arr):

    x = arr[np.newaxis, ...]

    # --- MODEL OUTPUTS ---
    cnn_prob = float(cnn.predict(x, verbose=0).flatten()[0])
    eff_prob = float(eff.predict(x, verbose=0).flatten()[0])

    # Meta combined probability
    combined_prob = float(meta.predict_proba([[cnn_prob, eff_prob]])[:, 1])
    detected = combined_prob >= 0.5

    # ===============================
    # FIXED SEVERITY LOGIC
    # If crack NOT detected → NO CRACK severity
    # ===============================
    if not detected:
        sev_label = "NO CRACK"
        summary = "No crack detected — no maintenance required."
    else:
        sev_label = severity_label(eff_prob)
        summary = maintenance(sev_label)

    return {
        "cnn_prob": cnn_prob,
        "eff_prob": eff_prob,
        "combined_prob": combined_prob,
        "crack_detected": "YES" if detected else "NO",
        "severity_score": eff_prob,
        "severity_label": sev_label,
        "maintenance_summary": summary
    }


# ===============================
# Streamlit UI
# ===============================
st.title("🧱 Structural Crack Detection & Severity Estimation")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    arr = load_image(img)
    out = predict(arr)

    st.markdown("---")
    st.markdown("## 🔍 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Crack detected")
        st.write(f"**{out['crack_detected']}**")
        st.write(f"Combined prob: {out['combined_prob']:.3f}")

    with col2:
        st.write(f"### Severity: **{out['severity_label']}**")
        st.write(out["maintenance_summary"])
