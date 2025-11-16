import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# === CONFIG ===
IMG_SIZE = (224, 224)

# MUST match the class order used in training (ImageDataGenerator class_indices)
CLASS_NAMES = [
    "Body Tissue or Organ",
    "Gauze",
    "Glass Equipment Package",
    "Glove Pair Latex",
    "Glove Pair Nitrile",
    "Glove Pair Surgery",
    "Glove Single Latex",
    "Glove Single Nitrile",
    "Glove Single Surgery",
    "Medical Cap",
    "Mask",
    "Medical Glasses",
    "Metal Equipment Package",
    "Organic Waste",
    "Paper Equipment Package",
    "Plastic Equipment Package",
    "Shoe Cover Pair",
    "Shoe Cover Single",
    "Syringe",
    "Syringe Needle",
    "Test Tube",
    "Tweezers",
    "Urine Bag",
]

# === LOAD MODEL ONCE ===
@st.cache_resource
def load_model():
    # If you put it in a subfolder, e.g. "models/medicalwaste_googlenet.h5",
    # change the path here.
    model_path = "medicalwaste_googlenet.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# === PREPROCESSING (must match your training pipeline) ===
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Ensure RGB, resize to 224x224, normalize 0–1 (like rescale=1./255.)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr

# === STREAMLIT UI ===
st.title("EcoMed – Medical Waste Classifier 🩺🗑️")
st.write("Upload an image of medical waste and let the GoogLeNet model classify it.")

uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Analyzing image with GoogLeNet..."):
            model = load_model()
            x = preprocess_image(image)
            preds = model.predict(x)
            class_idx = int(np.argmax(preds[0]))
            confidence = float(preds[0][class_idx])
            label = CLASS_NAMES[class_idx]

        st.subheader("Prediction")
        st.write(f"**Class:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        # Optional: show full probability distribution
        st.write("Class probabilities:")
        prob_dict = {
            CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))
        }
        st.dataframe(
            sorted(prob_dict.items(), key=lambda x: x[1], reverse=True),
            use_container_width=True,
        )
