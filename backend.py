# backend.py
import io
from typing import Dict

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# === KONFIGURASI MODEL ===
IMG_SIZE = (224, 224)

# PENTING:
# Ganti isi CLASS_NAMES ini dengan list yang URUTANNYA
# sama persis dengan class_indices di Colab:
# sorted(train_generator_balanced.class_indices, key=class_indices.get)
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
# NOTE: idealnya kamu cek lagi di Colab agar urutannya 100% sama.

# === LOAD MODEL SEKALI SAJA SAAT SERVER START ===
print("Loading model...")
MODEL = tf.keras.models.load_model("medicalwaste_googlenet.h5")
print("Model loaded.")

# === FASTAPI APP ===
app = FastAPI(title="EcoMed Medical Waste API")

# Biar bisa diakses dari web/JS (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # nanti bisa dibatasi ke domain tertentu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image_bytes(img_bytes: bytes) -> np.ndarray:
    """Preprocess gambar seperti di training (resize + normalisasi)."""
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0  # sama seperti rescale=1./255
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
    return arr

@app.get("/")
def health_check() -> Dict[str, str]:
    """Endpoint simple untuk cek server hidup."""
    return {"status": "ok", "message": "EcoMed backend is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Terima 1 file gambar,
    kembalikan label + confidence + semua probabilitas.
    """
    # Baca bytes gambar
    img_bytes = await file.read()

    # Preprocess
    x = preprocess_image_bytes(img_bytes)

    # Prediksi
    preds = MODEL.predict(x)[0]  # shape: (num_classes,)
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx])

    # Buat dictionary probabilitas per kelas (opsional, bisa dihapus kalau nggak perlu)
    probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(preds)}

    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
    }
