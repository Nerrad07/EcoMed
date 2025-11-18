# backend.py
import io
import os
import uuid
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


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

# Folder untuk menyimpan gambar hasil kiriman bot (sementara)
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Mount folder gambar supaya bisa diakses dari web
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Menyimpan prediksi terakhir (global in-memory)
LAST_PREDICTION: Optional[Dict] = None


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
    Terima 1 file gambar dari Telegram bot,
    simpan hasil prediksi TERAKHIR di memori,
    lalu kembalikan label + confidence + probs.
    """
    global LAST_PREDICTION

    # Baca bytes gambar
    img_bytes = await file.read()

    # === 1) Preprocess untuk model ===
    x = preprocess_image_bytes(img_bytes)

    # === 2) Prediksi dengan model ===
    preds = MODEL.predict(x)[0]  # shape: (num_classes,)
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx])
    probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(preds)}

    # === 3) Simpan gambar ke disk supaya bisa ditampilkan di web ===
    # (pakai UUID supaya unik)
    image_id = str(uuid.uuid4())
    image_filename = f"{image_id}.jpg"
    image_path = os.path.join(IMAGES_DIR, image_filename)

    # Simpan sebagai JPEG
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image.save(image_path, format="JPEG", quality=85)

    image_url = f"/images/{image_filename}"

    # === 4) Simpan prediksi terakhir di memori ===
    LAST_PREDICTION = {
        "id": image_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "label": label,
        "confidence": confidence,
        "probs": probs,
        "image_url": image_url,
    }

    # === 5) Balas ke bot seperti biasa ===
    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
        "image_url": image_url,
    }

@app.get("/latest_prediction")
def latest_prediction() -> Dict:
    """
    Dikonsumsi oleh dashboard web.
    Mengembalikan prediksi terakhir (jika ada).
    """
    if LAST_PREDICTION is None:
        return {"has_prediction": False}

    return {
        "has_prediction": True,
        **LAST_PREDICTION,
    }
