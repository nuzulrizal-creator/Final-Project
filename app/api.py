import os
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import engine, get_db, Base
from app.models_db import PredictionHistory

# Buat tabel di database jika belum ada
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Amazon Product Rating Prediction API",
    description="REST API untuk memprediksi kesuksesan produk Amazon (Rating >= 4.0) menggunakan model Deep Learning MLP.",
    version="1.0.0"
)

# Global variable untuk menampung model Keras
mlp_model = None

@app.on_event("startup")
def load_model():
    global mlp_model
    model_path_keras = os.path.join("models", "amazon_mlp_model.keras")
    model_path_h5 = os.path.join("models", "amazon_mlp_model.h5")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        if os.path.exists(model_path_keras):
            mlp_model = keras.models.load_model(model_path_keras)
            print(f"[INFO] Model berhasil dimuat dari {model_path_keras}")
        elif os.path.exists(model_path_h5):
            mlp_model = keras.models.load_model(model_path_h5)
            print(f"[INFO] Model berhasil dimuat dari {model_path_h5}")
        else:
            print("[WARNING] File model tidak ditemukan di folder models/.")
    except Exception as e:
        print(f"[WARNING] Gagal memuat model Keras: {e}")

# Pydantic Schemas
class ProductInput(BaseModel):
    product_name: str = Field("General Product", description="Nama produk")
    category: str = Field("General", description="Kategori produk")
    discounted_price: float = Field(..., gt=0, description="Harga setelah diskon")
    actual_price: float = Field(..., gt=0, description="Harga asli sebelum diskon")
    discount_percentage: float = Field(0.0, ge=0, le=100, description="Persentase diskon (0-100)")
    rating_count: int = Field(0, ge=0, description="Jumlah ulasan/rating")

class PredictionResponse(BaseModel):
    product_name: str
    category: str
    prediction_prob: float
    predicted_status: str
    recommendation: str

class HistoryResponse(BaseModel):
    id: int
    product_name: str
    category: str
    discounted_price: float
    actual_price: float
    discount_percentage: float
    rating_count: int
    prediction_prob: float
    predicted_status: str
    created_at: datetime

    class Config:
        from_attributes = True

@app.get("/")
def read_root():
    return {"message": "Selamat datang di Amazon Product Rating Prediction REST API! Kunjungi /docs untuk melihat dokumentasi interaktif (Swagger UI)."}

@app.post("/predict", response_model=PredictionResponse)
def predict_rating(product: ProductInput, db: Session = Depends(get_db)):
    global mlp_model
    prob = 0.5
    
    # Kalkulasi probabilitas dari Keras Model jika tersedia
    if mlp_model is not None:
        try:
            # Susun vektor input normalisasi sederhana berdasarkan fitur utama
            # Menyesuaikan dengan dimensi input yang diinginkan model
            input_dim = mlp_model.input_shape[-1] if mlp_model.input_shape else 4
            
            # Buat array fitur numeric dasar
            features = [
                product.discounted_price / 1000.0,
                product.actual_price / 1000.0,
                product.discount_percentage / 100.0,
                np.log1p(product.rating_count) / 10.0
            ]
            
            # Jika model membutuhkan lebih banyak fitur (misalnya one-hot encoding), pad dengan 0
            if len(features) < input_dim:
                features.extend([0.0] * (input_dim - len(features)))
            elif len(features) > input_dim:
                features = features[:input_dim]
                
            input_arr = np.array([features], dtype=np.float32)
            raw_pred = mlp_model.predict(input_arr, verbose=0)
            
            if isinstance(raw_pred, np.ndarray):
                prob = float(raw_pred.flatten()[0])
            else:
                prob = float(raw_pred)
        except Exception as e:
            print(f"[ERROR] Inference error: {e}. Menggunakan kalkulasi heuristik sebagai fallback.")
            # Fallback heuristik jika ada ketidakcocokan bentuk scaler
            prob = min(0.95, max(0.05, 0.5 + (product.discount_percentage / 200.0) + (min(product.rating_count, 5000) / 20000.0)))
    else:
        # Fallback heuristik jika model sedang tidak aktif / dalam pengujian
        prob = min(0.95, max(0.05, 0.5 + (product.discount_percentage / 200.0) + (min(product.rating_count, 5000) / 20000.0)))

    # Batasan threshold 0.5 untuk klasifikasi
    if prob >= 0.5:
        status = "SUKSES (Rating >= 4.0)"
        rec = "Produk berpotensi besar sukses di pasaran! Pertahankan strategi harga dan promosi saat ini."
    else:
        status = "KURANG SUKSES"
        rec = "Pertimbangkan kembali strategi harga jual, besaran diskon, atau tingkatkan kualitas deskripsi produk."

    # Simpan ke Database
    db_record = PredictionHistory(
        product_name=product.product_name,
        category=product.category,
        discounted_price=product.discounted_price,
        actual_price=product.actual_price,
        discount_percentage=product.discount_percentage,
        rating_count=product.rating_count,
        prediction_prob=prob,
        predicted_status=status
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return PredictionResponse(
        product_name=product.product_name,
        category=product.category,
        prediction_prob=prob,
        predicted_status=status,
        recommendation=rec
    )

@app.get("/history", response_model=List[HistoryResponse])
def get_prediction_history(limit: int = 50, db: Session = Depends(get_db)):
    records = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit).all()
    return records

@app.get("/stats")
def get_prediction_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(PredictionHistory.id)).scalar() or 0
    if total == 0:
        return {"total_predictions": 0, "success_rate": 0.0, "average_probability": 0.0}
    
    success_count = db.query(func.count(PredictionHistory.id)).filter(PredictionHistory.prediction_prob >= 0.5).scalar() or 0
    avg_prob = db.query(func.avg(PredictionHistory.prediction_prob)).scalar() or 0.0
    
    return {
        "total_predictions": total,
        "success_rate": round((success_count / total) * 100, 2),
        "average_probability": round(float(avg_prob) * 100, 2)
    }
