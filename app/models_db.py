from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String, default="Produk Tanpa Nama")
    category = Column(String, default="General")
    discounted_price = Column(Float, default=0.0)
    actual_price = Column(Float, default=0.0)
    discount_percentage = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    
    # Hasil Prediksi Model
    prediction_prob = Column(Float, nullable=False)
    predicted_status = Column(String, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
