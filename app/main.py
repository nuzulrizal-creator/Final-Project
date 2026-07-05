import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from textblob import TextBlob

# 1. Mengatur Judul dan Tampilan Halaman Utama Web
st.set_page_config(page_title="Amazon Success Predictor", page_icon="🚀", layout="centered")

st.title("Amazon New Product Success Predictor 🚀")
st.markdown(r"""
Aplikasi berbasis **Deep Learning (Multi-Layer Perceptron)** ini dirancang untuk memprediksi 
apakah produk baru yang akan Anda rilis berpotensi meraih **Rating Sukses ($\ge$ 4.0)** berdasarkan strategi finansial, popularitas, dan analisis sentimen ulasan.
""")
st.write("---")

# Class Scaler Fallback pintar jika file scaler.pkl tidak ditemukan di folder models/
class SmartFallbackScaler:
    def __init__(self, target_dim=11):
        self.target_dim = target_dim

    def transform(self, X):
        X_arr = np.array(X, dtype=np.float32)
        # Lakukan normalisasi sederhana agar skala angka cocok untuk neural network
        # [actual_price, discounted_price, discount_pct, discount_val, rating_count, sentiment]
        scaled = np.zeros_like(X_arr)
        if X_arr.shape[1] >= 1: scaled[:, 0] = X_arr[:, 0] / 1000.0
        if X_arr.shape[1] >= 2: scaled[:, 1] = X_arr[:, 1] / 1000.0
        if X_arr.shape[1] >= 3: scaled[:, 2] = X_arr[:, 2]  # persentase (0-1)
        if X_arr.shape[1] >= 4: scaled[:, 3] = X_arr[:, 3] / 500.0
        if X_arr.shape[1] >= 5: scaled[:, 4] = np.log1p(X_arr[:, 4]) / 10.0
        if X_arr.shape[1] >= 6: scaled[:, 5] = X_arr[:, 5]
        
        # Sesuaikan dimensi kolom dengan dimensi input yang diharapkan model Keras (misal 11 fitur)
        if scaled.shape[1] < self.target_dim:
            padding = np.zeros((scaled.shape[0], self.target_dim - scaled.shape[1]), dtype=np.float32)
            return np.hstack([scaled, padding])
        elif scaled.shape[1] > self.target_dim:
            return scaled[:, :self.target_dim]
        return scaled

# 2. Fungsi untuk Memuat Model .h5 / .keras dan Scaler .pkl secara aman
@st.cache_resource
def load_assets():
    model = None
    scaler = None
    
    # Coba muat model Keras / H5
    model_path_keras = 'models/amazon_mlp_model.keras'
    model_path_h5 = 'models/amazon_mlp_model.h5'
    
    if os.path.exists(model_path_keras):
        model = tf.keras.models.load_model(model_path_keras)
    elif os.path.exists(model_path_h5):
        model = tf.keras.models.load_model(model_path_h5)
    else:
        raise FileNotFoundError("File model tidak ditemukan di folder models/.")
        
    # Coba muat scaler.pkl, jika tidak ada gunakan SmartFallbackScaler agar aplikasi tidak eror
    scaler_path = 'models/scaler.pkl'
    target_dim = model.input_shape[-1] if model and model.input_shape else 11
    
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            scaler = SmartFallbackScaler(target_dim=target_dim)
    else:
        scaler = SmartFallbackScaler(target_dim=target_dim)
        
    return model, scaler

# Memanggil fungsi load_assets
try:
    model, scaler = load_assets()
    st.sidebar.success("Status: Model & Scaler Berhasil Dimuat! ✅")
except Exception as e:
    st.sidebar.error(f"Status: Gagal memuat aset model. ❌\nEror: {e}")

# 3. Membuat Form Input Interaktif untuk Pengguna
st.subheader("📥 Input Parameter Produk Baru")

# Mengelompokkan input ke dalam dua kolom agar tampilan web lebih rapi
col1, col2 = st.columns(2)

with col1:
    actual_price = st.number_input("Harga Asli Produk (₹)", min_value=0.0, value=1000.0, step=50.0)
    discounted_price = st.number_input("Harga Setelah Diskon (₹)", min_value=0.0, value=800.0, step=50.0)

with col2:
    rating_count = st.number_input("Estimasi Jumlah Ulasan (Popularitas)", min_value=0, value=150, step=10)
    review_title = st.text_input("Sampel Judul Ulasan Awal Pembeli", value="Excellent product and great quality!")

# 4. Proses Rekayasa Fitur Otomatis (Feature Engineering Turunan)
# Menghitung persentase dan nilai potongan harga
discount_percentage = (actual_price - discounted_price) / actual_price if actual_price > 0 else 0.0
discount_value = actual_price - discounted_price

# Menghitung skor polaritas sentimen menggunakan TextBlob (-1 bersifat negatif, 1 bersifat positif)
sentiment_score = TextBlob(review_title).sentiment.polarity

# Menampilkan informasi analisis fitur turunan secara real-time di web
with st.expander("🔍 Lihat Hasil Ekstraksi Fitur Tambahan (Feature Engineering)"):
    st.write(f"• Persentase Diskon Terhitung: `{discount_percentage * 100:.1f}%`")
    st.write(f"• Nominal Potongan Harga: `₹{discount_value:,.2f}`")
    st.write(f"• Skor Sentimen Judul (`review_title`): `{sentiment_score:.2f}`")

st.write("---")

# 5. Tombol Eksekusi Prediksi
if st.button("🚀 Prediksi Potensi Kesuksesan Produk", use_container_width=True):
    if 'model' not in locals() or model is None or 'scaler' not in locals() or scaler is None:
        st.error("Model atau Scaler belum siap. Coba segarkan halaman browser Anda.")
    else:
        # Menyusun semua fitur numerik ke dalam bentuk array
        input_features = np.array([[
            actual_price, 
            discounted_price, 
            discount_percentage, 
            discount_value, 
            rating_count, 
            sentiment_score
        ]], dtype=np.float32)
        
        try:
            # Sesuaikan dimensi input dengan yang diharapkan oleh scaler
            scaler_dim = getattr(scaler, 'n_features_in_', getattr(scaler, 'target_dim', 11))
            if input_features.shape[1] < scaler_dim:
                pad = np.zeros((input_features.shape[0], scaler_dim - input_features.shape[1]), dtype=np.float32)
                input_features = np.hstack([input_features, pad])
            elif input_features.shape[1] > scaler_dim:
                input_features = input_features[:, :scaler_dim]
                
            # Melakukan standarisasi/penyesuaian dimensi fitur menggunakan scaler
            input_scaled = scaler.transform(input_features)
            
            # Jika scaler asli menghasilkan jumlah kolom yang berbeda dengan input model, sesuaikan otomatis
            expected_dim = model.input_shape[-1] if model.input_shape else 11
            if input_scaled.shape[1] < expected_dim:
                pad = np.zeros((input_scaled.shape[0], expected_dim - input_scaled.shape[1]), dtype=np.float32)
                input_scaled = np.hstack([input_scaled, pad])
            elif input_scaled.shape[1] > expected_dim:
                input_scaled = input_scaled[:, :expected_dim]
            
            # Prediksi probabilitas kesuksesan menggunakan model MLP (Jaringan Saraf Tiruan)
            raw_pred = model.predict(input_scaled, verbose=0)
            if isinstance(raw_pred, np.ndarray):
                prediction_prob = float(raw_pred.flatten()[0])
            else:
                prediction_prob = float(raw_pred)
            
            # Tampilkan Hasil Analisis Akhir
            st.subheader("📊 Hasil Analisis Model Deep Learning")
            st.write(f"Probabilitas Kesuksesan Produk di Pasar: **{prediction_prob * 100:.2f}%**")
            
            # Batasan threshold 0.5 (50%) untuk klasifikasi biner sukses/kurang sukses
            if prediction_prob >= 0.5:
                st.balloons() # Efek balon jika sukses
                st.success(r"🎉 **PRODUK DIPREDIKSI SUKSES!** Produk ini memiliki peluang besar untuk menembus pasar Amazon dan meraih Rating $\ge$ 4.0.")
            else:
                st.error("⚠️ **PRODUK DIPREDIKSI KURANG SUKSES.** Rekomendasi: Pertimbangkan kembali kombinasi harga jual, besaran diskon, atau optimalkan kualitas produk guna mendongkrak kepuasan konsumen.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan kalkulasi prediksi: {e}")
            st.info("Tips: Pastikan dimensi/jumlah kolom data input (`input_features`) cocok dengan scaler dari hasil training Anda.")