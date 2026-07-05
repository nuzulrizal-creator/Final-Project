import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from textblob import TextBlob

# 1. Mengatur Judul dan Tampilan Halaman Utama Web
st.set_page_config(page_title="Amazon Success Predictor", page_icon="🚀", layout="centered")

st.title("Amazon New Product Success Predictor 🚀")
st.markdown("""
Aplikasi berbasis **Deep Learning (Multi-Layer Perceptron)** ini dirancang untuk memprediksi 
apakah produk baru yang akan Anda rilis berpotensi meraih **Rating Sukses ($\ge$ 4.0)** berdasarkan strategi finansial, popularitas, dan analisis sentimen ulasan.
""")
st.write("---")

# 2. Fungsi untuk Memuat Model .h5 dan Scaler .pkl secara aman
@st.cache_resource
def load_assets():
    # Memuat arsitektur model neural network yang sudah dilatih
    model = tf.keras.models.load_model('models/amazon_mlp_model.h5')
    # Memuat scaler untuk menyamakan skala fitur numerik
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
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
    
    # Menyusun semua fitur numerik ke dalam bentuk array (pastikan urutannya sama persis seperti saat training)
    # Catatan: Jumlah input di array ini disesuaikan dengan fitur yang lolos uji overfitting kemarin
    input_features = np.array([[
        actual_price, 
        discounted_price, 
        discount_percentage, 
        discount_value, 
        rating_count, 
        sentiment_score
    ]])
    
    try:
        # Melakukan standarisasi skala data menggunakan scaler bawaan model
        input_scaled = scaler.transform(input_features)
        
        # Prediksi probabilitas kesuksesan menggunakan model MLP (Jaringan Saraf Tiruan)
        prediction_prob = model.predict(input_scaled)[0][0]
        
        # Tampilkan Hasil Analisis Akhir
        st.subheader("📊 Hasil Analisis Model Deep Learning")
        st.write(f"Probabilitas Kesuksesan Produk di Pasar: **{prediction_prob * 100:.2f}%**")
        
        # Batasan threshold 0.5 (50%) untuk klasifikasi biner sukses/kurang sukses
        if prediction_prob >= 0.5:
            st.balloons() # Efek balon jika sukses
            st.success("🎉 **PRODUK DIPREDIKSI SUKSES!** Produk ini memiliki peluang besar untuk menembus pasar Amazon dan meraih Rating $\ge$ 4.0.")
        else:
            st.error("⚠️ **PRODUK DIPREDIKSI KURANG SUKSES.** Rekomendasi: Pertimbangkan kembali kombinasi harga jual, besaran diskon, atau optimalkan kualitas produk guna mendongkrak kepuasan konsumen.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan kalkulasi prediksi: {e}")
        st.info("Tips: Pastikan dimensi/jumlah kolom data input (`input_features`) cocok dengan scaler dari hasil training Anda.")