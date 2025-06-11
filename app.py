import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="K-Means Clustering Produk", layout="wide")
st.title("\U0001F4E6 Segmentasi Produk Berdasarkan Penjualan")
st.caption("Analisis Klaster Produk | Algoritma: K-Means Clustering")

# Upload file
uploaded_file = st.file_uploader("\U0001F4C1 Upload file Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    st.subheader("\U0001F9FE Data Awal")
    st.dataframe(df.head())

    # Pilih kolom numerik yang digunakan
    fitur_numerik = ["Jumlah Order", "Harga", "Total Penjualan"]
    if not all(col in df.columns for col in fitur_numerik):
        st.error(f"❌ Kolom yang dibutuhkan tidak ditemukan. Harus ada: {fitur_numerik}")
    else:
        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[fitur_numerik])

        # Elbow method
        st.subheader("\U0001F4C9 Penentuan Jumlah Klaster (Elbow Method)")
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)

        fig_elbow, ax = plt.subplots()
        ax.plot(K, distortions, marker='o')
        ax.set_xlabel('Jumlah Klaster (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method untuk Menentukan k Optimal')
        st.pyplot(fig_elbow)

        # Pilih jumlah cluster
        k = st.slider("Pilih jumlah klaster", 2, 10, 3)

        # Jalankan K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df["Cluster"] = clusters

        # Visualisasi hasil clustering
        st.subheader("\U0001F4CA Hasil Clustering")
        fig_scatter, ax2 = plt.subplots()
        sns.scatterplot(
            x=df[fitur_numerik[0]],
            y=df[fitur_numerik[2]],
            hue=df["Cluster"],
            palette="viridis",
            ax=ax2
        )
        ax2.set_title("Visualisasi Cluster (Jumlah Order vs Total Penjualan)")
        st.pyplot(fig_scatter)

        # Tampilkan hasil
        st.subheader("\U0001F4CB Data dengan Klaster")
        st.dataframe(df)

        # Analisis per cluster
        st.subheader("\U0001F4C8 Rata-rata per Cluster")
        st.dataframe(df.groupby("Cluster")[fitur_numerik].mean())

else:
    st.info("Silakan upload file Excel dengan kolom: 'Jumlah Order', 'Harga', 'Total Penjualan'")
