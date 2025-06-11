import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analisis Data Excel & CSV", layout="wide")
st.title("📊 Analisis Data - Streamlit App")

# Upload file
uploaded_file = st.file_uploader("📁 Upload file Excel (.xlsx) atau CSV (.csv)", type=["xlsx", "csv"])

if uploaded_file is not None:
    file_name = uploaded_file.name

    try:
        # Cek ekstensi file dan baca data
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        st.success(f"✅ Berhasil membaca file: `{file_name}`")
        st.subheader("🧾 Data Awal:")
        st.dataframe(df)

        # Statistik Deskriptif
        if st.checkbox("📈 Tampilkan Ringkasan Statistik"):
            st.subheader("Ringkasan Statistik:")
            st.write(df.describe())

        # Filter kolom numerik
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols:
            st.subheader("🔍 Filter Data Berdasarkan Nilai")
            col_to_filter = st.selectbox("Pilih Kolom Numerik", numeric_cols)
            min_val = float(df[col_to_filter].min())
            max_val = float(df[col_to_filter].max())
            range_val = st.slider("Pilih Rentang Nilai", min_val, max_val, (min_val, max_val))
            df_filtered = df[(df[col_to_filter] >= range_val[0]) & (df[col_to_filter] <= range_val[1])]
            st.write(f"Hasil Filter pada kolom **{col_to_filter}**:")
            st.dataframe(df_filtered)
        else:
            st.info("Tidak ada kolom numerik untuk difilter.")

    except Exception as e:
        st.error(f"❌ Gagal membaca file. Error: {e}")

else:
    st.info("Silakan upload file terlebih dahulu.")
