import streamlit as st
import pandas as pd

# Judul Aplikasi
st.title("Analisis Data Excel - Streamlit App")

# Upload file
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    # Membaca file Excel
    xls = pd.ExcelFile(uploaded_file)
    
    # Menampilkan daftar sheet
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Pilih Sheet", sheet_names)

    # Membaca sheet yang dipilih
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    st.write("Dataframe:")
    st.dataframe(df)

    # Menampilkan ringkasan statistik
    if st.checkbox("Tampilkan Ringkasan Statistik"):
        st.write(df.describe())

    # Fitur filter data
    st.subheader("Filter Data")
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_columns:
        col_to_filter = st.selectbox("Pilih Kolom Angka untuk Difilter", numeric_columns)
        min_val = float(df[col_to_filter].min())
        max_val = float(df[col_to_filter].max())
        user_range = st.slider("Pilih rentang nilai", min_val, max_val, (min_val, max_val))
        df_filtered = df[(df[col_to_filter] >= user_range[0]) & (df[col_to_filter] <= user_range[1])]
        st.write("Hasil Filter:")
        st.dataframe(df_filtered)
    else:
        st.write("Tidak ada kolom numerik yang tersedia untuk filter.")

else:
    st.info("Silakan upload file Excel untuk mulai.")
