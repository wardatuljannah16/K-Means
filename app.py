import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Risiko Pasien", layout="centered")
st.title("🎯 Prediksi Risiko Pasien Menggunakan Decision Tree")
st.caption("Dataset: Rumah Sakit XYZ | Sumber: Excel")

# Load data dari Excel
@st.cache_data
def load_data():
    df = pd.read_excel("data_pasien.xlsx", engine="openpyxl")  # <- sesuaikan nama file
    if "No" in df.columns:
        df = df.drop("No", axis=1)  # kolom No tidak dipakai jika ada
    return df

data = load_data()
st.subheader("📊 Data Pasien")
st.dataframe(data.head())

# Label Encoding
label_encoders = {}
df_encoded = data.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

# Split data
X = df_encoded.drop("Hasil", axis=1)
y = df_encoded["Hasil"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Akurasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"🎉 Akurasi Model: {acc*100:.2f}%")

# Visualisasi Decision Tree
st.subheader("🌳 Visualisasi Pohon Keputusan")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=["Tidak", "Ya"], filled=True, ax=ax)
st.pyplot(fig)

# Form prediksi manual
st.subheader("🔍 Prediksi Risiko Baru")

def user_input():
    input_data = []
    for col in X.columns:
        input_data.append(st.selectbox(col, data[col].unique()))
    df_input = pd.DataFrame([input_data], columns=X.columns)

    for col in df_input.columns:
        le = label_encoders[col]
        df_input[col] = le.transform(df_input[col])

    return df_input

input_df = user_input()
pred = model.predict(input_df)[0]
hasil_label = label_encoders["Hasil"].inverse_transform([pred])[0]

st.info(f"🧾 Prediksi Risiko Pasien: **{hasil_label}**")

# Tampilkan data asli + prediksi model
st.subheader("📋 Hasil Prediksi Data Asli")
df_encoded["Prediksi"] = model.predict(X)
df_encoded["Prediksi_Label"] = label_encoders["Hasil"].inverse_transform(df_encoded["Prediksi"])
st.dataframe(df_encoded)
