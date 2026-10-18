import os
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Proyek 3 Machine Learning",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Proyek 3 — Aplikasi Machine Learning")
st.write(
    "Aplikasi ini menyediakan eksperimen **Supervised Learning** untuk prediksi "
    "kepuasan pelanggan dan **Unsupervised Learning** untuk pengelompokan teks berita."
)

# Dataset dapat diletakkan di folder yang sama dengan app.py pada repositori GitHub.
SUPERVISED_PATH = "customer_satisfaction_data (1).csv"
UNSUPERVISED_PATH = "dataset_berita_cluster.csv"


def read_csv_safely(uploaded_file, default_path):
    """Baca file unggahan; jika tidak ada, coba baca dataset di repositori."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None


def make_one_hot_encoder():
    # Mendukung versi scikit-learn yang lebih baru maupun lama.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


tab_supervised, tab_unsupervised, tab_about = st.tabs(
    ["📈 Supervised Learning", "📰 Unsupervised Learning", "ℹ️ Tentang"]
)

with tab_supervised:
    st.header("Prediksi Skor Kepuasan Pelanggan")
    st.write(
        "Dataset yang disiapkan menggunakan kolom **Satisfaction Score** sebagai "
        "target numerik. Karena itu, eksperimen utama menggunakan model regresi."
    )

    supervised_upload = st.file_uploader(
        "Unggah dataset pelanggan CSV (opsional jika file sudah ada di GitHub)",
        type=["csv"],
        key="supervised_upload",
    )
    customer_df = read_csv_safely(supervised_upload, SUPERVISED_PATH)

    if customer_df is None:
        st.warning(
            "Dataset pelanggan belum ditemukan. Unggah file "
            "`customer_satisfaction_data (1).csv` di atas, atau tambahkan file itu "
            "ke repositori GitHub pada folder yang sama dengan app.py."
        )
    else:
        st.success(f"Dataset dimuat: {customer_df.shape[0]:,} baris dan {customer_df.shape[1]} kolom.")
        with st.expander("Lihat contoh data"):
            st.dataframe(customer_df.head(10), use_container_width=True)
            st.write("Jumlah nilai kosong per kolom:")
            st.dataframe(customer_df.isna().sum().rename("Nilai kosong"))

        target_options = [
            c for c in customer_df.columns
            if pd.api.types.is_numeric_dtype(customer_df[c])
        ]
        default_target = (
            target_options.index("Satisfaction Score")
            if "Satisfaction Score" in target_options
            else 0
        )

        if not target_options:
            st.error("Tidak ditemukan kolom target numerik untuk regresi.")
        else:
            target = st.selectbox(
                "Pilih kolom target yang ingin diprediksi",
                target_options,
                index=default_target,
            )
            exclude_id = st.checkbox(
                "Keluarkan kolom ID pelanggan jika tersedia",
                value=True,
                help="Kolom ID biasanya hanya penanda, bukan fitur prediksi.",
            )
            feature_candidates = [c for c in customer_df.columns if c != target]
            if exclude_id:
                feature_candidates = [
                    c for c in feature_candidates
                    if c.strip().lower() not in {"customer id", "customer_id", "id"}
                ]
            selected_features = st.multiselect(
                "Pilih fitur prediksi",
                feature_candidates,
                default=feature_candidates,
            )

            model_choice = st.selectbox(
                "Pilih algoritma regresi",
                ["Random Forest Regressor", "Linear Regression"],
            )
            test_size = st.slider(
                "Persentase data uji (%)", min_value=10, max_value=40, value=20, step=5
            )
            random_seed = st.number_input(
                "Random seed", min_value=0, max_value=9999, value=42, step=1
            )

            if st.button("🚀 Latih dan evaluasi model", key="train_supervised"):
                if not selected_features:
                    st.error("Pilih setidaknya satu fitur prediksi.")
                else:
                    data = customer_df[selected_features + [target]].copy()
                    data[target] = pd.to_numeric(data[target], errors="coerce")
                    data = data.dropna(subset=[target])

                    if len(data) < 10:
                        st.error("Data berlabel yang tersedia terlalu sedikit untuk dibagi menjadi data latih dan uji.")
                    else:
                        X = data[selected_features]
                        y = data[target]
                        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
                        categorical_features = [c for c in X.columns if c not in numeric_features]

                        numeric_pipeline = Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
